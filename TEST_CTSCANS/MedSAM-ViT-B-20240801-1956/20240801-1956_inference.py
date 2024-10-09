# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
import wandb

from torchmetrics.utilities.data import to_categorical
from torchmetrics.utilities.distributed import reduce
from cal_dice import dice_score
from importlib import import_module
# set seeds
torch.manual_seed(2024)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="red", facecolor=(1, 1, 1, 0), lw=2)
    )


class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20, num_classes =4):
        self.data_root = data_root
        
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.num_classes = num_classes
        print("self.gt_path", self.gt_path)
        print("self.img_path", self.img_path)

        #self.gt_path_files = sorted(
        #    glob.glob(join(self.gt_path, "**/*.npy"), recursive=True)
        #)
        img_files = os.listdir(self.img_path)
        gt_files = os.listdir(self.gt_path)
        #print("all files", all_files)
        self.img_path_files = [
        os.path.join(self.img_path, file) for file in img_files
        if file.endswith('.npy')
        ]
        #print(self.img_path_files)

        self.gt_path_files = [
        os.path.join(self.gt_path, file) for file in gt_files
        if file.endswith('.npy')
        ]

        self.bbox_shift = bbox_shift
        print(f"number of images in img_path: {len(self.img_path_files)}")
        print(f"number of labels in gt_path: {len(self.gt_path_files)}")


    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.img_path_files[index])
        gt_pathname = self.gt_path + '/Final_' +img_name
        #print("img_name", img_name)
        
        #img_1024 here is the original size - changed code.
        img_1024 = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )  # (1024, 1024, 3)
        #print("image path name",join(self.img_path, img_name))
        #print("gt_path_name", gt_pathname)
        # convert the shape to (3, H, W)
        #img_1024 = np.transpose(img_1024, (2, 0, 1))
        
        # Preprocessing data.
        # Clip values: if greater than 80, set to 80; if below 0, set to 0
        img_1024 = np.clip(img_1024, 0, 1)
        #print("image range", img_1024.min(), img_1024.max())

        gt = np.load(
            gt_pathname, "r", allow_pickle=True
        )  # multiple labels [0, 1,4,5...], (256,256)
        
        #assert img_name == os.path.basename(self.gt_path_files[index]), (
        #    "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        #)
        
        label_ids = np.unique(gt)[1:]
        
        selected_label=3
        gt2D = np.uint8(gt == selected_label)  # only one label, (256, 256)
        gt2D = gt2D[1, ...]
        
        assert np.all(np.isin(np.unique(gt2D), [0, 1])), "Ground truth should be 0 or 1"
        y_indices, x_indices = np.where(gt2D ==1)
        
        #print("x_indices, y_indices ", x_indices, y_indices)
        if x_indices.size == 0 or y_indices.size == 0:
            x_min, x_max = 10, 502
            y_min, y_max = 10, 502
        else:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
        #print("x_min, x_max, y_min, y_max", x_min, x_max, y_min, y_max)
        
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        
        # Fix for now
        bboxes = np.array([10, 10, 502, 502])

        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt[None, 1, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )

# %% sanity test of dataset class


test_dir = "/data/home/umang/Vader_umang/Seg_models/data/CTScans/test_scans"
sample_list = [f for f in os.listdir(test_dir)]

sample0 = sample_list[0]

test_sample_dir = os.path.join(test_dir, sample0)
scan0_dataset = NpyDataset(test_sample_dir, num_classes=4)
scan0_dataloader = DataLoader(scan0_dataset, batch_size=8, shuffle=True)
    
for step, (image, gt, bboxes, names_temp) in enumerate(scan0_dataloader):
    print(image.shape, gt.shape, bboxes.shape)
    #print("image range", image.min().item(), image.max().item())
    
    # show the example
    _, axs = plt.subplots(1, 2, figsize=(25, 25))
    idx = random.randint(0, 7)
    selected_label =3
    gt2D = np.uint8(gt == selected_label)   # only one label, (256, 256)
    axs[0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    show_mask(gt2D[idx], axs[0])
    print("bboxes", bboxes[idx])
    show_box(bboxes[idx].numpy(), axs[0])
    axs[0].axis("off")
    
    # set title
    axs[0].set_title(names_temp[idx])
    idx = random.randint(0, 7)
    selected_label =3
    gt2D = np.uint8(gt == selected_label)   # only one label, (256, 256)
    axs[1].imshow(image[idx].cpu().permute(1, 2, 0).numpy())
    show_mask(gt2D[idx], axs[1])
    print("bboxes", bboxes[idx])
    show_box(bboxes[idx].numpy(), axs[1])
    axs[1].axis("off")
    
    # set title
    axs[1].set_title(names_temp[idx])
    
    # plt.show()
    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.savefig("./data_sanitycheck_testimgs.png", bbox_inches="tight", dpi=300)
    plt.close()
    break

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--test_npy_path",
    type=str,
    default='/data/home/umang/Vader_umang/Seg_models/data/CTScans/test_scans',
    help="path to training npy files; two subfolders: gts and imgs",
)
parser.add_argument("-task_name", type=str, default="MedSAM-ViT-B")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument(
    "-sam_checkpoint", type=str, default="/data/home/umang/Vader_umang/Seg_models/MedSAM/medsam_vit_b.pth")

parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="load pretrain model"
)
parser.add_argument(
     "-trained_model_path", type=str, default="/data/home/umang/Vader_umang/Seg_models/MedSAM/checkpoint_dir/MEDSAM_finetune_CT/MedSAM_finetune_CT-20240801-1152/medsam_model_latest.pth")
parser.add_argument('-device', type=str, default='cuda:7')

parser.add_argument("-work_dir", type=str, default="./work_dir")

# train
parser.add_argument("-num_epochs", type=int, default=1000)
parser.add_argument("-batch_size", type=int, default=2)
parser.add_argument("-num_workers", type=int, default=0)
# Optimizer parameters
parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
)
parser.add_argument(
    "-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)"
)
parser.add_argument(
    "-use_wandb", type=bool, default=False, help="use wandb to monitor training"
)
parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
parser.add_argument(
    "--resume", type=str, default="", help="Resuming training from checkpoint"
)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--num_classes", type=int, default="4")
parser.add_argument("--img_size", type=int, default=512)
parser.add_argument("--include_bg", type=bool, default=False, help='include bacground for seg loss calculation')
parser.add_argument("--dice_param", type=float, default=0.8, help="ratio of dice loss to ce loss")
parser.add_argument("--train_split_ratio", type= float, default=0.75, help="train test split ratio")

args = parser.parse_args()

# Define parameters for the run name
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
run_name = f"{args.task_name}_{args.model_type}_{timestamp}"

if args.use_wandb:

    wandb.login()
    wandb.init(
        project=args.task_name,
        name=run_name,
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "data_path": args.tr_npy_path,
            "model_type": args.model_type,
            "learning_rate": args.lr
        },
    )

# %% set up model for training
device = args.device
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join("TEST_CTSCANS", args.task_name + "-" + run_id)
device = torch.device(args.device)
# %% set up model

class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        # Freeze all layers of image_encoder, mask_decoder, and prompt_encoder
        self._freeze_parameters(self.image_encoder)
        self._freeze_parameters(self.mask_decoder)
        self._freeze_parameters(self.prompt_encoder)

    def _freeze_parameters(self, module):
            """Freeze all parameters of the given module."""
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, image, box):
        # print("image shape", image.shape)
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)
        
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, iou_pred = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=True,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks

def main():
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )

    sam_model = sam_model_registry[args.model_type](image_size=args.img_size,
                                                    num_classes=4,
                                                    checkpoint=args.sam_checkpoint, 
                                                    pixel_mean=[0, 0, 0],
                                                    pixel_std=[1, 1, 1])

    medsam_model = MedSAM(
        image_encoder= sam_model.image_encoder,
        mask_decoder= sam_model.mask_decoder,
        prompt_encoder= sam_model.prompt_encoder,
    ).to(device)

    pretrained_model = torch.load(args.trained_model_path, map_location=device)
    medsam_model.load_state_dict(pretrained_model["model"])
    medsam_model.train()

    print(
        "Number of total parameters: ",
        sum(p.numel() for p in medsam_model.parameters()),
    )  # 93735472
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in medsam_model.parameters() if p.requires_grad),
    )  # 93729252

    img_mask_encdec_params = list(medsam_model.image_encoder.parameters()) + list(
        medsam_model.mask_decoder.parameters()
    )
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    )
    print(
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
    )  # 93729252

    # %% test
    num_epochs = args.num_epochs
    iter_num = 0
    losses = []
    seg_losses =[]
    ce_losses =[]
    dice_scores = [[] for _ in range(args.num_classes)]

    mean_dice_scores =[]

    best_loss = 1e10
    test_dataset = NpyDataset(args.tr_npy_path, args.num_classes)

    sample_list = [f for f in os.listdir(test_dataset)]

    # Calculate the number of training and validation samples
    num_test = int(len(test_dataset))

    # Create indices for training and validation
    test_dataset = NpyDataset("/data/home/umang/Vader_umang/Seg_models/data/CTScans/test_scans", num_classes=4)
    sample_list = [f for f in os.listdir(test_dataset)]

    for num, sample_name in enumerate(sample_list):
        
        test_sample_dir = os.path.join(test_dataset, sample_name)
        test_dataloader = DataLoader(
            test_sample_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        if args.resume is not None:
            if os.path.isfile(args.resume):
                ## Map model to be loaded to specified single GPU
                checkpoint = torch.load(args.resume, map_location=device)
                medsam_model.load_state_dict(checkpoint["model"])
                optimizer.load_state_dict(checkpoint["optimizer"])
        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # Dice score segmentation loss.
        seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, to_onehot_y=False, reduction="mean", include_background = True)
        # cross entropy loss
        ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

        dice_param = args.dice_param
        
        for step, (image, gt2D, boxes, _) in enumerate(tqdm(test_dataloader)):
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)

            # Compute Dice scores
            with torch.no_grad():
                if args.use_amp:
                    ## AMP - not using for now
                    with torch.autocast(device_type="cuda:7", dtype=torch.float32):
                        medsam_pred = medsam_model(image, boxes_np)
                        loss = dice_param*seg_loss(medsam_pred, gt2D) \
                        + (1-dice_param)*ce_loss(medsam_pred, gt2D.float())
                else:
                    medsam_pred = medsam_model(image, boxes_np)
                    #print("medsam pred shape", medsam_pred.shape)
                    gt2D=gt2D.squeeze(dim=1)   # becomes [B,H,W]   
                    # Convert ground truth to one-hot format
                    gt2D_one_hot = F.one_hot(gt2D, num_classes=args.num_classes + 1).permute(0, 3, 1, 2).float()
                    relevant_classes_mask = gt2D_one_hot[:, 1:, :, :].float()  # Exclude background channel
                    
                    # Compute Dice loss for all classes
                    segmentation_loss = seg_loss(medsam_pred, gt2D_one_hot)

                    #segmentation_loss = seg_loss(medsam_pred, gt2D_expanded)     

                    # Cross entropy on all classes.
                    cross_entropy_loss = ce_loss(medsam_pred, gt2D_one_hot)
                    loss = dice_param * segmentation_loss + (1-dice_param)*cross_entropy_loss
                   
                    pred_probs = torch.sigmoid(medsam_pred)  # Get probabilities
                    dice_scores_batch, mean_dice_score = dice_score(pred_probs, gt2D, bg=True)
                    for cls in range(args.num_classes):
                        dice_scores[cls].append(dice_scores_batch[cls].item())  # Store individual Dice score for the batch
                    mean_dice_scores.append(mean_dice_score.item())  # Store average Dice score for the batch
                    print("dice_scores_batch", dice_scores_batch)


                # concatenate numpy predictions for each slice
                # Get argmax predictions
                pred_class = torch.argmax(pred_probs, dim=1)  # [B, H, W] or [B, D, H, W] depending on the network output
                
                # Convert predictions to numpy arrays and concatenate slices
                pred_class_np = pred_class.cpu().numpy()
                if step == 0:
                    all_preds = pred_class_np  # Initialize with first batch
                else:
                    all_preds = np.concatenate((all_preds, pred_class_np), axis=0)  # Concatenate along the batch dimension
                segment_loss += segmentation_loss.item()
                centropy_loss += cross_entropy_loss.item()
                loss += loss.item()
        
        if args.use_wandb:
            wandb.log({"loss per scan ": loss})
            wandb.log({"seg loss per scan": segment_loss})
            wandb.log({"cross entropy loss per scan": centropy_loss})
            wandb.log({"mean_dice_scores per scan": mean_dice_scores[-1]})  # Log the latest Dice score
            wandb.log({"ALL_dice_scores per slice": dice_scores[-1]})  # Log the latest Dice score

        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Seg Loss: {seg_loss}, CE_Loss: {ce_loss}, Loss: {loss}, 'f'Dice Score: {dice_scores[-1]}'
        )

        # Convert to NIfTI format
        nifti_img = nib.Nifti1Image(all_preds, np.eye(4))  # np.eye(4) is a dummy affine matrix
        nifti_img.to_filename(test_dataset+f"output_volume_{sample_name}.nii.gz")
        print(f"Saved volume {num} as output_volume_{sample_name}.nii.gz")


if __name__ == "__main__":
    main()

# %%
