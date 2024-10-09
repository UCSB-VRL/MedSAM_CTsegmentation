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
import re

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
from scipy.ndimage import zoom

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
    def __init__(self, data_root, bbox_pt_lst = [10, 10, 502, 502], bbox_shift=20, num_classes =4):
        self.data_root = data_root
        self.bbox_pt_lst = bbox_pt_lst
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.num_classes = num_classes
        print("self.gt_path", self.gt_path)
        print("self.img_path", self.img_path)

        #self.gt_path_files = sorted(
        #    glob.glob(join(self.gt_path, "**/*.npy"), recursive=True)
        #)
        # List and sort image files numerically
        # List and sort image files numerically
        img_files = [f for f in os.listdir(self.img_path) if f.endswith('.npy')]
        self.img_path_files = sorted(
            [os.path.join(self.img_path, file) for file in img_files],
            key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))))
        )

        # List and sort ground truth files numerically
        gt_files = [f for f in os.listdir(self.gt_path) if f.endswith('.npy')]
        self.gt_path_files = sorted(
            [os.path.join(self.gt_path, file) for file in gt_files],
            key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))))
        )
        
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
        
        # Fix for now.
        bboxes = np.array(self.bbox_pt_lst)

        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt[None, 1, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--test_dir",
    type=str,
    default='/data/home/umang/Vader_umang/Seg_models/data/CTScans/test_scans',
    help="path to training npy files; two subfolders: gts and imgs",
)
parser.add_argument("--scans_test_save_path", type= str, default="/data/home/umang/Vader_umang/Seg_models/MedSAM/Inference_scans/temp/", help="give path to save scans")
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
parser.add_argument("-batch_size", type=int, default=1)
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
parser.add_argument("--num_classes", type=int, default=4)
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
                                                    num_classes=args.num_classes,
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

    best_loss = 1e10
    import csv
    # Define the path to the CSV file
    csv_file_path = args.scans_test_save_path+'log_file.csv'
    
    # Open the CSV file in write mode and write the header
    # Define the number of classes

    # Check if CSV file exists to decide if headers need to be written
    file_exists = not os.path.exists(csv_file_path)

    # Open the CSV file in append mode
    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = [
            'Image Name', 'Seg Loss of Scan', 'CE Loss of Scan', 'Loss of Scan',
            'Mean Dice Scores of Scan'
        ] + [f'Dice Score Class {i}' for i in range(args.num_classes+1)]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if file_exists:
            writer.writeheader()
           
    PER_CLASS_SEG_LOSS=[]
    PER_CLASS_CE_LOSS=[]
    PER_CLASS_LOSS=[]
    PER_CLASS_DICE_SCORE = []
    sample_list = [f for f in os.listdir(args.test_dir)]
    
    for num, sample_name in enumerate(sample_list):
        print("sample_name", sample_name)
        test_sample_dir = os.path.join(args.test_dir, sample_name)
        scan_dataset = NpyDataset(test_sample_dir, num_classes=4, bbox_pt_lst=bbox_pt_lst)
        test_dataloader = DataLoader(
            scan_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False
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
        scan_seg_loss=0
        scan_ce_loss=0
        net_loss=0

        list_perclass_dice_scores_batch=[]
        mean_dice_scores =[]
        for step, (image, gt2D, boxes, img_name) in enumerate(tqdm(test_dataloader)):
            print("img name:", img_name)
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
                    
                    # Compute Dice loss for all classes
                    segmentation_loss = seg_loss(medsam_pred, gt2D_one_hot)  

                    # Cross entropy on all classes.
                    cross_entropy_loss = ce_loss(medsam_pred, gt2D_one_hot)
                    loss = dice_param * segmentation_loss + (1-dice_param)*cross_entropy_loss
                   
                    pred_probs = torch.sigmoid(medsam_pred)  # Get probabilities
                    dice_scores_batch, mean_dice_score = dice_score(pred_probs, gt2D, bg=True)
                    for cls in range(args.num_classes):
                        dice_scores[cls].append(dice_scores_batch[cls].item())  # Store individual Dice score for the batch
                    mean_dice_scores.append(mean_dice_score.item())  # Store average Dice score for the batch
                    list_perclass_dice_scores_batch.append(dice_scores_batch.detach().cpu().numpy())
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
                scan_seg_loss += segmentation_loss.item()
                scan_ce_loss += cross_entropy_loss.item()
                net_loss += loss.item()
        
        scan_per_class_dice_Score = mean_excluding_below_threshold_axis(list_perclass_dice_scores_batch, 0.01, axis=0)
        
        PER_CLASS_SEG_LOSS.append(scan_seg_loss)
        PER_CLASS_CE_LOSS.append(scan_ce_loss)
        PER_CLASS_LOSS.append(net_loss)
        PER_CLASS_DICE_SCORE.append(scan_per_class_dice_Score)

        print(
            f'Image Name: {img_name}, Seg Loss of scan: {scan_seg_loss}, CE_Loss of scan: {scan_ce_loss}, Loss of scan: {loss}, 'f'mean_dice_scores of scan: {np.mean(mean_dice_scores)}, 'f'Dice Score each class of scan: {scan_per_class_dice_Score}'
        )

        csv_entry = {
            'Image Name': img_name,
            'Seg Loss of Scan': scan_seg_loss,
            'CE Loss of Scan': scan_ce_loss,
            'Loss of Scan': loss,
            'Mean Dice Scores of Scan': np.mean(mean_dice_scores),
        }

        # Add Dice scores for each class to the CSV entry
        for i in range(args.num_classes+1):
            csv_entry[f'Dice Score Class {i}'] = scan_per_class_dice_Score[i]

        # Save to CSV file
        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(csv_entry)

        os.makedirs(args.scans_test_save_path, exist_ok=True)
        # Convert to NIfTI format
        # Open the corresponding original scan and the gt truth

        org_data_dir='/data/home/umang/Vader_data/data/CTScans/Scans_org/Initial_Scans'
        # load scan
        org_Scan = nib.load(os.path.join(org_data_dir, sample_name+'.nii.gz'))
        # Extract the image data and affine matrix
        img_data = org_Scan.get_fdata()  # or nifti_image.get_data() in older nibabel versions
        affine = org_Scan.affine
        img_shape = img_data.shape

        all_preds = np.transpose(all_preds, (1, 2, 0))
        print("Shape of all preds", all_preds.shape)
        # Compute the zoom factors for height and width
        zoom_factors = np.array(img_shape[:2]) / np.array(all_preds.shape[:2])
        # Resize predictions
        resized_preds = np.zeros((*img_shape[:2], all_preds.shape[2]))
        
        for i in range(all_preds.shape[2]):
            resized_preds[:, :, i] = zoom(all_preds[:, :, i], zoom_factors, order=1)  # 'order' defines interpolation method
        
        nifti_img = nib.Nifti1Image(resized_preds, affine)  # np.eye(4) is a dummy affine matrix
        nifti_img.to_filename(args.scans_test_save_path+f"Segmentation_{sample_name}.nii.gz")
        print(f"Saved volume {num} as Segmentation_{sample_name}.nii.gz")

if __name__ == "__main__":
    main()

# %%
