# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import numpy as np
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
from nph_utils.cal_dice import dice_score

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
# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--tr_npy_path",
    type=str,
    default='/data/home/umang/Vader_umang/Seg_models/data/CTScans/preprocessed_RGB',
    help="path to training npy files; two subfolders: gts and imgs",
)
parser.add_argument("-task_name", type=str, default="MedSAM-ViT-B")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument(
    "-checkpoint", type=str, default="/data/home/umang/Vader_umang/Seg_models/MedSAM/sam_vit_b_01ec64.pth")
parser.add_argument('-device', type=str, default='cuda:7')
parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="load pretrain model"
)
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir")
parser.add_argument("-checkpoint_dir", type=str, default="./checkpoint_dir")
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

run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.checkpoint_dir, args.task_name + "-" + run_id)
torch.cuda.empty_cache()
print(torch.cuda.device_count())  # Number of GPUs available
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")
device = torch.device("cuda:"+args.device)
print("device:", device)
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
        # freeze prompt encoder
        #for param in self.mask_decoder.parameters():
        #    param.requires_grad = False
        
        for param in self.prompt_encoder.parameters():
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


# sanity test of dataset class
    tr_dataset = NpyDataset(args.tr_npy_path, args.num_classes)
    tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True)
    for step, (image, gt, bboxes, names_temp) in enumerate(tr_dataloader):
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
        plt.savefig("./data_sanitycheck.png", bbox_inches="tight", dpi=300)
        plt.close()
        break

    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )

    sam_model = sam_model_registry[args.model_type](image_size=args.img_size,
                                                    num_classes=args.num_classes,
                                                    checkpoint=args.checkpoint, 
                                                    pixel_mean=[0, 0, 0],
                                                    pixel_std=[1, 1, 1])

    medsam_model = MedSAM(
        image_encoder= sam_model.image_encoder,
        mask_decoder= sam_model.mask_decoder,
        prompt_encoder= sam_model.prompt_encoder,
    ).to(device)

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

    # %% train
    num_epochs = args.num_epochs
    iter_num = 0
    losses = []
    seg_losses =[]
    ce_losses =[]

    mean_dice_scores =[]

    best_loss = 1e10
    train_dataset = NpyDataset(args.tr_npy_path, args.num_classes)

    # Calculate the number of training and validation samples
    num_train = int(len(train_dataset))

    # Create indices for training and validation
    indices = list(range(len(train_dataset)))
    train_indices = indices[:num_train]
    valid_indices = indices[num_train:]
    print("Total number of samples: ", len(train_dataset))
    print("Number of training samples: ", len(train_indices))

    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler= train_sampler,
        pin_memory=True,
    )
    
    valid_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=valid_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            medsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # Dice score segmentation loss.
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, to_onehot_y=False, reduction="mean", include_background = True)
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    dice_param = args.dice_param
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        epoch_seg_loss = 0
        epoch_ce_loss = 0
        dice_scores = [[] for _ in range(args.num_classes+1)]
        
        for step, (image, gt2D, boxes, _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)
            if args.use_amp:
                ## AMP - not using for now
                with torch.autocast(device_type="cuda:7", dtype=torch.float32):
                    medsam_pred = medsam_model(image, boxes_np)
                    loss = dice_param*seg_loss(medsam_pred, gt2D) \
                    + (1-dice_param)*ce_loss(medsam_pred, gt2D.float())

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                medsam_pred = medsam_model(image, boxes_np)
                #print("medsam pred shape", medsam_pred.shape)
                gt2D =gt2D.squeeze(dim=1)   # becomes [B,H,W]   
                # Convert ground truth to one-hot format
                gt2D_one_hot = F.one_hot(gt2D, num_classes=args.num_classes + 1).permute(0, 3, 1, 2).float()
                
                # Select only the relevant classes for loss calculation
                # Create mask for relevant classes (excluding background class index 0)
                relevant_classes_mask = gt2D_one_hot[:, 1:, :, :].float()  # Exclude background channel
                #print("relevant classes at dim 1 example", relevant_classes_mask[:,:,250,250])
                # Ensure preds and gt2D_one_hot align for relevant classes (excluding background here)

                # Compute Dice loss. Here the computation is done considering that background is already excluded
                # Relevant classes mask is one hot encoding containing non-bavkgorund classes.
                # Medsam pred only contains logits for all classes. classes.
                # So include_bg is set True here because class 0 in the
                #  computation here is ventricles - orginally class1.
                segmentation_loss = seg_loss(medsam_pred[:, 1:, :, :], relevant_classes_mask)

                #segmentation_loss = seg_loss(medsam_pred, gt2D_expanded)     

                # Cross entropy on all classes.
                cross_entropy_loss = ce_loss(medsam_pred, gt2D_one_hot)
                loss = dice_param * segmentation_loss + (1-dice_param)*cross_entropy_loss
                #print("seg loss, ce_loss, loss", segmentation_loss, cross_entropy_loss, loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                #print("medsam pred unique gt2D unique", np.unique(torch.sigmoid(medsam_pred).detach().cpu().numpy()), np.unique(gt2D.detach().cpu().numpy()))

                # Compute Dice scores
                with torch.no_grad():
                    pred_probs = torch.sigmoid(medsam_pred)  # Get probabilities
                    dice_scores_batch, mean_dice_score = dice_score(pred_probs, gt2D, bg=True)
                    for cls in range(args.num_classes+1):
                        dice_scores[cls].append(dice_scores_batch[cls].item())  # Store individual Dice score for the batch
                    mean_dice_scores.append(mean_dice_score.item())  # Store average Dice score for the batch
                    print("dice_scores_batch", dice_scores_batch)

            epoch_seg_loss += segmentation_loss.item()
            epoch_ce_loss += cross_entropy_loss.item()
            epoch_loss += loss.item()
            iter_num += 1

        print("total epoch loss", epoch_loss)
        epoch_seg_loss /= step
        epoch_ce_loss /= step
        epoch_loss /= step
        losses.append(epoch_loss)
        seg_losses.append(seg_loss)
        ce_losses.append(ce_loss)
        
        if args.use_wandb:
            wandb.log({"epoch_loss per slice ": epoch_loss})
            wandb.log({"segemntation epoch_loss per slice": epoch_seg_loss})
            wandb.log({"CE epoch_loss per slice": epoch_ce_loss})
            wandb.log({"mean_dice_scores per slice for class 1": np.mean(dice_scores[1])})
            wandb.log({"mean_dice_scores per slice for class 2": np.mean(dice_scores[2])})
            wandb.log({"mean_dice_scores per slice for class 3": np.mean(dice_scores[3])})
            wandb.log({"mean_dice_scores per slice for class 4": np.mean(dice_scores[4])})
            wandb.log({"mean_dice_scores per slice every epoch": np.mean(mean_dice_scores)})  # Log the latest Dice score
            #wandb.log({"ALL_dice_scores per slice": dice_scores[-1]})  # Log the latest Dice score

        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Seg Loss: {epoch_seg_loss}, CE_Loss: {epoch_ce_loss}, Loss: {epoch_loss}, 'f'Dice Score: {np.mean(dice_scores, axis=1)}')

        # Save the latest model
        checkpoint = {
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, args.task_name+"_model_latest.pth"))

        # Save the best model if applicable
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, args.task_name+"_model_best.pth"))

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, args.task_name+f"_model_epoch_{epoch + 1}.pth"))

    # Ensure lengths match
    assert len(losses) == len(seg_losses) == len(ce_losses), "All loss lists must be of the same length."

    # Plotting all three losses
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Total Loss", color="blue")
    plt.plot(seg_losses, label="Dice Segmentation Loss", color="green")
    plt.plot(ce_losses, label="Cross Entropy Loss", color="red")
    #plt.plot(dice_scores, label="Dice Score for all classes", color="orange")
    
    plt.title("Losses Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(join(model_save_path, args.task_name + "_train_loss.png"))
    plt.close()


if __name__ == "__main__":
    main()

# %%
