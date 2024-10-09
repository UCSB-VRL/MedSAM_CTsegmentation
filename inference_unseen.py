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
from nph_utils.cal_dice import dice_score
from importlib import import_module
from scipy.ndimage import zoom
from preprocess_utils.preprocess_CT_only_scans import preprocess_CT_only_scans 
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
        self.img_path = join(data_root, "imgs")
        self.num_classes = num_classes
        
        print("self.img_path", self.img_path)

        img_files = [f for f in os.listdir(self.img_path) if f.endswith('.npy')]
        self.img_path_files = sorted(
            [os.path.join(self.img_path, file) for file in img_files],
            key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))))
        )
        
        self.bbox_shift = bbox_shift
        print(f"number of images in img_path: {len(self.img_path_files)}")

    def __len__(self):
        return len(self.img_path_files)

    def __getitem__(self, index):
        # load npy image (512, 512, 3), [0,1]
        img_name = os.path.basename(self.img_path_files[index])
        
        #img_1024 here is the original size - changed code.
        img_1024 = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )  # (1024, 1024, 3)

        img_1024 = np.clip(img_1024, 0, 1)

        # Fix for now
        bboxes = np.array(self.bbox_pt_lst)

        return (
            torch.tensor(img_1024).float(),
            torch.tensor(bboxes).float(),
            img_name,
        )
    
# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--test_dir",
    type=str,
    default='/data/home/umang/Vader_umang/Seg_models/data/CTScans/test_scans_unseen_',
    help="path to training npy files; two subfolders: gts and imgs",
)
parser.add_argument("--org_data_dir", type=str, default='/data/home/umang/Vader_umang/Seg_models/data/CTScans/Scans_org/Inital_scans')
parser.add_argument("--scans_test_save_path", type= str, default="/data/home/umang/Vader_umang/Seg_models/MedSAM/Inference_scans_unseen/changed_bbox/", help="give path to save scans")
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
device = torch.device(args.device)

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
    )  

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
    #seen_dir = '/data/home/umang/Vader_data/data/CTScans/skull_stripped_0.3_avail_annotations'
    sample_list = [f for f in os.listdir(args.test_dir) if f.endswith(('.nii.gz', '.nii'))]
    print("number of samples to be tested", len(sample_list))
    #print("sample list", sample_list)
    bbox_pt_lst = [10, 10, 502, 502]

    # CAll preprocessing function here
    if not os.path.exists(args.test_dir+'/preprocess_'+sample_list[0].rstrip('.nii.gz')):
        preprocess_CT_only_scans(args.test_dir)

    for num, sample_name in enumerate(sample_list):
        if sample_name.endswith('.nii.gz'):
            print("args test dir", args.test_dir)
            test_sample_dir = os.path.join(args.test_dir, 'preprocess_' + sample_name.rstrip('.nii.gz'))

            scan_dataset = NpyDataset(test_sample_dir, bbox_pt_lst=bbox_pt_lst, bbox_shift=20, num_classes =args.num_classes)
            test_dataloader = DataLoader(
                scan_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
                shuffle=False
            )
            for step, (image, boxes, img_name) in enumerate(tqdm(test_dataloader)):
                print("img name:", img_name)
                optimizer.zero_grad()
                boxes_np = boxes.detach().cpu().numpy()
                image = image.to(device)
                
                with torch.no_grad():
                    if args.use_amp:
                        ## AMP - not using for now
                        with torch.autocast(device_type="cuda:7", dtype=torch.float32):
                            medsam_pred = medsam_model(image, boxes_np)
                    else:
                        medsam_pred = medsam_model(image, boxes_np)
                        #print("medsam pred shape", medsam_pred.shape)
                        
                        pred_probs = torch.sigmoid(medsam_pred)  # Get probabilities
                        
                    # concatenate numpy predictions for each slice
                    # Get argmax predictions
                    pred_class = torch.argmax(pred_probs, dim=1)  # [B, H, W] or [B, D, H, W] depending on the network output
                    # Convert predictions to numpy arrays and concatenate slices
                    pred_class_np = pred_class.cpu().numpy()
                    print("Unique classes", np.unique(pred_class_np))
                    if step == 0:
                        all_preds = pred_class_np  # Initialize with first batch
                    else:
                        all_preds = np.concatenate((all_preds, pred_class_np), axis=0)  # Concatenate along the batch dimension
                    
            os.makedirs(args.scans_test_save_path, exist_ok=True)
            # Convert to NIfTI format
            # Open the corresponding original scan

            org_data_dir= args.org_data_dir
            print("org_Data_Dir", org_data_dir)
            # load scan
            org_Scan = nib.load(os.path.join(org_data_dir, sample_name))
            # Extract the image data and affine matrix
            img_data = org_Scan.get_fdata()  # or nifti_image.get_data() in older nibabel versions
            header= org_Scan.header
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
            
            # Get original data type
            original_dtype = img_data.dtype  # Get the original data type

            nifti_img = nib.Nifti1Image(resized_preds, affine, header=header)  # np.eye(4) is a dummy affine matrix
            nifti_img.set_data_dtype(original_dtype)  # Ensure the data type is set to the original type
            nifti_img.to_filename(args.scans_test_save_path+f"Segmentation_{sample_name}")
            print(f"Saved volume as Segmentation_{sample_name}.nii.gz")

            # Check segmentation integrity
            temp_check = nib.load(args.scans_test_save_path+f"Segmentation_{sample_name}")
            print("loaded segmentation img dimensions:", temp_check.get_fdata().shape)
            print("loaded segmentation affine", temp_check.affine)
            if temp_check.get_fdata().shape != img_shape or not np.allclose(temp_check.affine, affine):
                print("mismatch!")
                exit(0)
            

if __name__ == "__main__":
    main()

# %%
