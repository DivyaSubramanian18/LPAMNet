import warnings
warnings.filterwarnings("ignore", category=UserWarning)
%matplotlib inline

import os, cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
import random

from scipy.ndimage import distance_transform_edt, median_filter, binary_fill_holes, generate_binary_structure
from skimage.morphology import opening, closing, disk, remove_small_objects
from skimage.measure import label

from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy.ndimage import binary_dilation, binary_erosion

SEED = 247
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

TRAIN_IMG_DIR = r"/input/datasets/PolypMS/train/images/"
TRAIN_MASK_DIR = r"/input/datasets/PolypMS/train/masks/"

VAL_IMG_DIR = r"/input/datasets/PolypMS/val/images/"
VAL_MASK_DIR = r"/input/datasets/PolypMS/val/masks"

TEST_IMG_DIR = r"/input/datasets/PolypMS/test/images/"
TEST_MASK_DIR = r"/input/datasets/PolypMS/test/masks/"

IMG_SIZE = 352       
BATCH_SIZE = 4
EPOCHS = 40
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_VISUALIZE = 3

EVAL_RES = 704       
TTA_FLIPS = True

class PolypDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=352, augment=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment
        self.images = sorted(os.listdir(img_dir))
        self.masks  = sorted(os.listdir(mask_dir))

    def fault_augment(self, image):
        image = image.astype(np.float32)

        if np.random.rand() < 0.5:
            sigma = np.random.uniform(5, 25)   
            image += np.random.normal(0, sigma, image.shape).astype(np.float32)
        
        if np.random.rand() < 0.2:
            image = cv2.GaussianBlur(image, (5,5), 0)
        
        if np.random.rand() < 0.3:
            alpha = np.random.uniform(0.9, 1.1)
            beta = np.random.uniform(-10, 10)
            image = image * alpha + beta
        
        return np.clip(image, 0, 255).astype(np.uint8)

    def __getitem__(self, idx):
        # Load image & mask
        img = cv2.imread(os.path.join(self.img_dir, self.images[idx]))
        mask = cv2.imread(os.path.join(self.mask_dir, self.masks[idx]), 0)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)  
        
        if self.augment:
            img = self.fault_augment(img)
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=1)
                mask = np.flip(mask, axis=1)

            img=img.copy()
            mask = mask.copy()
        
        img = torch.from_numpy(img).float().permute(2,0,1)/255.0
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return img, mask

    def __len__(self):
        return len(self.images)
     
class DEM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,3,1)
        )

    def compute_cfi(self, x):
        gray = 0.299*x[:,0] + 0.587*x[:,1] + 0.114*x[:,2]
        gray = gray.unsqueeze(1)

        gray_smooth = F.avg_pool2d(gray, kernel_size=5, stride=1, padding=2)
        mean = F.avg_pool2d(gray_smooth, 3, 1, 1)
        var = F.avg_pool2d((gray_smooth - mean)**2, 3, 1, 1)
        std = torch.sqrt(var + 1e-6)
        max_std = std.flatten(2).max(dim=-1)[0].view(-1,1,1,1)

        cfi = 1 - std / (max_std + 1e-6)
        cfi = torch.clamp(cfi, 0.0, 1.0)
        return cfi

    def forward(self, x):
        feat = self.conv(x)
        cfi = self.compute_cfi(x)

        feat = feat * (1 + 0.5 * cfi.detach()) + 0.5 * feat
        out = F.avg_pool2d(feat, 3, 1, 1)
        return out, cfi

class ChannelAttention(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, ch//r, 1)
        self.fc2 = nn.Conv2d(ch//r, ch, 1)

    def forward(self, x):
        a = F.adaptive_avg_pool2d(x, 1)
        a = self.fc2(F.relu(self.fc1(a)))
        
        attn = torch.sigmoid(a)
        attn = torch.clamp(attn, 0.05, 0.95)  

        return x * attn + 0.1 * x

class LPAM(nn.Module):
    def __init__(self, ch, alpha=1.0):
        super().__init__()
        self.alpha = alpha

        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        )

        self.ca = ChannelAttention(ch)

    def forward(self, x, cfi):
        cfi = F.interpolate(
            cfi, size=x.shape[2:], mode="bilinear", align_corners=False
        )

        contrast_weight = 1.0 + self.alpha * cfi
        contrast_weight = torch.clamp(contrast_weight, 1.0, 2.0) 

        feat = self.conv(x)
        feat = self.ca(feat)

        out = x + feat * contrast_weight
        out = out + 0.05 * x  

        return out

class HiFiDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.up1 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)

        self.skip_conv = nn.Conv2d(320, 128, 1)

        self.refine = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.boundary_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Softplus()
        )

        self.aux = nn.Conv2d(128, 1, 1)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.drop = nn.Dropout2d(0.2)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x, skip):
        x = self.up1(x)
        x = self.up2(x)

        skip = self.skip_conv(skip)
        skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)

        x = x + skip
        x_skip = x

        x = self.refine(x)
        x = x + 0.5 * x_skip

        boundary_pred = self.boundary_head(x)


        gate = torch.sigmoid(5.0 * (1.0 - torch.clamp(boundary_pred, 0, 1)))  
        x = x * (1 + 0.5 * gate) + 0.05 * x  

        aux = self.aux(x)

        x = self.up3(x)
        x = self.drop(x)
        x = self.out(x)

        return x, aux, boundary_pred

class LPAMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dem = DEM()
        self.encoder = timm.create_model(
            "pvt_v2_b2", pretrained=True, features_only=True
        )
        self.lpam = LPAM(512)
        self.decoder = HiFiDecoder()

    def forward(self, x):
        x_stable, cfi = self.dem(x)
        feats = self.encoder(x_stable)
        
        deep = self.lpam(feats[-1], cfi)
        skip = feats[-2]
        
        pred, aux, boundary_pred = self.decoder(deep, skip)
        pred = F.interpolate(pred, size=x_stable.shape[2:], mode="bilinear", align_corners=False)
        aux = F.interpolate(aux, size=x_stable.shape[2:], mode="bilinear", align_corners=False)
        boundary_pred = F.interpolate(boundary_pred, size=x_stable.shape[2:], mode="bilinear", align_corners=False)
        
        return pred, aux, boundary_pred, cfi

bce = nn.BCEWithLogitsLoss()

class UncertaintyWeightedDiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, p, t):
        p_sig = torch.sigmoid(p)

        u = -(p_sig * torch.log(p_sig + self.eps) +
              (1 - p_sig) * torch.log(1 - p_sig + self.eps))
        

        boundary_weight = get_boundary_belt(t, width=3)
        
        w = 1.0 + u + (2.0 * boundary_weight) 

        inter = (p_sig * t * w).sum(dim=(1,2,3))
        denom = ((p_sig + t) * w).sum(dim=(1,2,3)) + self.eps
        return (1 - 2 * inter / denom).mean()


uw_dice = UncertaintyWeightedDiceLoss()

def distance_loss(pred_dt, gt_dt):
    weight = torch.exp(-5.0 * gt_dt) 
    loss = F.smooth_l1_loss(pred_dt, gt_dt, reduction='none')
    return (loss * weight).mean()


def ft_loss(pred, target):
    return 0.5 * bce(pred, target) + 0.5 * uw_dice(pred, target)

def get_boundary_belt(mask, width=3):
    kernel_size = width * 2 + 1
    padding = width
    
    dilation = F.max_pool2d(mask, kernel_size, stride=1, padding=padding)
    erosion = -F.max_pool2d(-mask, kernel_size, stride=1, padding=padding)
    
    return dilation - erosion
def boundary_distance_gt(mask, max_dist=50.0):
    mask_np = mask.cpu().numpy()
    dist_maps = []

    for m in mask_np:
        m = m[0]
        
        if m.sum() == 0:
            dist_maps.append(np.zeros_like(m, dtype=np.float32))
            continue

        dt = distance_transform_edt(1 - m).astype(np.float32)

        dt = np.clip(dt / max_dist, 0, 1.0)

        dist_maps.append(dt)

    return torch.from_numpy(np.stack(dist_maps)).unsqueeze(1).float().to(mask.device)

def dice_coef(p, g, boundary_weight=None):
    if boundary_weight is None:
        inter = (p * g).sum()
        denom = p.sum() + g.sum() + 1e-6
    else:
        inter = (p * g * boundary_weight).sum()
        denom = (p * boundary_weight).sum() + (g * boundary_weight).sum() + 1e-6

    return (2 * inter) / denom


def apply_fault(img, sigma=0):
    if sigma == 0:
        return img
    noise = torch.randn_like(img) * (sigma / 255.0)
    return torch.clamp(img + noise, 0, 1)

def compute_confusion(pred, gt):
    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)
    TP = np.logical_and(pred, gt).sum()
    FP = np.logical_and(pred, ~gt).sum()
    FN = np.logical_and(~pred, gt).sum()
    TN = np.logical_and(~pred, ~gt).sum()
    return TP, FP, FN, TN

def dice_score(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    return (2 * inter) / (pred.sum() + gt.sum() + 1e-6)

def iou_score(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter / (union + 1e-6)

def precision_score(pred, gt):
    TP, FP, _, _ = compute_confusion(pred, gt)
    return TP / (TP + FP + 1e-6)

def recall_score(pred, gt):
    TP, _, FN, _ = compute_confusion(pred, gt)
    return TP / (TP + FN + 1e-6)

def extract_boundary(mask, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel)
    boundary = mask.astype(np.uint8) - eroded
    return boundary

def assd(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    if not np.any(gt): 
        return 0.0
    if not np.any(pred):
        return 100.0
    dt_gt = distance_transform_edt(~gt)
    dt_pred = distance_transform_edt(~pred)

    pred_b = extract_boundary(pred_mask).astype(bool)
    gt_b = extract_boundary(gt_mask).astype(bool)
    
    s2g = dt_gt[pred_b].mean() 
    g2s = dt_pred[gt_b].mean() 
    return (s2g + g2s) / 2.0

def hd95(pred_mask, gt_mask, penalty=250): 
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    
    if not np.any(gt): return 0.0 
    if not np.any(pred): return penalty 
    
    dt_gt = distance_transform_edt(~gt)
    dt_pred = distance_transform_edt(~pred)

    pred_b = extract_boundary(pred_mask).astype(bool)
    gt_b = extract_boundary(gt_mask).astype(bool)
    
    if not np.any(pred_b) or not np.any(gt_b):
        return penalty

    s2g = dt_gt[pred_b]
    g2s = dt_pred[gt_b]
    all_dist = np.concatenate([s2g, g2s])
    return np.percentile(all_dist, 95)

def post_process(single_pred_probs, threshold=0.5):
    mask = (single_pred_probs > threshold).astype(np.uint8)
    

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1: return np.zeros_like(mask)
    
    refined_mask = np.zeros_like(mask)
    img_area = mask.shape[0] * mask.shape[1]
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > (img_area * 0.005):
            refined_mask[labels == i] = 1
            
    return refined_mask



def evaluate_metrics(val_loader, model, sigma=0):
    model.eval()
    metrics = {
        "Dice": [], "IoU": [], "Precision": [], "Recall": [], 
        "ASSD": [], "HD95": []
    }

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            imgs = apply_fault(imgs, sigma)

            # --- Test Time Augmentation: Flip ---
            p_orig, _, b_orig, _ = model(imgs)
            p_flip, _, b_flip, _ = model(torch.flip(imgs, [3]))
            pred = (p_orig + torch.flip(p_flip, [3])) / 2.0
            boundary_pred = (b_orig + torch.flip(b_flip, [3])) / 2.0

            # Upsample for evaluation
            eval_res = (EVAL_RES, EVAL_RES)
            pred = F.interpolate(pred, size=eval_res, mode="bicubic", align_corners=False)
            boundary_pred = F.interpolate(boundary_pred, size=eval_res, mode="bicubic", align_corners=False)
            masks_eval = F.interpolate(masks, size=eval_res, mode="nearest")

            # Boundary-guided refinement
            gate = torch.sigmoid(5.0 * (1.0 - torch.clamp(boundary_pred, 0, 1)))
            pred_refined = pred * (1 + 0.5 * gate)
            pred_probs = torch.sigmoid(pred_refined)

            target_size = masks.shape[2:]
            pred_probs_resized = F.interpolate(pred_probs, size=target_size, mode="bilinear", align_corners=False)

            for i in range(pred_probs_resized.shape[0]):
                single_pred = pred_probs_resized[i].squeeze().detach().cpu().numpy()
                pred_bin = post_process(single_pred)
                gt_np = masks[i][0].cpu().numpy() 
                if pred_bin.shape != gt_np.shape:
                    pred_bin = cv2.resize(pred_bin, (gt_np.shape[1], gt_np.shape[0]), interpolation=cv2.INTER_NEAREST)

                metrics["Dice"].append(dice_score(pred_bin, gt_np))
                metrics["IoU"].append(iou_score(pred_bin, gt_np))
                metrics["Precision"].append(precision_score(pred_bin, gt_np))
                metrics["Recall"].append(recall_score(pred_bin, gt_np))
                metrics["ASSD"].append(assd(pred_bin, gt_np))
                metrics["HD95"].append(hd95(pred_bin, gt_np))

    return {k: np.mean(v) for k,v in metrics.items()}

train_dataset = PolypDataset(img_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR, img_size=IMG_SIZE, augment=True)
val_dataset = PolypDataset(img_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR, img_size=IMG_SIZE, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

test_dataset = PolypDataset(img_dir=TEST_IMG_DIR, mask_dir=TEST_MASK_DIR, img_size=IMG_SIZE, augment=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


model = LPAMNet().to(DEVICE)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',      # HD95 should decrease
    factor=0.5,      # Reduce LR by half when plateau
    patience=5       # Wait 5 epochs before reducing
)


best_hd95 = float('inf')

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        pred, aux, boundary_pred, cfi = model(imgs)

        target_size = masks.shape[2:]
        pred = F.interpolate(pred, target_size, mode="bicubic", align_corners=False)
        aux = F.interpolate(aux, target_size, mode="bicubic", align_corners=False)
        boundary_pred = F.interpolate(boundary_pred, target_size, mode="bicubic", align_corners=False)

        gt_dt = boundary_distance_gt(masks).to(DEVICE)
        dt_loss = F.smooth_l1_loss(boundary_pred, gt_dt)

        gate = torch.sigmoid(5.0 *(1.0 - torch.clamp(boundary_pred, 0, 1)))
        pred_refined = pred * (1 + 0.5 * gate)

        with torch.no_grad():
            dist_penalty = 1.0 + (gt_dt * 20.0)
        
        belt = get_boundary_belt(masks)
        
        l_bce = F.binary_cross_entropy_with_logits(pred_refined, masks)
        l_edge = F.binary_cross_entropy_with_logits(pred_refined, masks, weight = belt)

        l_dice = uw_dice(pred_refined, masks)
        probs = torch.sigmoid(pred_refined)
        inter = (probs * masks).sum(dim=(1,2,3))
        union = (probs + masks).sum(dim=(1,2,3)) - inter
        l_iou = (1 - (inter + 1e-6) / (union + 1e-6)).mean()

        l_surface = (torch.sigmoid(pred_refined) * gt_dt).mean()


        loss_main = ((0.1 * l_bce) + (4.0 * l_edge) + (1.0 * l_dice) + (2.0 * l_surface))
        loss_aux = ft_loss(aux, masks)
        aux_weight = max(0.4 * (1 - epoch/EPOCHS), 0.1)
        loss = loss_main + (aux_weight * loss_aux) + (2.0 * dt_loss)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    metrics_clean = evaluate_metrics(val_loader, model, sigma=0)
    scheduler.step(metrics_clean['HD95'])
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch [{epoch+1}/{EPOCHS}] | LR: {current_lr:.6f} | "
          f"Dice: {metrics_clean['Dice']:.4f} | "
          f"HD95: {metrics_clean['HD95']:.4f} | "
          f"ASSD: {metrics_clean['ASSD']:.4f}")


    if metrics_clean['HD95'] < best_hd95:
        best_hd95 = metrics_clean['HD95']
        torch.save(model.state_dict(), "best_polyp_model.pth")
        print(f" New Record! HD95 dropped to {best_hd95:.4f}. Model saved.")

# Load the best model
model.load_state_dict(torch.load(r"/home/LPAMNet_checkpoints/best_polyp_model.pth", weights_only=True))
model.eval()

# Evaluate on clean validation data (sigma=0)
final_metrics = evaluate_metrics(test_loader, model, sigma=0)

# Print all metrics including boundary metrics
for k, v in final_metrics.items():
    print(f"{k}: {v:.4f}")

# Define noise levels for robustness testing
noise_levels = [0, 10, 20, 40, 80, 100]

# Print table header including all boundary metrics
print(f"\n{'Sigma':<6} | {'Dice':<6} | {'IoU':<6} | {'Precision':<9} | {'Recall':<6} | "
      f"{'ASSD':<6} | {'HD95':<6}")
print("-" * 100)

# Evaluate metrics under different noise levels
for sigma in noise_levels:
    results = evaluate_metrics(test_loader, model, sigma=sigma)
    print(f"{sigma:<6} | "
          f"{results['Dice']:.4f} | "
          f"{results['IoU']:.4f} | "
          f"{results['Precision']:.4f} | "
          f"{results['Recall']:.4f} | "
          f"{results['ASSD']:.2f} | "
          f"{results['HD95']:.2f} | "
          )





