import numpy as np
import os
from pathlib import Path
from PIL import Image
import cv2
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.metrics.functional import iou_score, get_stats

# -------------------- device & seed --------------------
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")
print(f"Using device: {device}")

SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)

CLASS_MAP = {"background": 0, "nerve": 1}

# -------------------- Dataset --------------------
class SegDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, multiclass=False):
        self.image_paths = list(image_paths)
        self.mask_paths  = list(mask_paths)
        assert len(self.image_paths) == len(self.mask_paths), "images/masks count mismatch"
        self.transform = transform
        self.multiclass = multiclass

    def __len__(self): 
        return len(self.image_paths)

    def __getitem__(self, i):
        img_path = str(self.image_paths[i])
        mask_path = str(self.mask_paths[i])

        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"cv2.imread failed for mask: {mask_path}")
        # mask: uint8 [H,W] in {0,255} -> {0,1}
        mask = (mask > 0).astype(np.uint8)

        if self.transform:
            try:
                out = self.transform(image=image, mask=mask)
            except Exception as e:
                raise RuntimeError(f"Augment failed for\n image: {img_path}\n mask: {mask_path}\n error: {e}")
            image, mask = out["image"], out["mask"]

        mask = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)  # [1,H,W]
        return image, mask

# -------------------- utils --------------------
def check_data_dirs(data_dir: Path):
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    masks_dir  = data_dir / "masks"
    models_dir = Path("./models").resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")
    for p in [images_dir, labels_dir, masks_dir]:
        if not p.exists():
            raise FileNotFoundError(f"{p} does not exist.")

    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {data_dir}")
    print(f"Images directory: {images_dir}")
    print(f"Labels directory: {labels_dir}")
    print(f"Masks directory: {masks_dir}")
    print(f"Models directory: {models_dir}")

    return images_dir, labels_dir, masks_dir, models_dir

def check_img_size(images_dir: Path, masks_dir: Path):
    image_shapes = []
    mask_shapes = []

    for image in images_dir.glob("*.png"):
        img = np.array(Image.open(image))
        image_shapes.append(img.shape[:2])  # (h, w)

    for mask in masks_dir.glob("*.png"):
        m = np.array(Image.open(mask))
        mask_shapes.append(m.shape[:2])  # (h, w)

    unique_image_shapes = set(image_shapes)
    unique_mask_shapes  = set(mask_shapes)

    if len(unique_image_shapes) > 1:
        print("⚠️ Not all images have the same size:", unique_image_shapes)
    if len(unique_mask_shapes) > 1:
        print("⚠️ Not all masks have the same size:", unique_mask_shapes)

    if unique_image_shapes != unique_mask_shapes:
        print("⚠️ Image and mask sizes differ.")
    else:
        print("✅ All images and masks have the same size:", unique_image_shapes.pop())

def split_data(images, masks):
    train_imgs, valtest_imgs, train_masks, valtest_masks = train_test_split(
        images, masks, test_size=0.3, random_state=SEED, shuffle=True
    )
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(
        valtest_imgs, valtest_masks, test_size=0.5, random_state=SEED, shuffle=True
    )
    print(len(train_imgs), len(val_imgs), len(test_imgs))
    return train_imgs, val_imgs, test_imgs, train_masks, val_masks, test_masks

@torch.no_grad()
def batch_iou(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    # logits: [B,1,H,W]; targets: [B,1,H,W] float in {0,1}
    preds = (torch.sigmoid(logits) > thr)  # bool [B,1,H,W]
    preds = preds[:, 0].to(torch.long)     # [B,H,W] long
    t = targets[:, 0].to(torch.long)       # [B,H,W] long
    tp, fp, fn, tn = get_stats(preds, t, mode="binary", threshold=None)
    return iou_score(tp, fp, fn, tn, reduction="micro")

def run_epoch(model, loader, train: bool, loss_fn=None, optimizer=None) -> Tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    tot_loss, tot_iou, n = 0.0, 0.0, 0
    for imgs, masks in loader:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)  # [B,1,H,W]

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(imgs)               # [B,1,H,W]
        loss   = loss_fn(logits, masks)    # DiceLoss expects float targets
        iou    = batch_iou(logits, masks)

        if train:
            loss.backward()
            optimizer.step()

        bs = imgs.size(0)
        tot_loss += loss.item() * bs
        tot_iou  += iou.item()  * bs
        n += bs

    return tot_loss / n, tot_iou / n

def create_cache_dirs():
    hf_cache    = Path("./.cache/hf").resolve()
    torch_cache = Path("./.cache/torch").resolve()
    hf_cache.mkdir(parents=True, exist_ok=True)
    torch_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"]    = str(hf_cache)
    os.environ["TORCH_HOME"] = str(torch_cache)

def load_test_model(model_path: str, test_loader: DataLoader):
    model = smp.Linknet(encoder_name="resnet34", encoder_weights=None,
                        in_channels=3, classes=1).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    loss_fn = smp.losses.DiceLoss(mode="binary")
    te_loss, te_iou = run_epoch(model, test_loader, train=False, loss_fn=loss_fn, optimizer=None)
    print(f"TEST | loss {te_loss:.4f} IoU {te_iou:.3f}")
    

def build_transforms():
    # ภาพคุณ 648x864 → 648 ไม่หาร 32 ลงตัว จึง pad สูงเป็น 672 (ต่อให้กว้าง 864 ก็ใส่ divisor ไว้ได้)
    pad_to_32 = A.PadIfNeeded(
        pad_height_divisor=32,
        pad_width_divisor=32,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        mask_value=0,
        position='top_left'
    )

    train_tf = A.Compose([
        pad_to_32,
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.10, rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    eval_tf = A.Compose([
        pad_to_32,
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return train_tf, eval_tf

def main():
    create_cache_dirs()

    data_dir = Path("./dataset").resolve()
    images_dir, labels_dir, masks_dir, models_dir = check_data_dirs(data_dir)
    check_img_size(images_dir, masks_dir)

    images = sorted(images_dir.glob("*.png"))
    masks  = sorted(masks_dir.glob("*.png"))
    train_imgs, val_imgs, test_imgs, train_masks, val_masks, test_masks = split_data(images, masks)

    train_tf, eval_tf = build_transforms()

    train_ds = SegDataset(train_imgs, train_masks, transform=train_tf,  multiclass=False)
    val_ds   = SegDataset(val_imgs,   val_masks,   transform=eval_tf,   multiclass=False)
    test_ds  = SegDataset(test_imgs,  test_masks,  transform=eval_tf,   multiclass=False)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

    model = smp.Linknet(encoder_name="resnet34", encoder_weights="imagenet",
                        in_channels=3, classes=1).to(device)
    
    loss_fn = smp.losses.DiceLoss(mode="binary") + smp.losses.SoftBCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = float("inf")
    for epoch in range(1, 21):
        tr_loss, tr_iou = run_epoch(model, train_loader, train=True,  loss_fn=loss_fn, optimizer=optimizer)
        va_loss, va_iou = run_epoch(model, val_loader,   train=False, loss_fn=loss_fn, optimizer=None)
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} IoU {tr_iou:.3f} | "
              f"val loss {va_loss:.4f} IoU {va_iou:.3f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), str(models_dir / "linknet_best.pt"))

    # optional: test at the end
    load_test_model(str(models_dir / "linknet_best.pt"), test_loader)

if __name__ == "__main__":
    main()