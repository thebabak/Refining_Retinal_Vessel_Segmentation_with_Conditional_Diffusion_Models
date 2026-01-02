import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np

class CHASEDataset(Dataset):
    """CHASE-DB1 retinal vessel segmentation dataset.
    
    Images are in RGB (jpg), vessel ground-truth masks are binary (png).
    We use the 1st human observer masks for training.
    """
    def __init__(self, data_dir, img_size=(512, 512), augment=False):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.augment = augment
        
        # Find all unique image pairs (ignore _1stHO and _2ndHO suffixes)
        self.images = sorted(self.data_dir.glob("Image_*.jpg"))
        print(f"Found {len(self.images)} CHASE images in {data_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        # Construct mask path: Image_01L.jpg -> Image_01L_1stHO.png
        mask_name = img_path.stem + "_1stHO.png"
        mask_path = img_path.parent / mask_name
        
        # Load image (RGB)
        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        image = cv2.resize(image, self.img_size)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        
        # Load mask (binary vessel segmentation)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to load mask: {mask_path}")
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)  # Binarize
        mask = torch.from_numpy(mask).unsqueeze(0)  # (H,W) -> (1,H,W)
        
        return {
            'image': image,      # (3, 512, 512)
            'mask': mask,        # (1, 512, 512)
            'path': str(img_path)
        }

class DummyDriveDataset(Dataset):
    """Placeholder dataset that yields random tensors."""
    def __init__(self, n=100, img_size=(3, 256, 256)):
        self.n = n
        self.img_size = img_size
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        img = torch.randn(self.img_size)
        mask = (torch.rand(1, self.img_size[1], self.img_size[2]) > 0.8).float()
        return {'image': img, 'mask': mask}
