from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


class CHASEDataset(Dataset):
    """
    CHASE-DB1 style dataset.

    Expected files:
        Image_01L.jpg
        Image_01L_1stHO.png
        Image_01L_2ndHO.png

    Returns:
        image       -> fundus image
        mask        -> first human observer, used as GT
        baseline    -> second human observer, used as reference comparison
        name        -> image stem
    """

    def __init__(self, data_dir, img_size=(512, 512)):
        self.data_dir = Path(data_dir)
        self.img_size = img_size

        self.image_paths = sorted(
            list(self.data_dir.glob("*.jpg")) +
            list(self.data_dir.glob("*.jpeg")) +
            list(self.data_dir.glob("*.png")) +
            list(self.data_dir.glob("*.tif")) +
            list(self.data_dir.glob("*.tiff"))
        )

        # Keep only fundus images, not observer/mask files
        self.image_paths = [
            p for p in self.image_paths
            if "_1stHO" not in p.stem
            and "_2ndHO" not in p.stem
            and "_mask" not in p.stem.lower()
            and "_manual" not in p.stem.lower()
        ]

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No fundus images found in {self.data_dir}")

    def __len__(self):
        return len(self.image_paths)

    def _load_mask(self, path):
        mask = Image.open(path).convert("L")
        mask = mask.resize(self.img_size, Image.Resampling.NEAREST)
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()
        return mask

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        stem = img_path.stem

        gt_path = self.data_dir / f"{stem}_1stHO.png"
        second_path = self.data_dir / f"{stem}_2ndHO.png"

        if not gt_path.exists():
            raise FileNotFoundError(f"Missing GT mask: {gt_path}")

        if not second_path.exists():
            raise FileNotFoundError(f"Missing second observer mask: {second_path}")

        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.img_size, Image.Resampling.BILINEAR)
        image = TF.to_tensor(image)

        gt_mask = self._load_mask(gt_path)
        second_observer = self._load_mask(second_path)

        return {
            "image": image,
            "mask": gt_mask,
            "baseline": second_observer,
            "name": stem,
        }


class DRIVEDataset(Dataset):
    """
    DRIVE-style dataset.

    Expected structure:
        images_dir/
            image files
        masks_dir/
            matching mask files

    The class tries to match masks by image stem.
    If no real baseline is available, baseline is set as mask.clone().
    Replace baseline with actual LU-Net+RA predictions if available.
    """

    def __init__(self, images_dir, masks_dir, img_size=(512, 512), baseline_dir=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.baseline_dir = Path(baseline_dir) if baseline_dir is not None else None
        self.img_size = img_size

        self.image_paths = sorted(
            list(self.images_dir.glob("*.jpg")) +
            list(self.images_dir.glob("*.jpeg")) +
            list(self.images_dir.glob("*.png")) +
            list(self.images_dir.glob("*.tif")) +
            list(self.images_dir.glob("*.tiff"))
        )

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No DRIVE images found in {self.images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def _load_mask(self, path):
        mask = Image.open(path).convert("L")
        mask = mask.resize(self.img_size, Image.Resampling.NEAREST)
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()
        return mask

    def _find_matching_file(self, folder, image_path):
        stem = image_path.stem

        candidates = []

        # Exact stem match
        candidates += list(folder.glob(f"{stem}.*"))

        # Common DRIVE naming variants
        candidates += list(folder.glob(f"{stem}_manual1.*"))
        candidates += list(folder.glob(f"{stem}_manual.*"))
        candidates += list(folder.glob(f"{stem}_mask.*"))
        candidates += list(folder.glob(f"{stem}*.*"))

        candidates = [
            p for p in candidates
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif"]
        ]

        # Remove possible image files if masks are in same folder
        candidates = [
            p for p in candidates
            if "mask" in p.stem.lower()
            or "manual" in p.stem.lower()
            or "1st" in p.stem.lower()
            or "2nd" in p.stem.lower()
            or folder != self.images_dir
        ]

        if len(candidates) == 0:
            raise FileNotFoundError(
                f"No matching file found in {folder} for image {image_path.name}"
            )

        return candidates[0]

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        mask_path = self._find_matching_file(self.masks_dir, img_path)

        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.img_size, Image.Resampling.BILINEAR)
        image = TF.to_tensor(image)

        mask = self._load_mask(mask_path)

        if self.baseline_dir is not None:
            baseline_path = self._find_matching_file(self.baseline_dir, img_path)
            baseline = self._load_mask(baseline_path)
        else:
            baseline = mask.clone()

        return {
            "image": image,
            "mask": mask,
            "baseline": baseline,
            "name": img_path.stem,
        }


class HRFDataset(Dataset):
    """
    HRF-style dataset.

    Expected structure:
        images_dir/
            image files
        masks_dir/
            matching mask files

    If baseline_dir is provided, it loads baseline masks from there.
    Otherwise baseline is set as mask.clone().
    """

    def __init__(self, images_dir, masks_dir, img_size=(512, 512), baseline_dir=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.baseline_dir = Path(baseline_dir) if baseline_dir is not None else None
        self.img_size = img_size

        self.image_paths = sorted(
            list(self.images_dir.glob("*.jpg")) +
            list(self.images_dir.glob("*.jpeg")) +
            list(self.images_dir.glob("*.png")) +
            list(self.images_dir.glob("*.tif")) +
            list(self.images_dir.glob("*.tiff"))
        )

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No HRF images found in {self.images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def _load_mask(self, path):
        mask = Image.open(path).convert("L")
        mask = mask.resize(self.img_size, Image.Resampling.NEAREST)
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()
        return mask

    def _find_matching_file(self, folder, image_path):
        stem = image_path.stem

        candidates = []
        candidates += list(folder.glob(f"{stem}.*"))
        candidates += list(folder.glob(f"{stem}_manual.*"))
        candidates += list(folder.glob(f"{stem}_mask.*"))
        candidates += list(folder.glob(f"{stem}*.*"))

        candidates = [
            p for p in candidates
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif"]
        ]

        if len(candidates) == 0:
            raise FileNotFoundError(
                f"No matching file found in {folder} for image {image_path.name}"
            )

        return candidates[0]

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        mask_path = self._find_matching_file(self.masks_dir, img_path)

        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.img_size, Image.Resampling.BILINEAR)
        image = TF.to_tensor(image)

        mask = self._load_mask(mask_path)

        if self.baseline_dir is not None:
            baseline_path = self._find_matching_file(self.baseline_dir, img_path)
            baseline = self._load_mask(baseline_path)
        else:
            baseline = mask.clone()

        return {
            "image": image,
            "mask": mask,
            "baseline": baseline,
            "name": img_path.stem,
        }


class DummyDriveDataset(Dataset):
    """
    Small dummy dataset for testing code only.
    """

    def __init__(self, n=4, img_size=(3, 128, 128)):
        self.n = n
        self.img_size = img_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        image = torch.rand(*self.img_size)

        h, w = self.img_size[1], self.img_size[2]
        mask = torch.zeros(1, h, w)

        # Simple fake vessel-like line
        mask[:, h // 2 - 2:h // 2 + 2, :] = 1.0

        return {
            "image": image,
            "mask": mask,
            "baseline": mask.clone(),
            "name": f"dummy_{idx}",
        }