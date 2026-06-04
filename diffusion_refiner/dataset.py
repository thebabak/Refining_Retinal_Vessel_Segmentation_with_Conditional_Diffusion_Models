from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
MASK_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.gif', '.bmp')


def _list_images(folder: Path) -> list[Path]:
    paths: list[Path] = []
    for ext in IMAGE_EXTS:
        paths.extend(folder.glob(f'*{ext}'))
        paths.extend(folder.glob(f'*{ext.upper()}'))
    return sorted(set(paths))


def _is_probably_annotation(path: Path) -> bool:
    stem = path.stem.lower()
    bad_tokens = (
        '_1stho', '_2ndho', '_mask', '_manual', '_gt', '_label',
        '_coarse', '_pred', '_prediction', '_lunet', '_ra'
    )
    return any(tok in stem for tok in bad_tokens)


def _find_matching_file(folder: Path, image_path: Path, extra_stems: Optional[Sequence[str]] = None) -> Path:
    """Find a mask/prediction file matching an image stem."""
    stem = image_path.stem
    candidates: list[Path] = []

    stems = [stem]
    if extra_stems:
        stems.extend(extra_stems)

    for s in stems:
        candidates.extend(folder.glob(f'{s}.*'))
        candidates.extend(folder.glob(f'{s}_mask.*'))
        candidates.extend(folder.glob(f'{s}_manual.*'))
        candidates.extend(folder.glob(f'{s}_manual1.*'))
        candidates.extend(folder.glob(f'{s}_1stHO.*'))
        candidates.extend(folder.glob(f'{s}_2ndHO.*'))
        candidates.extend(folder.glob(f'{s}_pred.*'))
        candidates.extend(folder.glob(f'{s}_prediction.*'))
        candidates.extend(folder.glob(f'{s}_lunet_ra.*'))
        candidates.extend(folder.glob(f'{s}*.*'))

    candidates = sorted({p for p in candidates if p.suffix.lower() in MASK_EXTS})
    if not candidates:
        raise FileNotFoundError(f'No matching file found in {folder} for image {image_path.name}')

    # Prefer exact stem matches, then shorter names to avoid arbitrary glob order surprises.
    candidates.sort(key=lambda p: (p.stem != stem, len(p.name), p.name))
    return candidates[0]


def _load_image(path: Path, img_size: tuple[int, int]) -> torch.Tensor:
    image = Image.open(path).convert('RGB')
    image = image.resize(img_size, Image.Resampling.BILINEAR)
    return TF.to_tensor(image).float()


def _load_binary_mask(path: Path, img_size: tuple[int, int]) -> torch.Tensor:
    mask = Image.open(path).convert('L')
    mask = mask.resize(img_size, Image.Resampling.NEAREST)
    mask = TF.to_tensor(mask).float()
    return (mask > 0.5).float()


def _load_probability_mask(path: Path, img_size: tuple[int, int], binarize: bool = False) -> torch.Tensor:
    """
    Load a coarse prediction. Keep probabilities by default.
    Use binarize=True only if your saved coarse masks are binary masks.
    """
    pred = Image.open(path).convert('L')
    pred = pred.resize(img_size, Image.Resampling.BILINEAR)
    pred = TF.to_tensor(pred).float()
    if binarize:
        pred = (pred > 0.5).float()
    return pred.clamp(0.0, 1.0)


class CHASEDataset(Dataset):
    """
    CHASE-DB1 dataset for diffusion refinement.

    Required for real refinement experiments:
        data_dir/Image_01L.jpg
        data_dir/Image_01L_1stHO.png       -> ground truth
        coarse_dir/Image_01L.png or matching LU-Net+RA prediction -> coarse mask

    Optional:
        data_dir/Image_01L_2ndHO.png       -> returned as second_observer only

    Important: second_observer is NOT returned as coarse/baseline.
    """

    def __init__(
        self,
        data_dir,
        img_size: tuple[int, int] = (512, 512),
        coarse_dir=None,
        require_coarse: bool = True,
        binarize_coarse: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.coarse_dir = Path(coarse_dir) if coarse_dir is not None else None
        self.img_size = img_size
        self.require_coarse = require_coarse
        self.binarize_coarse = binarize_coarse

        self.image_paths = [p for p in _list_images(self.data_dir) if not _is_probably_annotation(p)]
        if not self.image_paths:
            raise RuntimeError(f'No fundus images found in {self.data_dir}')

        if self.require_coarse and self.coarse_dir is None:
            raise ValueError('coarse_dir is required for real diffusion-refiner training/evaluation.')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        stem = img_path.stem

        gt_path = self.data_dir / f'{stem}_1stHO.png'
        if not gt_path.exists():
            raise FileNotFoundError(f'Missing CHASE GT mask: {gt_path}')

        sample = {
            'image': _load_image(img_path, self.img_size),
            'mask': _load_binary_mask(gt_path, self.img_size),
            'name': stem,
        }

        second_path = self.data_dir / f'{stem}_2ndHO.png'
        if second_path.exists():
            sample['second_observer'] = _load_binary_mask(second_path, self.img_size)

        if self.coarse_dir is not None:
            coarse_path = _find_matching_file(self.coarse_dir, img_path)
            sample['coarse'] = _load_probability_mask(coarse_path, self.img_size, self.binarize_coarse)
            sample['coarse_path'] = str(coarse_path)
        elif self.require_coarse:
            raise RuntimeError('Missing coarse prediction directory.')

        return sample


class VesselDataset(Dataset):
    """
    Generic DRIVE/HRF-style dataset.

    images_dir: fundus images
    masks_dir: ground-truth masks
    coarse_dir: LU-Net+RA coarse predictions, matched by image stem
    """

    def __init__(
        self,
        images_dir,
        masks_dir,
        img_size: tuple[int, int] = (512, 512),
        coarse_dir=None,
        require_coarse: bool = True,
        binarize_coarse: bool = False,
        dataset_name: str = 'VesselDataset',
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.coarse_dir = Path(coarse_dir) if coarse_dir is not None else None
        self.img_size = img_size
        self.require_coarse = require_coarse
        self.binarize_coarse = binarize_coarse
        self.dataset_name = dataset_name

        self.image_paths = _list_images(self.images_dir)
        if not self.image_paths:
            raise RuntimeError(f'No {dataset_name} images found in {self.images_dir}')

        if self.require_coarse and self.coarse_dir is None:
            raise ValueError('coarse_dir is required for real diffusion-refiner training/evaluation.')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = _find_matching_file(self.masks_dir, img_path)

        sample = {
            'image': _load_image(img_path, self.img_size),
            'mask': _load_binary_mask(mask_path, self.img_size),
            'name': img_path.stem,
        }

        if self.coarse_dir is not None:
            coarse_path = _find_matching_file(self.coarse_dir, img_path)
            sample['coarse'] = _load_probability_mask(coarse_path, self.img_size, self.binarize_coarse)
            sample['coarse_path'] = str(coarse_path)
        elif self.require_coarse:
            raise RuntimeError('Missing coarse prediction directory.')

        return sample


class DRIVEDataset(VesselDataset):
    def __init__(self, images_dir, masks_dir, img_size=(512, 512), coarse_dir=None, require_coarse=True, binarize_coarse=False):
        super().__init__(images_dir, masks_dir, img_size, coarse_dir, require_coarse, binarize_coarse, 'DRIVE')


class HRFDataset(VesselDataset):
    def __init__(self, images_dir, masks_dir, img_size=(512, 512), coarse_dir=None, require_coarse=True, binarize_coarse=False):
        super().__init__(images_dir, masks_dir, img_size, coarse_dir, require_coarse, binarize_coarse, 'HRF')


class DummyDriveDataset(Dataset):
    """Small dummy dataset for debugging only. Never use for reported metrics."""

    def __init__(self, n=4, img_size=(3, 128, 128)):
        self.n = n
        self.img_size = img_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        image = torch.rand(*self.img_size)
        h, w = self.img_size[1], self.img_size[2]
        mask = torch.zeros(1, h, w)
        mask[:, h // 2 - 2:h // 2 + 2, :] = 1.0

        # Deliberately imperfect coarse mask for debugging refinement behavior.
        coarse = torch.zeros_like(mask)
        coarse[:, h // 2 - 1:h // 2 + 1, :] = 1.0

        return {'image': image, 'mask': mask, 'coarse': coarse, 'name': f'dummy_{idx}'}
