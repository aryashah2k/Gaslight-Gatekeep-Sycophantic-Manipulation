"""
Algonauts 2023 Dataset Loader

This module provides a robust data loader for the Algonauts 2023 Challenge dataset,
including fMRI data, images, and ROI masks.

References:
    - Algonauts 2023 Challenge: http://algonauts.csail.mit.edu/
    - Natural Scenes Dataset: https://naturalscenesdataset.org/
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SubjectData:
    """Container for a single subject's data."""
    subject_id: int
    lh_fmri: np.ndarray  # Shape: (n_images, n_vertices_lh)
    rh_fmri: np.ndarray  # Shape: (n_images, n_vertices_rh)
    image_paths: List[Path]
    roi_masks: Dict[str, Dict[str, np.ndarray]]
    n_images: int
    n_vertices_lh: int
    n_vertices_rh: int


@dataclass
class ROIMasks:
    """Container for ROI masks."""
    challenge_space: np.ndarray
    fsaverage_space: np.ndarray
    mapping: Dict[int, str]


class AlgonautsDataLoader:
    """
    Loader for Algonauts 2023 Challenge dataset.
    
    The dataset contains fMRI responses from 8 subjects viewing natural scenes
    from the COCO database. Data is organized per-subject with:
    - Training fMRI data (z-scored, averaged across repeats)
    - Training/test images
    - ROI masks for different brain regions
    
    Attributes:
        data_dir: Path to the algonauts data directory
        subjects: List of available subject IDs
    
    Example:
        >>> loader = AlgonautsDataLoader("./data")
        >>> subject_data = loader.load_subject(1)
        >>> print(f"Subject 1: {subject_data.n_images} images, {subject_data.n_vertices_lh} LH vertices")
    """
    
    # Expected number of vertices per hemisphere (may vary slightly for some subjects)
    EXPECTED_LH_VERTICES = 19004
    EXPECTED_RH_VERTICES = 20544
    
    # Subjects with different vertex counts
    VERTEX_EXCEPTIONS = {
        6: {"lh": 18978, "rh": 20220},
        8: {"lh": 18981, "rh": 20530}
    }
    
    # Number of training images per subject
    TRAIN_IMAGES_PER_SUBJECT = {
        1: 9841, 2: 9841, 3: 9082, 4: 8779,
        5: 9841, 6: 9082, 7: 9841, 8: 8779
    }
    
    # Number of test images per subject
    TEST_IMAGES_PER_SUBJECT = {
        1: 159, 2: 159, 3: 293, 4: 395,
        5: 159, 6: 293, 7: 159, 8: 395
    }
    
    # ROI categories
    ROI_CATEGORIES = [
        "prf-visualrois",  # V1v, V1d, V2v, V2d, V3v, V3d, hV4
        "floc-bodies",     # EBA, FBA-1, FBA-2, mTL-bodies
        "floc-faces",      # OFA, FFA-1, FFA-2, mTL-faces, aTL-faces
        "floc-places",     # OPA, PPA, RSC
        "floc-words",      # OWFA, VWFA-1, VWFA-2, mfs-words, mTL-words
        "streams"          # early, midventral, midlateral, midparietal, ventral, lateral, parietal
    ]
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the data directory containing subj01, subj02, etc.
        """
        self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Discover available subjects
        self.subjects = self._discover_subjects()
        logger.info(f"Found {len(self.subjects)} subjects: {self.subjects}")
    
    def _discover_subjects(self) -> List[int]:
        """Discover available subject directories."""
        subjects = []
        for i in range(1, 9):
            subj_dir = self.data_dir / f"subj{i:02d}"
            if subj_dir.exists():
                subjects.append(i)
        return subjects
    
    def load_subject(
        self, 
        subject_id: int,
        load_images: bool = False
    ) -> SubjectData:
        """
        Load all data for a single subject.
        
        Args:
            subject_id: Subject number (1-8)
            load_images: Whether to verify image paths exist
            
        Returns:
            SubjectData object containing all subject data
        """
        if subject_id not in self.subjects:
            raise ValueError(f"Subject {subject_id} not found. Available: {self.subjects}")
        
        subj_dir = self.data_dir / f"subj{subject_id:02d}"
        
        # Load fMRI data
        lh_fmri = self._load_fmri(subj_dir, "lh")
        rh_fmri = self._load_fmri(subj_dir, "rh")
        
        # Load image paths
        image_paths = self._get_training_image_paths(subj_dir)
        
        # Load ROI masks
        roi_masks = self._load_all_roi_masks(subj_dir)
        
        # Validate dimensions
        assert lh_fmri.shape[0] == rh_fmri.shape[0], "LH/RH image count mismatch"
        assert lh_fmri.shape[0] == len(image_paths), "fMRI/image count mismatch"
        
        return SubjectData(
            subject_id=subject_id,
            lh_fmri=lh_fmri,
            rh_fmri=rh_fmri,
            image_paths=image_paths,
            roi_masks=roi_masks,
            n_images=lh_fmri.shape[0],
            n_vertices_lh=lh_fmri.shape[1],
            n_vertices_rh=rh_fmri.shape[1]
        )
    
    def _load_fmri(self, subj_dir: Path, hemisphere: str) -> np.ndarray:
        """Load fMRI data for a hemisphere."""
        fmri_path = subj_dir / "training_split" / "training_fmri" / f"{hemisphere}_training_fmri.npy"
        
        if not fmri_path.exists():
            raise FileNotFoundError(f"fMRI file not found: {fmri_path}")
        
        fmri = np.load(fmri_path)
        logger.debug(f"Loaded {hemisphere} fMRI: shape {fmri.shape}")
        
        return fmri.astype(np.float32)
    
    def load_fmri_by_hemisphere(
        self, 
        subject_id: int, 
        hemisphere: str
    ) -> np.ndarray:
        """
        Load fMRI data for a specific subject and hemisphere.
        
        Args:
            subject_id: Subject number (1-8)
            hemisphere: 'lh' or 'rh'
            
        Returns:
            fMRI array of shape (n_images, n_vertices)
        """
        if hemisphere not in ["lh", "rh"]:
            raise ValueError("hemisphere must be 'lh' or 'rh'")
        
        subj_dir = self.data_dir / f"subj{subject_id:02d}"
        return self._load_fmri(subj_dir, hemisphere)
    
    def _get_training_image_paths(self, subj_dir: Path) -> List[Path]:
        """Get sorted list of training image paths."""
        images_dir = subj_dir / "training_split" / "training_images"
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Training images not found: {images_dir}")
        
        # Get all PNG files and sort by train index
        image_files = sorted(
            images_dir.glob("train-*.png"),
            key=lambda p: int(p.stem.split("_")[0].replace("train-", ""))
        )
        
        return image_files
    
    def get_test_image_paths(self, subject_id: int) -> List[Path]:
        """Get sorted list of test image paths for a subject."""
        subj_dir = self.data_dir / f"subj{subject_id:02d}"
        images_dir = subj_dir / "test_split" / "test_images"
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Test images not found: {images_dir}")
        
        image_files = sorted(
            images_dir.glob("test-*.png"),
            key=lambda p: int(p.stem.split("_")[0].replace("test-", ""))
        )
        
        return image_files
    
    def _load_all_roi_masks(self, subj_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
        """Load all ROI masks for a subject."""
        roi_dir = subj_dir / "roi_masks"
        masks = {}
        
        for category in self.ROI_CATEGORIES:
            masks[category] = {}
            
            for hemi in ["lh", "rh"]:
                # Challenge space mask
                challenge_path = roi_dir / f"{hemi}.{category}_challenge_space.npy"
                if challenge_path.exists():
                    masks[category][f"{hemi}_challenge"] = np.load(challenge_path)
                
                # Fsaverage space mask
                fsaverage_path = roi_dir / f"{hemi}.{category}_fsaverage_space.npy"
                if fsaverage_path.exists():
                    masks[category][f"{hemi}_fsaverage"] = np.load(fsaverage_path)
            
            # Mapping file
            mapping_path = roi_dir / f"mapping_{category}.npy"
            if mapping_path.exists():
                masks[category]["mapping"] = np.load(mapping_path, allow_pickle=True).item()
        
        return masks
    
    def load_roi_mask(
        self, 
        subject_id: int, 
        roi_category: str, 
        hemisphere: str,
        space: str = "challenge"
    ) -> np.ndarray:
        """
        Load a specific ROI mask.
        
        Args:
            subject_id: Subject number (1-8)
            roi_category: One of the ROI_CATEGORIES
            hemisphere: 'lh' or 'rh'
            space: 'challenge' or 'fsaverage'
            
        Returns:
            ROI mask array
        """
        subj_dir = self.data_dir / f"subj{subject_id:02d}"
        roi_dir = subj_dir / "roi_masks"
        
        mask_path = roi_dir / f"{hemisphere}.{roi_category}_{space}_space.npy"
        
        if not mask_path.exists():
            raise FileNotFoundError(f"ROI mask not found: {mask_path}")
        
        return np.load(mask_path)
    
    def load_roi_mapping(self, roi_category: str, subject_id: int = 1) -> Dict[int, str]:
        """
        Load ROI index to name mapping.
        
        Args:
            roi_category: One of the ROI_CATEGORIES
            subject_id: Subject to load from (mappings are same across subjects)
            
        Returns:
            Dictionary mapping integer indices to ROI names
        """
        subj_dir = self.data_dir / f"subj{subject_id:02d}"
        mapping_path = subj_dir / "roi_masks" / f"mapping_{roi_category}.npy"
        
        if not mapping_path.exists():
            raise FileNotFoundError(f"ROI mapping not found: {mapping_path}")
        
        return np.load(mapping_path, allow_pickle=True).item()
    
    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """Load an image as PIL Image."""
        return Image.open(image_path).convert("RGB")
    
    def load_images_batch(
        self, 
        image_paths: List[Path], 
        resize: Optional[Tuple[int, int]] = None
    ) -> List[Image.Image]:
        """
        Load a batch of images.
        
        Args:
            image_paths: List of image paths
            resize: Optional (width, height) to resize images
            
        Returns:
            List of PIL Images
        """
        images = []
        for path in image_paths:
            img = self.load_image(path)
            if resize:
                img = img.resize(resize, Image.Resampling.LANCZOS)
            images.append(img)
        return images
    
    def get_nsd_id_from_filename(self, filename: str) -> int:
        """
        Extract NSD image ID from filename.
        
        Args:
            filename: e.g., "train-0001_nsd-00013.png"
            
        Returns:
            NSD ID (e.g., 13)
        """
        # Extract nsd-XXXXX part
        nsd_part = filename.split("_")[1]
        nsd_id = int(nsd_part.replace("nsd-", "").replace(".png", ""))
        return nsd_id
    
    def get_train_index_from_filename(self, filename: str) -> int:
        """
        Extract training index from filename.
        
        Args:
            filename: e.g., "train-0001_nsd-00013.png"
            
        Returns:
            Train index (e.g., 1) - NOTE: 1-indexed in filenames
        """
        train_part = filename.split("_")[0]
        train_idx = int(train_part.replace("train-", ""))
        return train_idx
    
    def get_combined_fmri(self, subject_id: int) -> np.ndarray:
        """
        Get concatenated LH + RH fMRI data.
        
        Args:
            subject_id: Subject number (1-8)
            
        Returns:
            Combined fMRI array of shape (n_images, n_vertices_lh + n_vertices_rh)
        """
        subj_data = self.load_subject(subject_id)
        return np.concatenate([subj_data.lh_fmri, subj_data.rh_fmri], axis=1)
    
    def get_subject_info(self, subject_id: int) -> Dict:
        """Get summary information about a subject."""
        subj_dir = self.data_dir / f"subj{subject_id:02d}"
        
        lh_fmri = self._load_fmri(subj_dir, "lh")
        rh_fmri = self._load_fmri(subj_dir, "rh")
        train_images = self._get_training_image_paths(subj_dir)
        test_images = self.get_test_image_paths(subject_id)
        
        return {
            "subject_id": subject_id,
            "n_train_images": len(train_images),
            "n_test_images": len(test_images),
            "n_vertices_lh": lh_fmri.shape[1],
            "n_vertices_rh": rh_fmri.shape[1],
            "n_vertices_total": lh_fmri.shape[1] + rh_fmri.shape[1],
            "fmri_dtype": str(lh_fmri.dtype)
        }
    
    def get_expected_image_count(self, subject_id: int, split: str = "train") -> int:
        """
        Get expected image count for a subject.
        
        This is useful for validating extracted features against expected counts.
        
        Args:
            subject_id: Subject ID (1-8)
            split: 'train' or 'test'
            
        Returns:
            Expected number of images
        """
        if split == "train":
            return self.TRAIN_IMAGES_PER_SUBJECT.get(subject_id, 0)
        elif split == "test":
            return self.TEST_IMAGES_PER_SUBJECT.get(subject_id, 0)
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split}")
    
    def get_expected_vertex_count(self, subject_id: int, hemisphere: str) -> int:
        """
        Get expected vertex count for a subject/hemisphere.
        
        Subjects 6 and 8 have fewer vertices due to missing data.
        
        Args:
            subject_id: Subject ID (1-8)
            hemisphere: 'lh' or 'rh'
            
        Returns:
            Expected number of vertices
        """
        if subject_id in self.VERTEX_EXCEPTIONS:
            return self.VERTEX_EXCEPTIONS[subject_id][hemisphere]
        
        if hemisphere == "lh":
            return self.EXPECTED_LH_VERTICES
        else:
            return self.EXPECTED_RH_VERTICES
    
    def validate_features_alignment(
        self, 
        features: np.ndarray, 
        subject_id: int
    ) -> bool:
        """
        Validate that extracted features are properly aligned with subject data.
        
        Args:
            features: Feature array of shape (n_images, feature_dim)
            subject_id: Subject ID
            
        Returns:
            True if aligned, raises ValueError if not
        """
        expected = self.get_expected_image_count(subject_id, "train")
        actual = features.shape[0]
        
        if actual != expected:
            raise ValueError(
                f"Feature count mismatch for subject {subject_id}!\n"
                f"  Expected: {expected} images\n"
                f"  Got: {actual} features\n"
                f"  This usually means features were extracted for a different subject."
            )
        
        return True
    
    def print_dataset_summary(self):
        """Print a summary of the entire dataset with validation."""
        print("\n" + "=" * 70)
        print("ALGONAUTS 2023 DATASET SUMMARY")
        print("=" * 70)
        print(f"\n{'Subject':<10} {'Train Imgs':<12} {'Test Imgs':<12} {'LH Verts':<12} {'RH Verts':<12}")
        print("-" * 70)
        
        for subj_id in self.subjects:
            info = self.get_subject_info(subj_id)
            
            # Check if counts match expected
            expected_train = self.TRAIN_IMAGES_PER_SUBJECT.get(subj_id, 0)
            expected_test = self.TEST_IMAGES_PER_SUBJECT.get(subj_id, 0)
            
            train_status = "✓" if info['n_train_images'] == expected_train else "✗"
            test_status = "✓" if info['n_test_images'] == expected_test else "✗"
            
            print(f"subj{subj_id:02d}    {info['n_train_images']:<12} {info['n_test_images']:<12} "
                  f"{info['n_vertices_lh']:<12} {info['n_vertices_rh']:<12}")
        
        print("-" * 70)
        print("\nNote: Subjects 3,4,6,8 have different image counts (incomplete sessions)")
        print("      Subjects 6,8 have fewer vertices (missing voxels)")
        print("=" * 70)
