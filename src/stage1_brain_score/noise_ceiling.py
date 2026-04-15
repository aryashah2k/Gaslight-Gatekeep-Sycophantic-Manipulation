"""
Noise Ceiling Estimation for fMRI Data

The noise ceiling represents the maximum predictable variance in fMRI data,
accounting for measurement noise. It provides an upper bound on any model's
predictive performance and is essential for normalizing brain scores.

Methods:
    1. Split-half reliability: Correlate even vs odd trials
    2. Spearman-Brown correction: Adjust for reduced sample size

References:
    - Schrimpf et al. (2018) - Brain-Score benchmark
    - Allen et al. (2022) - Natural Scenes Dataset
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
from scipy.stats import pearsonr
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NoiseCeiling:
    """Container for noise ceiling estimates."""
    per_voxel: np.ndarray  # Per-voxel noise ceiling
    mean: float
    std: float
    median: float
    method: str
    n_voxels: int
    n_images: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "mean": float(self.mean),
            "std": float(self.std),
            "median": float(self.median),
            "method": self.method,
            "n_voxels": self.n_voxels,
            "n_images": self.n_images
        }


class NoiseCeilingEstimator:
    """
    Estimates the noise ceiling for fMRI data.
    
    The noise ceiling represents the theoretical maximum correlation
    that any model can achieve, given the inherent noise in the data.
    
    For the Algonauts dataset, fMRI responses are already averaged across
    3 repetitions, so we use split-half reliability on the trial dimension.
    
    Example:
        >>> estimator = NoiseCeilingEstimator()
        >>> fmri = np.load("lh_training_fmri.npy")
        >>> ceiling = estimator.estimate_split_half(fmri)
        >>> print(f"Mean noise ceiling: {ceiling.mean:.4f}")
    """
    
    def __init__(self, min_ceiling: float = 0.01, max_ceiling: float = 1.0):
        """
        Initialize the estimator.
        
        Args:
            min_ceiling: Minimum noise ceiling value (avoids division issues)
            max_ceiling: Maximum noise ceiling value (theoretical limit)
        """
        self.min_ceiling = min_ceiling
        self.max_ceiling = max_ceiling
    
    def estimate_split_half(
        self,
        fmri: np.ndarray,
        correction: str = "spearman_brown"
    ) -> NoiseCeiling:
        """
        Estimate noise ceiling using split-half reliability.
        
        Splits the data into even and odd indexed samples and computes
        the correlation between halves. Applies Spearman-Brown correction
        to estimate the reliability of the full dataset.
        
        Args:
            fmri: fMRI data of shape (n_images, n_voxels)
            correction: 'spearman_brown' or 'none'
            
        Returns:
            NoiseCeiling object with per-voxel and summary statistics
        """
        n_images, n_voxels = fmri.shape
        logger.info(f"Estimating noise ceiling: {n_images} images, {n_voxels} voxels")
        
        # Split into even and odd indices
        half1 = fmri[::2]   # Even indices
        half2 = fmri[1::2]  # Odd indices
        
        # Ensure same size
        min_samples = min(len(half1), len(half2))
        half1 = half1[:min_samples]
        half2 = half2[:min_samples]
        
        logger.debug(f"Split halves: {half1.shape}, {half2.shape}")
        
        # Compute per-voxel correlation between halves
        per_voxel_r = np.zeros(n_voxels)
        
        for v in range(n_voxels):
            std1 = np.std(half1[:, v])
            std2 = np.std(half2[:, v])
            
            if std1 > 1e-8 and std2 > 1e-8:
                r, _ = pearsonr(half1[:, v], half2[:, v])
                per_voxel_r[v] = r
            else:
                per_voxel_r[v] = 0.0
        
        # Handle NaN values
        per_voxel_r = np.nan_to_num(per_voxel_r, nan=0.0)
        
        # Apply Spearman-Brown prophecy formula
        if correction == "spearman_brown":
            # r_full = 2 * r_split / (1 + r_split)
            # Clip to avoid issues with negative correlations
            per_voxel_r_clipped = np.clip(per_voxel_r, -0.99, 0.99)
            noise_ceiling = (2 * per_voxel_r_clipped) / (1 + np.abs(per_voxel_r_clipped))
        else:
            noise_ceiling = per_voxel_r
        
        # Clip to valid range
        noise_ceiling = np.clip(noise_ceiling, self.min_ceiling, self.max_ceiling)
        
        result = NoiseCeiling(
            per_voxel=noise_ceiling,
            mean=float(np.mean(noise_ceiling)),
            std=float(np.std(noise_ceiling)),
            median=float(np.median(noise_ceiling)),
            method=f"split_half_{correction}",
            n_voxels=n_voxels,
            n_images=n_images
        )
        
        logger.info(f"Noise ceiling: mean={result.mean:.4f}, std={result.std:.4f}")
        
        return result
    
    def estimate_by_roi(
        self,
        fmri: np.ndarray,
        roi_mask: np.ndarray,
        roi_mapping: Optional[Dict[int, str]] = None
    ) -> Dict[str, NoiseCeiling]:
        """
        Estimate noise ceiling separately for each ROI.
        
        Args:
            fmri: fMRI data of shape (n_images, n_voxels)
            roi_mask: ROI labels of shape (n_voxels,)
            roi_mapping: Optional mapping from ROI indices to names
            
        Returns:
            Dictionary mapping ROI names to NoiseCeiling objects
        """
        roi_ceilings = {}
        unique_rois = np.unique(roi_mask)
        
        for roi_idx in unique_rois:
            if roi_idx == 0:  # Skip background
                continue
            
            # Get voxels for this ROI
            voxel_mask = roi_mask == roi_idx
            roi_fmri = fmri[:, voxel_mask]
            
            if roi_fmri.shape[1] < 10:  # Skip very small ROIs
                continue
            
            # Estimate ceiling
            ceiling = self.estimate_split_half(roi_fmri)
            
            # Get ROI name
            roi_name = roi_mapping.get(roi_idx, f"ROI_{roi_idx}") if roi_mapping else f"ROI_{roi_idx}"
            
            roi_ceilings[roi_name] = ceiling
        
        return roi_ceilings
    
    def save_ceiling(
        self,
        ceiling: NoiseCeiling,
        output_path: Union[str, Path],
        subject_id: int,
        hemisphere: str
    ) -> Path:
        """
        Save noise ceiling to disk.
        
        Args:
            ceiling: NoiseCeiling object
            output_path: Output directory
            subject_id: Subject ID
            hemisphere: 'lh' or 'rh'
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"noise_ceiling_subj{subject_id:02d}_{hemisphere}.npz"
        filepath = output_path / filename
        
        np.savez_compressed(
            filepath,
            per_voxel=ceiling.per_voxel,
            mean=ceiling.mean,
            std=ceiling.std,
            median=ceiling.median,
            method=ceiling.method,
            n_voxels=ceiling.n_voxels,
            n_images=ceiling.n_images
        )
        
        logger.info(f"Saved noise ceiling to {filepath}")
        return filepath
    
    @staticmethod
    def load_ceiling(path: Union[str, Path]) -> NoiseCeiling:
        """Load noise ceiling from disk."""
        data = np.load(path, allow_pickle=True)
        
        return NoiseCeiling(
            per_voxel=data['per_voxel'],
            mean=float(data['mean']),
            std=float(data['std']),
            median=float(data['median']),
            method=str(data['method']),
            n_voxels=int(data['n_voxels']),
            n_images=int(data['n_images'])
        )


def compute_noise_ceilings_all_subjects(
    data_loader,
    output_dir: Union[str, Path],
    subjects: Optional[list] = None
) -> Dict[str, Dict[str, NoiseCeiling]]:
    """
    Compute noise ceilings for all subjects and hemispheres.
    
    Args:
        data_loader: AlgonautsDataLoader instance
        output_dir: Directory to save results
        subjects: List of subject IDs (default: all available)
        
    Returns:
        Nested dict: {subject_id: {hemisphere: NoiseCeiling}}
    """
    from src.utils.data_loader import AlgonautsDataLoader
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    estimator = NoiseCeilingEstimator()
    subjects = subjects or data_loader.subjects
    
    all_ceilings = {}
    
    for subj_id in subjects:
        logger.info(f"Processing subject {subj_id}")
        all_ceilings[subj_id] = {}
        
        for hemi in ["lh", "rh"]:
            fmri = data_loader.load_fmri_by_hemisphere(subj_id, hemi)
            ceiling = estimator.estimate_split_half(fmri)
            
            # Save
            estimator.save_ceiling(ceiling, output_dir, subj_id, hemi)
            
            all_ceilings[subj_id][hemi] = ceiling
    
    return all_ceilings
