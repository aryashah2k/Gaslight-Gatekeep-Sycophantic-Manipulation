"""
Script 02b: Fit ROI-Specific Encoders and Compute Per-ROI Brain Scores

Computes brain scores separately for each ROI category:
- prf-visualrois (V1, V2, V3, hV4)
- floc-bodies (EBA, FBA, mTL-bodies)
- floc-faces (OFA, FFA, aTL-faces)
- floc-places (OPA, PPA, RSC)
- floc-words (OWFA, VWFA, mfs-words, mTL-words)
- streams (early, midventral, midlateral, ventral, lateral, parietal)

Usage:
    python scripts/02b_fit_roi_encoders.py --models all --subjects 1 2 3 4
    python scripts/02b_fit_roi_encoders.py --models llava_7b qwen2vl_2b
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import logging
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


AVAILABLE_MODELS = [
    "qwen2vl_2b", "qwen25vl_3b", "gemma3_1b", "llava_7b",
    "paligemma2_10b", "blip2_opt27b", "lfm2vl_8b", "lfm25vl_1b", "idefics2_8b",
    "phi35_vision", "smolvlm_256m", "smolvlm_500m"
]

# ROI categories available in Algonauts 2023
ROI_CATEGORIES = [
    "prf-visualrois",
    "floc-bodies",
    "floc-faces",
    "floc-places",
    "floc-words",
    "streams"
]


@dataclass
class ROIBrainScore:
    """Container for ROI-specific brain scores."""
    model_name: str
    roi_scores: Dict[str, float]  # roi_category -> score
    roi_scores_per_subject: Dict[str, Dict[int, float]]  # roi_category -> subject -> score
    overall_score: float
    n_subjects: int
    
    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "roi_scores": self.roi_scores,
            "roi_scores_per_subject": {k: {str(sk): sv for sk, sv in v.items()} 
                                        for k, v in self.roi_scores_per_subject.items()},
            "overall_score": self.overall_score,
            "n_subjects": self.n_subjects
        }


class ROIBrainScoreComputer:
    """
    Computes brain scores for each ROI category.
    
    For each ROI:
    1. Load ROI mask
    2. Extract voxels within mask from fMRI data
    3. Fit Ridge regression on masked voxels
    4. Compute correlation score
    """
    
    def __init__(
        self,
        data_dir: Path,
        features_dir: Path,
        output_dir: Path,
        alphas: List[float] = None
    ):
        self.data_dir = Path(data_dir)
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.alphas = alphas or [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
        
        # Cache
        self._fmri_cache = {}
        self._mask_cache = {}
        self._features_cache = {}
    
    def load_fmri(self, subject_id: int, hemisphere: str) -> np.ndarray:
        """Load fMRI data for a subject/hemisphere."""
        cache_key = (subject_id, hemisphere)
        if cache_key in self._fmri_cache:
            return self._fmri_cache[cache_key]
        
        fmri_path = (self.data_dir / f"subj{subject_id:02d}" / "training_split" / 
                     "training_fmri" / f"{hemisphere}_training_fmri.npy")
        
        if not fmri_path.exists():
            raise FileNotFoundError(f"fMRI file not found: {fmri_path}")
        
        fmri = np.load(fmri_path)
        self._fmri_cache[cache_key] = fmri
        return fmri
    
    def load_roi_mask(self, subject_id: int, hemisphere: str, roi_category: str) -> np.ndarray:
        """
        Load ROI mask for a subject/hemisphere/ROI category.
        
        Returns:
            Mask array where values > 0 indicate ROI membership.
            Different values may indicate different sub-ROIs.
        """
        cache_key = (subject_id, hemisphere, roi_category)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]
        
        mask_path = (self.data_dir / f"subj{subject_id:02d}" / "roi_masks" / 
                     f"{hemisphere}.{roi_category}_challenge_space.npy")
        
        if not mask_path.exists():
            logger.warning(f"ROI mask not found: {mask_path}")
            return None
        
        mask = np.load(mask_path)
        self._mask_cache[cache_key] = mask
        return mask
    
    def load_features(self, model_name: str, subject_id: int) -> np.ndarray:
        """Load pre-extracted features for a model and subject."""
        cache_key = (model_name, subject_id)
        if cache_key in self._features_cache:
            return self._features_cache[cache_key]
        
        # Features are stored as: {model_name}_subj{id:02d}_features.npz
        features_path = self.features_dir / model_name / f"{model_name}_subj{subject_id:02d}_features.npz"
        
        if not features_path.exists():
            # Try alternative patterns
            alt_path = self.features_dir / model_name / f"subj{subject_id:02d}_features.npy"
            if alt_path.exists():
                features_path = alt_path
            else:
                raise FileNotFoundError(f"Features not found: {features_path}")
        
        # Load features - handle both .npy and .npz formats
        if features_path.suffix == '.npz':
            data = np.load(features_path)
            # NPZ files are archives - get the first array or 'features' key
            if 'features' in data.files:
                features = data['features']
            elif 'arr_0' in data.files:
                features = data['arr_0']
            else:
                # Get first array in archive
                features = data[data.files[0]]
        else:
            features = np.load(features_path)
        
        self._features_cache[cache_key] = features
        return features
    
    def compute_roi_score(
        self,
        features: np.ndarray,
        fmri: np.ndarray,
        mask: np.ndarray
    ) -> float:
        """
        Compute brain score for a specific ROI.
        
        Args:
            features: Visual features (n_images, feature_dim)
            fmri: Full fMRI data (n_images, n_voxels)
            mask: ROI mask (n_voxels,) where values > 0 are in ROI
            
        Returns:
            Mean Pearson correlation across voxels in ROI
        """
        # Extract voxels within ROI
        roi_indices = np.where(mask > 0)[0]
        
        if len(roi_indices) == 0:
            return 0.0
        
        roi_fmri = fmri[:, roi_indices]
        
        # Check for NaN or constant voxels
        valid_voxels = []
        for i in range(roi_fmri.shape[1]):
            voxel_data = roi_fmri[:, i]
            if not np.isnan(voxel_data).any() and np.std(voxel_data) > 0:
                valid_voxels.append(i)
        
        if len(valid_voxels) == 0:
            return 0.0
        
        roi_fmri = roi_fmri[:, valid_voxels]
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            features, roi_fmri, test_size=0.2, random_state=42
        )
        
        # Fit Ridge regression with cross-validation
        model = RidgeCV(alphas=self.alphas, cv=5)
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Compute per-voxel Pearson correlation
        correlations = []
        for i in range(y_test.shape[1]):
            r, _ = pearsonr(y_test[:, i], y_pred[:, i])
            if not np.isnan(r):
                correlations.append(r)
        
        if len(correlations) == 0:
            return 0.0
        
        return float(np.mean(correlations))
    
    def compute_model_roi_scores(
        self,
        model_name: str,
        subjects: List[int],
        hemispheres: List[str] = ["lh", "rh"]
    ) -> ROIBrainScore:
        """
        Compute ROI-specific brain scores for a model across subjects.
        
        Args:
            model_name: Name of the VLM
            subjects: List of subject IDs
            hemispheres: Hemispheres to include
            
        Returns:
            ROIBrainScore with per-ROI scores
        """
        # Collect scores per ROI category
        roi_scores_all: Dict[str, List[float]] = {roi: [] for roi in ROI_CATEGORIES}
        roi_scores_per_subject: Dict[str, Dict[int, float]] = {roi: {} for roi in ROI_CATEGORIES}
        
        total_fits = len(subjects) * len(hemispheres) * len(ROI_CATEGORIES)
        current_fit = 0
        
        for subject_id in subjects:
            logger.info(f"  Loading features for subject {subject_id}...")
            try:
                features = self.load_features(model_name, subject_id)
                logger.info(f"  Features shape: {features.shape}")
            except FileNotFoundError as e:
                logger.warning(f"Skipping subject {subject_id}: {e}")
                continue
            
            for hemi in hemispheres:
                try:
                    fmri = self.load_fmri(subject_id, hemi)
                    logger.info(f"  Subject {subject_id} {hemi}: fMRI shape {fmri.shape}")
                except FileNotFoundError as e:
                    logger.warning(f"Skipping {hemi} for subject {subject_id}: {e}")
                    continue
                
                # Align features and fMRI
                n_samples = min(features.shape[0], fmri.shape[0])
                features_aligned = features[:n_samples]
                fmri_aligned = fmri[:n_samples]
                
                for roi_category in ROI_CATEGORIES:
                    current_fit += 1
                    mask = self.load_roi_mask(subject_id, hemi, roi_category)
                    
                    if mask is None:
                        continue
                    
                    n_voxels = np.sum(mask > 0)
                    logger.info(f"    [{current_fit}/{total_fits}] {roi_category}: {n_voxels} voxels...")
                    
                    score = self.compute_roi_score(features_aligned, fmri_aligned, mask)
                    logger.info(f"    [{current_fit}/{total_fits}] {roi_category}: r={score:.4f}")
                    
                    roi_scores_all[roi_category].append(score)
                    
                    # Store per-subject (aggregate across hemispheres)
                    if subject_id not in roi_scores_per_subject[roi_category]:
                        roi_scores_per_subject[roi_category][subject_id] = []
                    roi_scores_per_subject[roi_category][subject_id].append(score)
        
        # Aggregate per-subject scores
        for roi_category in ROI_CATEGORIES:
            for subject_id in roi_scores_per_subject[roi_category]:
                scores = roi_scores_per_subject[roi_category][subject_id]
                if scores:
                    roi_scores_per_subject[roi_category][subject_id] = float(np.mean(scores))
        
        # Compute mean scores per ROI
        roi_scores = {}
        for roi_category in ROI_CATEGORIES:
            if roi_scores_all[roi_category]:
                roi_scores[roi_category] = float(np.mean(roi_scores_all[roi_category]))
            else:
                roi_scores[roi_category] = 0.0
        
        # Overall score (mean across ROIs)
        overall = float(np.mean(list(roi_scores.values()))) if roi_scores else 0.0
        
        return ROIBrainScore(
            model_name=model_name,
            roi_scores=roi_scores,
            roi_scores_per_subject=roi_scores_per_subject,
            overall_score=overall,
            n_subjects=len(subjects)
        )
    
    def save_roi_scores(self, roi_score: ROIBrainScore):
        """Save ROI brain scores to JSON file."""
        output_path = self.output_dir / "roi_brain_scores" / f"{roi_score.model_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(roi_score.to_dict(), f, indent=2)
        
        logger.info(f"Saved ROI brain scores to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute ROI-specific brain scores")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Models to process"
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Subject IDs"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default="results/features",
        help="Path to extracted features"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    data_dir = PROJECT_ROOT / args.data_dir
    features_dir = PROJECT_ROOT / args.features_dir
    output_dir = PROJECT_ROOT / args.output_dir
    
    # Determine models
    if "all" in args.models:
        models = AVAILABLE_MODELS
    else:
        models = [m for m in args.models if m in AVAILABLE_MODELS]
    
    logger.info(f"Computing ROI brain scores for {len(models)} models")
    logger.info(f"Subjects: {args.subjects}")
    logger.info(f"ROI categories: {ROI_CATEGORIES}")
    
    # Initialize computer
    computer = ROIBrainScoreComputer(
        data_dir=data_dir,
        features_dir=features_dir,
        output_dir=output_dir
    )
    
    # Compute for each model
    all_results = {}
    
    for model_name in models:
        logger.info(f"\nProcessing {model_name}...")
        
        try:
            result = computer.compute_model_roi_scores(
                model_name=model_name,
                subjects=args.subjects
            )
            
            computer.save_roi_scores(result)
            all_results[model_name] = result
            
            # Print summary
            logger.info(f"{model_name} ROI scores:")
            for roi, score in result.roi_scores.items():
                logger.info(f"  {roi}: {score:.4f}")
            
        except Exception as e:
            logger.error(f"Failed for {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save aggregated summary
    summary_path = output_dir / "roi_brain_scores" / "summary.json"
    summary = {}
    for model_name, result in all_results.items():
        summary[model_name] = result.roi_scores
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nSaved summary to {summary_path}")
    
    # Print final summary table
    logger.info("\n" + "=" * 80)
    logger.info("ROI BRAIN SCORE SUMMARY")
    logger.info("=" * 80)
    
    # Header
    header = f"{'Model':<20}"
    for roi in ROI_CATEGORIES:
        header += f" {roi[:12]:>12}"
    logger.info(header)
    logger.info("-" * 80)
    
    # Data rows
    for model_name, result in sorted(all_results.items()):
        row = f"{model_name:<20}"
        for roi in ROI_CATEGORIES:
            score = result.roi_scores.get(roi, 0.0)
            row += f" {score:>12.4f}"
        logger.info(row)


if __name__ == "__main__":
    main()
