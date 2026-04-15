"""
Brain Score Computation Module

This module orchestrates the complete brain score computation pipeline:
1. Load pre-extracted features and fMRI data
2. Compute noise ceilings
3. Fit Ridge encoders
4. Compute raw and normalized brain scores
5. Aggregate across subjects

The brain score represents how well a VLM's visual representations
predict human brain activity during natural scene viewing.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor
import logging

from .ridge_encoder import RidgeEncoder, EncodingResult, EncodingConfig, compute_random_baseline
from .noise_ceiling import NoiseCeilingEstimator, NoiseCeiling

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BrainScore:
    """Container for aggregated brain score results."""
    model_name: str
    overall_score: float  # Mean across subjects and hemispheres
    overall_normalized: Optional[float]
    per_subject: Dict[int, Dict[str, float]]  # {subj_id: {'lh': score, 'rh': score}}
    mean: float
    std: float
    median: float
    n_subjects: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "overall_score": float(self.overall_score),
            "overall_normalized": float(self.overall_normalized) if self.overall_normalized else None,
            "per_subject": {
                str(k): v for k, v in self.per_subject.items()
            },
            "mean": float(self.mean),
            "std": float(self.std),
            "median": float(self.median),
            "n_subjects": self.n_subjects
        }


class BrainScoreComputer:
    """
    Computes brain scores for VLM models against fMRI data.
    
    Orchestrates the complete pipeline from features to final scores.
    
    Example:
        >>> computer = BrainScoreComputer(data_dir="./data", features_dir="./results/features")
        >>> score = computer.compute_brain_score("llava_7b", subjects=[1, 2, 3])
        >>> print(f"LLaVA-7B Brain Score: {score.overall_score:.4f}")
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        features_dir: Union[str, Path],
        output_dir: Union[str, Path],
        encoding_config: Optional[EncodingConfig] = None
    ):
        """
        Initialize the brain score computer.
        
        Args:
            data_dir: Path to Algonauts data directory
            features_dir: Path to extracted features directory
            output_dir: Path to save results
            encoding_config: Configuration for Ridge encoding
        """
        self.data_dir = Path(data_dir)
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.encoding_config = encoding_config or EncodingConfig()
        self.encoder = RidgeEncoder(self.encoding_config)
        self.ceiling_estimator = NoiseCeilingEstimator()
        
        # Cache for loaded data
        self._fmri_cache: Dict[str, np.ndarray] = {}
        self._ceiling_cache: Dict[str, NoiseCeiling] = {}
        
        # Lazy load data loader
        self._data_loader = None
    
    @property
    def data_loader(self):
        """Lazy load data loader."""
        if self._data_loader is None:
            from src.utils.data_loader import AlgonautsDataLoader
            self._data_loader = AlgonautsDataLoader(self.data_dir)
        return self._data_loader
    
    def _get_fmri(self, subject_id: int, hemisphere: str) -> np.ndarray:
        """Get fMRI data with caching."""
        cache_key = f"{subject_id}_{hemisphere}"
        
        if cache_key not in self._fmri_cache:
            self._fmri_cache[cache_key] = self.data_loader.load_fmri_by_hemisphere(
                subject_id, hemisphere
            )
        
        return self._fmri_cache[cache_key]
    
    def _get_noise_ceiling(self, subject_id: int, hemisphere: str) -> NoiseCeiling:
        """Get or compute noise ceiling with caching."""
        cache_key = f"{subject_id}_{hemisphere}"
        
        if cache_key not in self._ceiling_cache:
            # Check if saved ceiling exists
            ceiling_path = self.output_dir / "noise_ceilings" / f"noise_ceiling_subj{subject_id:02d}_{hemisphere}.npz"
            
            if ceiling_path.exists():
                self._ceiling_cache[cache_key] = NoiseCeilingEstimator.load_ceiling(ceiling_path)
            else:
                # Compute and save
                fmri = self._get_fmri(subject_id, hemisphere)
                ceiling = self.ceiling_estimator.estimate_split_half(fmri)
                
                ceiling_dir = self.output_dir / "noise_ceilings"
                ceiling_dir.mkdir(parents=True, exist_ok=True)
                self.ceiling_estimator.save_ceiling(ceiling, ceiling_dir, subject_id, hemisphere)
                
                self._ceiling_cache[cache_key] = ceiling
        
        return self._ceiling_cache[cache_key]
    
    def _load_features(self, model_name: str, subject_id: int) -> np.ndarray:
        """
        Load pre-extracted features for a model and subject.
        
        Features MUST be saved per-subject because each subject has different
        numbers of training images:
            - Subjects 1,2,5,7: 9841 images
            - Subjects 3,6: 9082 images  
            - Subjects 4,8: 8779 images
        
        Args:
            model_name: VLM model name
            subject_id: Subject ID (1-8)
            
        Returns:
            Feature array of shape (n_images, feature_dim)
        """
        # Subject-specific features file (required for proper alignment)
        subject_pattern = f"{model_name}_subj{subject_id:02d}_features.npz"
        feature_path = self.features_dir / subject_pattern
        
        if feature_path.exists():
            data = np.load(feature_path)
            features = data['features']
            logger.debug(f"Loaded features for {model_name} subject {subject_id}: shape {features.shape}")
            return features
        
        # Fallback: shared features file (only valid if all subjects have same image count)
        # This is NOT recommended for Algonauts data due to varying image counts
        shared_pattern = f"{model_name}_features.npz"
        shared_path = self.features_dir / shared_pattern
        
        if shared_path.exists():
            logger.warning(
                f"Using shared features file for {model_name}. "
                f"This may cause alignment issues with subject {subject_id}!"
            )
            data = np.load(shared_path)
            return data['features']
        
        raise FileNotFoundError(
            f"Features not found for {model_name} subject {subject_id}. "
            f"Expected: {feature_path}\n"
            f"Run: python scripts/01_extract_features.py --models {model_name} --subjects {subject_id}"
        )
    
    def compute_brain_score_single(
        self,
        model_name: str,
        subject_id: int,
        hemisphere: str,
        use_normalized: bool = True
    ) -> EncodingResult:
        """
        Compute brain score for a single subject/hemisphere.
        
        Args:
            model_name: Name of the VLM
            subject_id: Subject ID
            hemisphere: 'lh' or 'rh'
            use_normalized: Whether to normalize by noise ceiling
            
        Returns:
            EncodingResult with performance metrics
        """
        # Load features
        features = self._load_features(model_name, subject_id)
        
        # Load fMRI
        fmri = self._get_fmri(subject_id, hemisphere)
        
        # Get noise ceiling if normalizing
        noise_ceiling = None
        if use_normalized:
            ceiling = self._get_noise_ceiling(subject_id, hemisphere)
            noise_ceiling = ceiling.per_voxel
        
        # Fit and evaluate
        result = self.encoder.fit_and_evaluate(
            features=features,
            fmri=fmri,
            model_name=model_name,
            subject_id=subject_id,
            hemisphere=hemisphere,
            noise_ceiling=noise_ceiling
        )
        
        # Save result
        results_dir = self.output_dir / "encoding_results" / model_name
        results_dir.mkdir(parents=True, exist_ok=True)
        self.encoder.save_result(result, results_dir)
        
        return result
    
    def compute_brain_score(
        self,
        model_name: str,
        subjects: Optional[List[int]] = None,
        hemispheres: List[str] = ["lh", "rh"],
        use_normalized: bool = True
    ) -> BrainScore:
        """
        Compute aggregated brain score across subjects and hemispheres.
        
        Args:
            model_name: Name of the VLM
            subjects: List of subject IDs (default: all available)
            hemispheres: List of hemispheres to include
            use_normalized: Whether to normalize by noise ceiling
            
        Returns:
            BrainScore object with aggregated results
        """
        subjects = subjects or self.data_loader.subjects
        
        logger.info(f"Computing brain score for {model_name}")
        logger.info(f"  Subjects: {subjects}")
        logger.info(f"  Hemispheres: {hemispheres}")
        
        per_subject = {}
        all_scores = []
        all_normalized = []
        
        for subj_id in subjects:
            per_subject[subj_id] = {}
            
            for hemi in hemispheres:
                result = self.compute_brain_score_single(
                    model_name=model_name,
                    subject_id=subj_id,
                    hemisphere=hemi,
                    use_normalized=use_normalized
                )
                
                per_subject[subj_id][hemi] = result.raw_score
                all_scores.append(result.raw_score)
                
                if result.normalized_score is not None:
                    per_subject[subj_id][f"{hemi}_normalized"] = result.normalized_score
                    all_normalized.append(result.normalized_score)
        
        # Compute aggregates
        overall_score = float(np.mean(all_scores))
        overall_normalized = float(np.mean(all_normalized)) if all_normalized else None
        
        brain_score = BrainScore(
            model_name=model_name,
            overall_score=overall_score,
            overall_normalized=overall_normalized,
            per_subject=per_subject,
            mean=overall_score,
            std=float(np.std(all_scores)),
            median=float(np.median(all_scores)),
            n_subjects=len(subjects)
        )
        
        # Save aggregated result
        self._save_brain_score(brain_score)
        
        logger.info(f"Brain score for {model_name}:")
        logger.info(f"  Overall: {brain_score.overall_score:.4f}")
        if brain_score.overall_normalized:
            logger.info(f"  Normalized: {brain_score.overall_normalized:.4f}")
        
        return brain_score
    
    def _save_brain_score(self, brain_score: BrainScore) -> Path:
        """Save brain score to JSON file."""
        output_dir = self.output_dir / "brain_scores"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{brain_score.model_name}.json"
        
        with open(output_path, 'w') as f:
            json.dump(brain_score.to_dict(), f, indent=2)
        
        logger.info(f"Saved brain score to {output_path}")
        return output_path
    
    def compute_random_baseline(
        self,
        subjects: Optional[List[int]] = None,
        feature_dim: int = 768,
        n_runs: int = 10
    ) -> Dict[str, float]:
        """
        Compute random feature baseline for comparison.
        
        Args:
            subjects: List of subject IDs
            feature_dim: Dimension of random features
            n_runs: Number of random runs
            
        Returns:
            Dictionary with baseline statistics
        """
        subjects = subjects or self.data_loader.subjects[:1]  # Use first subject by default
        
        all_baselines = []
        
        for subj_id in subjects:
            for hemi in ["lh", "rh"]:
                fmri = self._get_fmri(subj_id, hemi)
                baseline = compute_random_baseline(
                    fmri=fmri,
                    feature_dim=feature_dim,
                    n_runs=n_runs
                )
                all_baselines.append(baseline["mean"])
        
        result = {
            "mean": float(np.mean(all_baselines)),
            "std": float(np.std(all_baselines)),
            "feature_dim": feature_dim,
            "n_runs": n_runs,
            "subjects": subjects
        }
        
        # Save baseline
        output_path = self.output_dir / "brain_scores" / "random_baseline.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Random baseline: {result['mean']:.4f} ± {result['std']:.4f}")
        
        return result
    
    def compute_all_models(
        self,
        model_names: List[str],
        subjects: Optional[List[int]] = None,
        save_summary: bool = True
    ) -> Dict[str, BrainScore]:
        """
        Compute brain scores for multiple models.
        
        Args:
            model_names: List of model names
            subjects: List of subject IDs
            save_summary: Whether to save summary CSV
            
        Returns:
            Dictionary mapping model names to BrainScore objects
        """
        results = {}
        
        for model_name in model_names:
            try:
                results[model_name] = self.compute_brain_score(
                    model_name=model_name,
                    subjects=subjects
                )
            except Exception as e:
                logger.error(f"Failed to compute brain score for {model_name}: {e}")
                continue
        
        if save_summary:
            self._save_summary(results)
        
        return results
    
    def _save_summary(self, results: Dict[str, BrainScore]) -> Path:
        """Save summary of all brain scores."""
        import pandas as pd
        
        summary_data = []
        for model_name, score in results.items():
            summary_data.append({
                "model": model_name,
                "brain_score": score.overall_score,
                "brain_score_normalized": score.overall_normalized,
                "std": score.std,
                "n_subjects": score.n_subjects
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values("brain_score", ascending=False)
        
        output_path = self.output_dir / "brain_scores" / "summary.csv"
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved summary to {output_path}")
        return output_path
    
    @staticmethod
    def load_brain_score(path: Union[str, Path]) -> BrainScore:
        """Load brain score from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Convert per_subject keys back to int
        per_subject = {int(k): v for k, v in data["per_subject"].items()}
        
        return BrainScore(
            model_name=data["model_name"],
            overall_score=data["overall_score"],
            overall_normalized=data.get("overall_normalized"),
            per_subject=per_subject,
            mean=data["mean"],
            std=data["std"],
            median=data["median"],
            n_subjects=data["n_subjects"]
        )
