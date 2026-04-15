"""
Ridge Regression Encoder for fMRI Prediction

This module implements Ridge regression for mapping VLM visual features
to fMRI voxel responses. The R² score on held-out data serves as the
"Brain-Score" metric.

Ridge regression is used because:
1. It handles high-dimensional features well
2. L2 regularization prevents overfitting
3. Computationally efficient (no GPU needed)
4. Standard in neuroscience encoding literature

References:
    - Naselaris et al. (2011) - Encoding and decoding in fMRI
    - Schrimpf et al. (2018) - Brain-Score benchmark
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import train_test_split, KFold
from scipy.stats import pearsonr
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EncodingResult:
    """Container for encoding model results."""
    model_name: str
    subject_id: int
    hemisphere: str
    raw_score: float  # Mean Pearson r across voxels
    normalized_score: Optional[float]  # Score / noise ceiling
    per_voxel_scores: np.ndarray  # Per-voxel Pearson r
    best_alpha: float  # Selected regularization strength
    n_voxels: int
    n_train: int
    n_test: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "subject_id": self.subject_id,
            "hemisphere": self.hemisphere,
            "raw_score": float(self.raw_score),
            "normalized_score": float(self.normalized_score) if self.normalized_score else None,
            "best_alpha": float(self.best_alpha),
            "n_voxels": self.n_voxels,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "score_std": float(np.std(self.per_voxel_scores)),
            "score_median": float(np.median(self.per_voxel_scores))
        }


@dataclass
class EncodingConfig:
    """Configuration for Ridge encoding."""
    alphas: List[float] = field(default_factory=lambda: [0.1, 1, 10, 100, 1000, 10000])
    test_size: float = 0.2
    n_cv_folds: int = 5
    random_state: int = 42
    n_jobs: int = -1  # Use all cores for CV
    store_coefficients: bool = False


class RidgeEncoder:
    """
    Ridge regression encoder for fMRI prediction.
    
    Maps visual features from VLMs to fMRI voxel responses using
    L2-regularized linear regression.
    
    Attributes:
        config: Encoding configuration
        fitted_models: Dictionary of fitted Ridge models
    
    Example:
        >>> encoder = RidgeEncoder()
        >>> features = np.random.randn(1000, 768)  # VLM features
        >>> fmri = np.random.randn(1000, 10000)     # fMRI data
        >>> result = encoder.fit_and_evaluate(features, fmri, "model_x", 1, "lh")
        >>> print(f"Brain-Score: {result.raw_score:.4f}")
    """
    
    def __init__(self, config: Optional[EncodingConfig] = None):
        """
        Initialize the encoder.
        
        Args:
            config: Encoding configuration
        """
        self.config = config or EncodingConfig()
        self.fitted_models: Dict[str, Ridge] = {}
        self._coefficients: Optional[np.ndarray] = None
    
    def fit_and_evaluate(
        self,
        features: np.ndarray,
        fmri: np.ndarray,
        model_name: str,
        subject_id: int,
        hemisphere: str,
        noise_ceiling: Optional[np.ndarray] = None
    ) -> EncodingResult:
        """
        Fit Ridge encoder and evaluate on held-out data.
        
        Args:
            features: Visual features of shape (n_images, feature_dim)
            fmri: fMRI responses of shape (n_images, n_voxels)
            model_name: Name of the VLM
            subject_id: Subject ID
            hemisphere: 'lh' or 'rh'
            noise_ceiling: Optional noise ceiling per voxel for normalization
            
        Returns:
            EncodingResult with performance metrics
        """
        # Validate inputs
        assert features.shape[0] == fmri.shape[0], \
            f"Feature and fMRI sample count mismatch: {features.shape[0]} vs {fmri.shape[0]}"
        
        n_images = features.shape[0]
        n_voxels = fmri.shape[1]
        
        logger.info(f"Fitting encoder: {model_name} | Subject {subject_id} | {hemisphere}")
        logger.info(f"  Features: {features.shape}, fMRI: {fmri.shape}")
        
        # Handle NaN and Inf values in features
        nan_count = np.sum(np.isnan(features))
        inf_count = np.sum(np.isinf(features))
        if nan_count > 0 or inf_count > 0:
            logger.warning(f"  Found {nan_count} NaN and {inf_count} Inf values in features, replacing with 0")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Handle NaN and Inf values in fMRI data
        fmri_nan_count = np.sum(np.isnan(fmri))
        fmri_inf_count = np.sum(np.isinf(fmri))
        if fmri_nan_count > 0 or fmri_inf_count > 0:
            logger.warning(f"  Found {fmri_nan_count} NaN and {fmri_inf_count} Inf values in fMRI, replacing with 0")
            fmri = np.nan_to_num(fmri, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            features, fmri,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        
        logger.info(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        
        # Fit RidgeCV with cross-validation for alpha selection
        ridge = RidgeCV(
            alphas=self.config.alphas,
            cv=self.config.n_cv_folds,
            scoring='r2'
        )
        
        ridge.fit(X_train, y_train)
        best_alpha = ridge.alpha_
        
        logger.info(f"  Best alpha: {best_alpha}")
        
        # Predict on test set
        y_pred = ridge.predict(X_test)
        
        # Compute per-voxel Pearson correlation
        per_voxel_scores = np.zeros(n_voxels)
        for v in range(n_voxels):
            if np.std(y_test[:, v]) > 1e-8 and np.std(y_pred[:, v]) > 1e-8:
                r, _ = pearsonr(y_test[:, v], y_pred[:, v])
                per_voxel_scores[v] = r
            else:
                per_voxel_scores[v] = 0.0
        
        # Handle NaN values
        per_voxel_scores = np.nan_to_num(per_voxel_scores, nan=0.0)
        
        # Mean score across voxels
        raw_score = float(np.mean(per_voxel_scores))
        
        # Normalize by noise ceiling if provided
        normalized_score = None
        if noise_ceiling is not None:
            # Avoid division by very small values
            valid_mask = noise_ceiling > 0.01
            if valid_mask.any():
                normalized_scores = per_voxel_scores[valid_mask] / noise_ceiling[valid_mask]
                normalized_scores = np.clip(normalized_scores, 0, 1)
                normalized_score = float(np.mean(normalized_scores))
        
        # Store model if needed
        model_key = f"{model_name}_{subject_id}_{hemisphere}"
        self.fitted_models[model_key] = ridge
        
        # Store coefficients if configured
        if self.config.store_coefficients:
            self._coefficients = ridge.coef_
        
        logger.info(f"  Raw score: {raw_score:.4f}")
        if normalized_score:
            logger.info(f"  Normalized score: {normalized_score:.4f}")
        
        return EncodingResult(
            model_name=model_name,
            subject_id=subject_id,
            hemisphere=hemisphere,
            raw_score=raw_score,
            normalized_score=normalized_score,
            per_voxel_scores=per_voxel_scores,
            best_alpha=best_alpha,
            n_voxels=n_voxels,
            n_train=X_train.shape[0],
            n_test=X_test.shape[0]
        )
    
    def fit_and_evaluate_cv(
        self,
        features: np.ndarray,
        fmri: np.ndarray,
        model_name: str,
        subject_id: int,
        hemisphere: str,
        n_folds: int = 5,
        noise_ceiling: Optional[np.ndarray] = None
    ) -> Tuple[EncodingResult, np.ndarray]:
        """
        Fit and evaluate with cross-validation for more robust estimates.
        
        Args:
            features: Visual features
            fmri: fMRI responses
            model_name: Name of the VLM
            subject_id: Subject ID
            hemisphere: Hemisphere
            n_folds: Number of CV folds
            noise_ceiling: Optional noise ceiling
            
        Returns:
            Tuple of (mean EncodingResult, per-fold scores array)
        """
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.config.random_state)
        
        fold_scores = []
        all_per_voxel_scores = []
        all_alphas = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(features)):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = fmri[train_idx], fmri[test_idx]
            
            # Fit Ridge
            ridge = RidgeCV(
                alphas=self.config.alphas,
                cv=3  # Inner CV for alpha selection
            )
            ridge.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = ridge.predict(X_test)
            
            per_voxel = np.array([
                pearsonr(y_test[:, v], y_pred[:, v])[0] 
                if np.std(y_test[:, v]) > 1e-8 else 0.0
                for v in range(fmri.shape[1])
            ])
            per_voxel = np.nan_to_num(per_voxel, nan=0.0)
            
            fold_scores.append(np.mean(per_voxel))
            all_per_voxel_scores.append(per_voxel)
            all_alphas.append(ridge.alpha_)
            
            logger.debug(f"  Fold {fold + 1}: {fold_scores[-1]:.4f}")
        
        # Average across folds
        mean_per_voxel = np.mean(all_per_voxel_scores, axis=0)
        mean_score = np.mean(fold_scores)
        
        # Normalize if noise ceiling provided
        normalized_score = None
        if noise_ceiling is not None:
            valid_mask = noise_ceiling > 0.01
            if valid_mask.any():
                normalized = mean_per_voxel[valid_mask] / noise_ceiling[valid_mask]
                normalized = np.clip(normalized, 0, 1)
                normalized_score = float(np.mean(normalized))
        
        result = EncodingResult(
            model_name=model_name,
            subject_id=subject_id,
            hemisphere=hemisphere,
            raw_score=mean_score,
            normalized_score=normalized_score,
            per_voxel_scores=mean_per_voxel,
            best_alpha=float(np.median(all_alphas)),
            n_voxels=fmri.shape[1],
            n_train=int(len(features) * (1 - 1/n_folds)),
            n_test=int(len(features) / n_folds)
        )
        
        return result, np.array(fold_scores)
    
    def save_result(
        self,
        result: EncodingResult,
        output_dir: Union[str, Path],
        include_per_voxel: bool = False
    ) -> Path:
        """
        Save encoding result to disk.
        
        Args:
            result: EncodingResult to save
            output_dir: Output directory
            include_per_voxel: Whether to include per-voxel scores
            
        Returns:
            Path to saved file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        filename = f"{result.model_name}_subj{result.subject_id:02d}_{result.hemisphere}.json"
        output_path = output_dir / filename
        
        # Prepare data
        data = result.to_dict()
        
        if include_per_voxel:
            # Save per-voxel scores separately as npz
            voxel_path = output_dir / f"{result.model_name}_subj{result.subject_id:02d}_{result.hemisphere}_per_voxel.npz"
            np.savez_compressed(voxel_path, scores=result.per_voxel_scores)
            data["per_voxel_path"] = str(voxel_path)
        
        # Save JSON
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved result to {output_path}")
        return output_path


def compute_random_baseline(
    fmri: np.ndarray,
    feature_dim: int = 768,
    n_runs: int = 10,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Compute brain-score for random features as chance baseline.
    
    This provides a lower bound on brain-score that any model should exceed.
    
    Args:
        fmri: fMRI data of shape (n_images, n_voxels)
        feature_dim: Dimension of random features
        n_runs: Number of random runs to average
        test_size: Test set fraction
        random_state: Base random seed
        
    Returns:
        Dictionary with mean, std of random baseline scores
    """
    logger.info(f"Computing random baseline with dim={feature_dim}, runs={n_runs}")
    
    scores = []
    n_images = fmri.shape[0]
    
    for run in range(n_runs):
        np.random.seed(random_state + run)
        
        # Generate random features
        random_features = np.random.randn(n_images, feature_dim).astype(np.float32)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            random_features, fmri,
            test_size=test_size,
            random_state=random_state + run
        )
        
        # Fit simple Ridge
        ridge = Ridge(alpha=1000)
        ridge.fit(X_train, y_train)
        
        # Evaluate
        y_pred = ridge.predict(X_test)
        
        per_voxel = np.array([
            pearsonr(y_test[:, v], y_pred[:, v])[0]
            if np.std(y_test[:, v]) > 1e-8 else 0.0
            for v in range(fmri.shape[1])
        ])
        per_voxel = np.nan_to_num(per_voxel, nan=0.0)
        
        scores.append(np.mean(per_voxel))
    
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "runs": n_runs,
        "feature_dim": feature_dim
    }
