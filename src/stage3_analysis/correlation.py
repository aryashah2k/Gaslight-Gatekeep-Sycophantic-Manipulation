"""
Correlation Analysis Module

This module performs statistical analysis to test whether neural alignment
(Brain-Score) correlates with resistance to sycophancy.

Includes:
- Pearson and Spearman correlations
- Bootstrap confidence intervals
- Partial correlations controlling for confounds (model size)
- Statistical significance testing
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats
from sklearn.linear_model import LinearRegression
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Container for correlation analysis results."""
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    n_models: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "pearson_r": float(self.pearson_r),
            "pearson_p": float(self.pearson_p),
            "spearman_rho": float(self.spearman_rho),
            "spearman_p": float(self.spearman_p),
            "ci_lower": float(self.ci_lower),
            "ci_upper": float(self.ci_upper),
            "ci_level": float(self.ci_level),
            "n_models": self.n_models
        }


@dataclass
class PartialCorrelationResult:
    """Container for partial correlation results."""
    partial_r: float
    partial_p: float
    confounds: List[str]
    n_models: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "partial_r": float(self.partial_r),
            "partial_p": float(self.partial_p),
            "confounds": self.confounds,
            "n_models": self.n_models
        }


@dataclass
class FullAnalysisResult:
    """Container for complete analysis results."""
    main_correlation: CorrelationResult
    partial_correlation: PartialCorrelationResult
    models: List[str]
    brain_scores: List[float]
    sycophancy_rates: List[float]
    model_sizes: List[float]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "main_correlation": self.main_correlation.to_dict(),
            "partial_correlation": self.partial_correlation.to_dict(),
            "models": self.models,
            "brain_scores": self.brain_scores,
            "sycophancy_rates": self.sycophancy_rates,
            "model_sizes": self.model_sizes
        }


# Model metadata for confound analysis
MODEL_METADATA = {
    "qwen2vl_2b": {"params_b": 2.0, "vision_family": "custom_vit"},
    "qwen25vl_3b": {"params_b": 3.0, "vision_family": "custom_vit"},
    "gemma3_1b": {"params_b": 4.0, "vision_family": "siglip"},
    "llava_7b": {"params_b": 7.0, "vision_family": "clip"},
    "paligemma2_10b": {"params_b": 10.0, "vision_family": "siglip"},  # Replaces moondream2/internvl2_2b
    "blip2_opt27b": {"params_b": 2.7, "vision_family": "vit"},       # Replaces minicpm_v26
    "lfm2vl_8b": {"params_b": 0.45, "vision_family": "custom"},
    "lfm25vl_1b": {"params_b": 1.6, "vision_family": "custom"},
    "idefics2_8b": {"params_b": 8.0, "vision_family": "siglip"},
    "phi35_vision": {"params_b": 4.2, "vision_family": "clip"},
    "smolvlm_256m": {"params_b": 0.256, "vision_family": "siglip"},
    "smolvlm_500m": {"params_b": 0.5, "vision_family": "siglip"},
}


class CorrelationAnalyzer:
    """
    Performs correlation analysis between brain scores and sycophancy rates.
    
    Tests the hypothesis that models with higher neural alignment
    (Brain-Score) are more resistant to sycophantic behavior.
    
    Example:
        >>> analyzer = CorrelationAnalyzer(
        ...     brain_scores_dir="results/brain_scores",
        ...     sycophancy_dir="results/sycophancy"
        ... )
        >>> result = analyzer.run_full_analysis()
        >>> print(f"Correlation: r={result.main_correlation.pearson_r:.3f}")
    """
    
    def __init__(
        self,
        brain_scores_dir: Union[str, Path],
        sycophancy_dir: Union[str, Path],
        output_dir: Union[str, Path] = "results/analysis"
    ):
        """
        Initialize the analyzer.
        
        Args:
            brain_scores_dir: Directory containing brain score JSONs
            sycophancy_dir: Directory containing sycophancy results
            output_dir: Directory to save analysis results
        """
        self.brain_scores_dir = Path(brain_scores_dir)
        self.sycophancy_dir = Path(sycophancy_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_brain_scores(self) -> Dict[str, float]:
        """
        Load brain scores for all models.
        
        Returns:
            Dictionary mapping model names to brain scores
        """
        scores = {}
        
        for json_file in self.brain_scores_dir.glob("*.json"):
            if json_file.stem == "random_baseline" or json_file.stem == "summary":
                continue
            
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            model_name = data.get("model_name", json_file.stem)
            
            # Use normalized score if available, otherwise raw
            if data.get("overall_normalized"):
                scores[model_name] = data["overall_normalized"]
            else:
                scores[model_name] = data["overall_score"]
        
        logger.info(f"Loaded brain scores for {len(scores)} models")
        return scores
    
    def load_sycophancy_rates(self) -> Dict[str, float]:
        """
        Load sycophancy rates for all models.
        
        Returns:
            Dictionary mapping model names to sycophancy rates
        """
        rates = {}
        
        for model_dir in self.sycophancy_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            metrics_path = model_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            
            with open(metrics_path, 'r') as f:
                data = json.load(f)
            
            rates[model_dir.name] = data["sycophancy_rate"]
        
        logger.info(f"Loaded sycophancy rates for {len(rates)} models")
        return rates
    
    def compute_correlation_with_bootstrap_ci(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_bootstrap: int = 10000,
        ci_level: float = 0.95
    ) -> CorrelationResult:
        """
        Compute Pearson/Spearman correlation with bootstrap CI.
        
        Args:
            x: Brain scores array
            y: Sycophancy rates array
            n_bootstrap: Number of bootstrap samples
            ci_level: Confidence level (default 0.95)
            
        Returns:
            CorrelationResult with correlations and CI
        """
        n = len(x)
        
        # Point estimates
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_rho, spearman_p = stats.spearmanr(x, y)
        
        # Bootstrap confidence interval for Pearson r
        np.random.seed(42)
        bootstrap_rs = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            boot_r, _ = stats.pearsonr(x[indices], y[indices])
            bootstrap_rs.append(boot_r)
        
        bootstrap_rs = np.array(bootstrap_rs)
        alpha = 1 - ci_level
        ci_lower = float(np.percentile(bootstrap_rs, 100 * alpha / 2))
        ci_upper = float(np.percentile(bootstrap_rs, 100 * (1 - alpha / 2)))
        
        return CorrelationResult(
            pearson_r=float(pearson_r),
            pearson_p=float(pearson_p),
            spearman_rho=float(spearman_rho),
            spearman_p=float(spearman_p),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=ci_level,
            n_models=n
        )
    
    def compute_partial_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        confounds: np.ndarray,
        confound_names: List[str]
    ) -> PartialCorrelationResult:
        """
        Compute partial correlation controlling for confounds.
        
        Args:
            x: Brain scores array
            y: Sycophancy rates array
            confounds: Confound matrix (n_models, n_confounds)
            confound_names: Names of confound variables
            
        Returns:
            PartialCorrelationResult
        """
        # Ensure 2D
        if confounds.ndim == 1:
            confounds = confounds.reshape(-1, 1)
        
        # Regress out confounds from x
        reg_x = LinearRegression().fit(confounds, x)
        residual_x = x - reg_x.predict(confounds)
        
        # Regress out confounds from y
        reg_y = LinearRegression().fit(confounds, y)
        residual_y = y - reg_y.predict(confounds)
        
        # Correlation of residuals
        partial_r, partial_p = stats.pearsonr(residual_x, residual_y)
        
        return PartialCorrelationResult(
            partial_r=float(partial_r),
            partial_p=float(partial_p),
            confounds=confound_names,
            n_models=len(x)
        )
    
    def run_full_analysis(
        self,
        n_bootstrap: int = 10000
    ) -> FullAnalysisResult:
        """
        Run complete correlation analysis.
        
        Args:
            n_bootstrap: Number of bootstrap samples for CI
            
        Returns:
            FullAnalysisResult with all analysis outputs
        """
        # Load data
        brain_scores = self.load_brain_scores()
        sycophancy_rates = self.load_sycophancy_rates()
        
        # Find common models
        common_models = sorted(
            set(brain_scores.keys()) & set(sycophancy_rates.keys())
        )
        
        if len(common_models) < 3:
            raise ValueError(
                f"Need at least 3 models for correlation analysis. "
                f"Found {len(common_models)} common models."
            )
        
        logger.info(f"Analyzing {len(common_models)} models: {common_models}")
        
        # Build arrays
        x = np.array([brain_scores[m] for m in common_models])
        y = np.array([sycophancy_rates[m] for m in common_models])
        
        # Get model sizes
        sizes = []
        for m in common_models:
            if m in MODEL_METADATA:
                sizes.append(MODEL_METADATA[m]["params_b"])
            else:
                logger.warning(f"No metadata for {m}, using default size 1.0")
                sizes.append(1.0)
        sizes = np.array(sizes)
        
        # Use log-transformed sizes for analysis
        log_sizes = np.log10(sizes + 0.1)
        
        # Main correlation
        main_result = self.compute_correlation_with_bootstrap_ci(
            x, y, n_bootstrap=n_bootstrap
        )
        
        logger.info(f"Main correlation: r={main_result.pearson_r:.3f} "
                   f"[{main_result.ci_lower:.3f}, {main_result.ci_upper:.3f}]")
        
        # Partial correlation controlling for model size
        partial_result = self.compute_partial_correlation(
            x, y, log_sizes, ["log_model_size"]
        )
        
        logger.info(f"Partial correlation (controlling size): r={partial_result.partial_r:.3f}")
        
        # Create full result
        result = FullAnalysisResult(
            main_correlation=main_result,
            partial_correlation=partial_result,
            models=common_models,
            brain_scores=x.tolist(),
            sycophancy_rates=y.tolist(),
            model_sizes=sizes.tolist()
        )
        
        # Save result
        self._save_result(result)
        
        return result
    
    def _save_result(self, result: FullAnalysisResult) -> Path:
        """Save analysis result to disk."""
        output_path = self.output_dir / "correlation_analysis.json"
        
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.info(f"Saved analysis to {output_path}")
        return output_path
    
    def interpret_result(self, result: FullAnalysisResult) -> str:
        """
        Provide interpretation of the correlation results.
        
        Args:
            result: Analysis result
            
        Returns:
            Interpretation string
        """
        r = result.main_correlation.pearson_r
        p = result.main_correlation.pearson_p
        ci_low = result.main_correlation.ci_lower
        ci_high = result.main_correlation.ci_upper
        
        partial_r = result.partial_correlation.partial_r
        n = result.main_correlation.n_models
        
        # Effect size interpretation
        abs_r = abs(r)
        if abs_r < 0.1:
            effect = "negligible"
        elif abs_r < 0.3:
            effect = "small"
        elif abs_r < 0.5:
            effect = "medium"
        else:
            effect = "large"
        
        # Direction
        direction = "negative" if r < 0 else "positive"
        
        # Significance
        sig = "significant" if p < 0.05 else "not significant"
        
        interpretation = f"""
CORRELATION ANALYSIS INTERPRETATION
===================================

Main Finding:
- Pearson r = {r:.3f} (95% CI: [{ci_low:.3f}, {ci_high:.3f}])
- p-value = {p:.4f} ({sig} at α=0.05)
- Effect size: {effect}
- Direction: {direction} correlation

After controlling for model size:
- Partial r = {partial_r:.3f}

Interpretation:
"""
        
        if r < -0.3 and p < 0.1:
            interpretation += """
Models with higher Brain-Score (better neural alignment with human visual 
cortex) tend to show LOWER sycophancy rates. This supports the hypothesis 
that biological alignment may serve as a proxy for visual robustness.
"""
        elif r > 0.3 and p < 0.1:
            interpretation += """
Unexpectedly, models with higher Brain-Score show HIGHER sycophancy rates.
This could indicate that modeling human vision does not necessarily confer
resistance to manipulation, or that other factors are at play.
"""
        else:
            interpretation += """
The correlation between Brain-Score and sycophancy rate is weak or not
statistically significant. Neural alignment and sycophancy resistance 
may be independent properties, or our sample size may be insufficient
to detect the relationship.
"""
        
        interpretation += f"""
Caveats:
- Sample size: n={n} models (limited statistical power)
- Correlation does not imply causation
- Model selection may introduce bias
"""
        
        return interpretation
    
    @staticmethod
    def load_result(path: Union[str, Path]) -> FullAnalysisResult:
        """Load analysis result from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        return FullAnalysisResult(
            main_correlation=CorrelationResult(**data["main_correlation"]),
            partial_correlation=PartialCorrelationResult(**data["partial_correlation"]),
            models=data["models"],
            brain_scores=data["brain_scores"],
            sycophancy_rates=data["sycophancy_rates"],
            model_sizes=data["model_sizes"]
        )
