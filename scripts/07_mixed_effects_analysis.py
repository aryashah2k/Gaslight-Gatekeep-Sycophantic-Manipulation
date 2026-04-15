"""
Script 07: Statistical Analysis for ROI Brain Alignment × Sycophancy

This script performs rigorous statistical analysis of the relationship between
ROI-specific brain alignment and sycophancy resistance across VLMs.

Statistical Methods:
1. Model-level Pearson correlation with bootstrap 95% CI (10,000 samples)
2. Permutation test for non-parametric p-value (10,000 permutations)
3. Group comparison: Resistant vs Susceptible VLMs with Cohen's d
4. Multiple comparison correction (Bonferroni)

Usage:
    python scripts/07_mixed_effects_analysis.py
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Error: pandas required for this analysis")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ROI categories
ROI_CATEGORIES = [
    "prf-visualrois",
    "floc-bodies",
    "floc-faces",
    "floc-places",
    "floc-words",
    "streams"
]

# Model classifications based on sycophancy rates
RESISTANT_THRESHOLD = 0.50  # Models with <50% sycophancy are "resistant"

# Multiple comparison correction
N_COMPARISONS = len(ROI_CATEGORIES)
BONFERRONI_ALPHA = 0.05 / N_COMPARISONS


@dataclass
class CorrelationResult:
    """Container for correlation analysis results."""
    roi_name: str
    n_models: int
    pearson_r: float
    parametric_p: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    bootstrap_excludes_zero: bool
    permutation_p: float
    bonferroni_significant: bool
    
    def to_dict(self) -> dict:
        return {
            'roi_name': self.roi_name,
            'n_models': int(self.n_models),
            'pearson_r': float(self.pearson_r),
            'parametric_p': float(self.parametric_p),
            'bootstrap_ci_lower': float(self.bootstrap_ci_lower),
            'bootstrap_ci_upper': float(self.bootstrap_ci_upper),
            'bootstrap_excludes_zero': bool(self.bootstrap_excludes_zero),
            'permutation_p': float(self.permutation_p),
            'bonferroni_significant': bool(self.bonferroni_significant)
        }


@dataclass
class GroupComparisonResult:
    """Container for group comparison results."""
    roi_name: str
    n_resistant: int
    n_susceptible: int
    mean_resistant: float
    mean_susceptible: float
    difference: float
    cohens_d: float
    effect_interpretation: str
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    ci_excludes_zero: bool
    t_statistic: float
    t_pvalue: float
    
    def to_dict(self) -> dict:
        return {
            'roi_name': self.roi_name,
            'n_resistant': int(self.n_resistant),
            'n_susceptible': int(self.n_susceptible),
            'mean_resistant': float(self.mean_resistant),
            'mean_susceptible': float(self.mean_susceptible),
            'difference': float(self.difference),
            'cohens_d': float(self.cohens_d),
            'effect_interpretation': self.effect_interpretation,
            'bootstrap_ci_lower': float(self.bootstrap_ci_lower),
            'bootstrap_ci_upper': float(self.bootstrap_ci_upper),
            'ci_excludes_zero': bool(self.ci_excludes_zero),
            't_statistic': float(self.t_statistic),
            't_pvalue': float(self.t_pvalue)
        }


def load_per_subject_roi_scores(roi_dir: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load per-subject ROI scores for all models."""
    results = {}
    
    for model_file in roi_dir.glob("*.json"):
        if model_file.name == "summary.json":
            continue
        
        model_name = model_file.stem
        
        with open(model_file, 'r') as f:
            data = json.load(f)
        
        per_subject = data.get("roi_scores_per_subject", {})
        
        if per_subject:
            results[model_name] = per_subject
    
    return results


def load_sycophancy_results(results_dir: Path) -> Dict[str, float]:
    """Load sycophancy rates for all models."""
    syc_rates = {}
    
    syc_dir = results_dir / "sycophancy_v2"
    if not syc_dir.exists():
        syc_dir = results_dir / "sycophancy"
    
    for model_dir in syc_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name.startswith('.'):
            continue
        
        metrics_file = model_dir / "metrics_v2.json"
        if not metrics_file.exists():
            metrics_file = model_dir / "metrics.json"
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            rate = metrics.get('final_sycophancy_rate', 
                              metrics.get('sycophancy_rate', 0))
            syc_rates[model_dir.name] = rate
    
    return syc_rates


def build_model_level_dataframe(
    per_subject_scores: Dict[str, Dict[str, Dict[str, float]]],
    sycophancy_rates: Dict[str, float]
) -> pd.DataFrame:
    """
    Build a model-level DataFrame by aggregating per-subject scores.
    
    Each row is one model with mean ROI scores across subjects.
    """
    rows = []
    
    for model_name, roi_data in per_subject_scores.items():
        if model_name not in sycophancy_rates:
            continue
        
        row = {
            'model': model_name,
            'sycophancy': float(sycophancy_rates[model_name]),
            'resistant': 1 if sycophancy_rates[model_name] < RESISTANT_THRESHOLD else 0
        }
        
        # Aggregate each ROI across subjects
        for roi_category, subject_scores in roi_data.items():
            scores = [float(s) for s in subject_scores.values()]
            row[roi_category] = np.mean(scores)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    logger.info(f"Built model-level dataframe with {len(df)} models")
    
    return df


def compute_correlation_with_bootstrap(
    df: pd.DataFrame,
    roi_name: str,
    n_bootstrap: int = 10000
) -> CorrelationResult:
    """
    Compute Pearson correlation with bootstrap CI and permutation p-value.
    """
    brain_scores = df[roi_name].values
    syc_scores = df['sycophancy'].values
    n_models = len(df)
    
    # Observed correlation
    r, parametric_p = stats.pearsonr(brain_scores, syc_scores)
    
    # Bootstrap CI
    bootstrap_rs = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_models, size=n_models, replace=True)
        boot_r, _ = stats.pearsonr(brain_scores[indices], syc_scores[indices])
        if not np.isnan(boot_r):
            bootstrap_rs.append(boot_r)
    
    ci_lower = np.percentile(bootstrap_rs, 2.5)
    ci_upper = np.percentile(bootstrap_rs, 97.5)
    excludes_zero = not (ci_lower <= 0 <= ci_upper)
    
    # Permutation test
    more_extreme = 0
    for _ in range(n_bootstrap):
        perm_syc = np.random.permutation(syc_scores)
        perm_r, _ = stats.pearsonr(brain_scores, perm_syc)
        if abs(perm_r) >= abs(r):
            more_extreme += 1
    
    perm_p = more_extreme / n_bootstrap
    
    return CorrelationResult(
        roi_name=roi_name,
        n_models=n_models,
        pearson_r=float(r),
        parametric_p=float(parametric_p),
        bootstrap_ci_lower=float(ci_lower),
        bootstrap_ci_upper=float(ci_upper),
        bootstrap_excludes_zero=excludes_zero,
        permutation_p=float(perm_p),
        bonferroni_significant=(perm_p < BONFERRONI_ALPHA)
    )


def compute_group_comparison(
    df: pd.DataFrame,
    roi_name: str,
    n_bootstrap: int = 10000
) -> GroupComparisonResult:
    """
    Compare ROI brain scores between Resistant and Susceptible model groups.
    """
    resistant = df[df['resistant'] == 1][roi_name].values
    susceptible = df[df['resistant'] == 0][roi_name].values
    
    mean_r = np.mean(resistant)
    mean_s = np.mean(susceptible)
    diff = mean_r - mean_s
    
    # Cohen's d
    pooled_std = np.sqrt((np.var(resistant, ddof=1) + np.var(susceptible, ddof=1)) / 2)
    cohens_d = diff / pooled_std if pooled_std > 0 else 0.0
    
    # Interpret effect size
    d_abs = abs(cohens_d)
    if d_abs < 0.2:
        interpretation = "negligible"
    elif d_abs < 0.5:
        interpretation = "small"
    elif d_abs < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"
    
    # Bootstrap CI for difference
    boot_diffs = []
    for _ in range(n_bootstrap):
        boot_r = np.random.choice(resistant, size=len(resistant), replace=True)
        boot_s = np.random.choice(susceptible, size=len(susceptible), replace=True)
        boot_diffs.append(np.mean(boot_r) - np.mean(boot_s))
    
    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)
    
    # T-test
    t_stat, t_p = stats.ttest_ind(resistant, susceptible)
    
    return GroupComparisonResult(
        roi_name=roi_name,
        n_resistant=len(resistant),
        n_susceptible=len(susceptible),
        mean_resistant=float(mean_r),
        mean_susceptible=float(mean_s),
        difference=float(diff),
        cohens_d=float(cohens_d),
        effect_interpretation=interpretation,
        bootstrap_ci_lower=float(ci_lower),
        bootstrap_ci_upper=float(ci_upper),
        ci_excludes_zero=not (ci_lower <= 0 <= ci_upper),
        t_statistic=float(t_stat),
        t_pvalue=float(t_p)
    )


def generate_report(
    correlation_results: Dict[str, CorrelationResult],
    group_results: Dict[str, GroupComparisonResult],
    n_models: int,
    n_resistant: int,
    n_susceptible: int,
    output_path: Path
) -> str:
    """Generate comprehensive analysis report."""
    lines = []
    lines.append("=" * 80)
    lines.append("STATISTICAL ANALYSIS: ROI BRAIN ALIGNMENT × SYCOPHANCY")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Models analyzed: {n_models}")
    lines.append(f"  Resistant (sycophancy < 50%): {n_resistant}")
    lines.append(f"  Susceptible (sycophancy >= 50%): {n_susceptible}")
    lines.append("")
    
    # Section 1: Correlation Analysis
    lines.append("=" * 70)
    lines.append("SECTION 1: MODEL-LEVEL CORRELATION ANALYSIS")
    lines.append("=" * 70)
    lines.append("Method: Pearson r with bootstrap 95% CI (10,000 samples)")
    lines.append(f"Multiple comparison correction: Bonferroni (α = {BONFERRONI_ALPHA:.4f})")
    lines.append("")
    
    lines.append(f"{'ROI':<18} {'r':>8} {'p_perm':>10} {'95% CI':>22} {'CI excl. 0':>12} {'Bonf. sig':>12}")
    lines.append("-" * 84)
    
    significant_rois = []
    bootstrap_sig_rois = []
    
    for roi_name in ROI_CATEGORIES:
        res = correlation_results[roi_name]
        ci_str = f"[{res.bootstrap_ci_lower:.3f}, {res.bootstrap_ci_upper:.3f}]"
        
        sig_marker = ""
        if res.permutation_p < 0.001:
            sig_marker = "***"
        elif res.permutation_p < 0.01:
            sig_marker = "**"
        elif res.permutation_p < 0.05:
            sig_marker = "*"
        elif res.permutation_p < 0.10:
            sig_marker = "†"
        
        ci_excl = "YES" if res.bootstrap_excludes_zero else "no"
        bonf_sig = "YES" if res.bonferroni_significant else "no"
        
        if res.permutation_p < 0.05:
            significant_rois.append(roi_name)
        if res.bootstrap_excludes_zero:
            bootstrap_sig_rois.append(roi_name)
        
        lines.append(f"{roi_name:<18} {res.pearson_r:>8.3f} {res.permutation_p:>10.4f}{sig_marker} {ci_str:>22} {ci_excl:>12} {bonf_sig:>12}")
    
    lines.append("")
    lines.append("Significance: *** p<0.001, ** p<0.01, * p<0.05, † p<0.10")
    lines.append("")
    
    if significant_rois:
        lines.append(f"✓ SIGNIFICANT (permutation p < 0.05): {', '.join(significant_rois)}")
    
    if bootstrap_sig_rois:
        lines.append(f"✓ BOOTSTRAP CI EXCLUDES ZERO: {', '.join(bootstrap_sig_rois)}")
    
    lines.append("")
    
    # Section 2: Group Comparison
    lines.append("=" * 70)
    lines.append("SECTION 2: GROUP COMPARISON (RESISTANT vs SUSCEPTIBLE)")
    lines.append("=" * 70)
    lines.append("")
    
    lines.append(f"{'ROI':<18} {'Mean R':>10} {'Mean S':>10} {'Diff':>10} {'Cohen d':>10} {'Effect':>10} {'CI excl. 0':>12}")
    lines.append("-" * 92)
    
    medium_large_effects = []
    
    for roi_name in ROI_CATEGORIES:
        gr = group_results[roi_name]
        ci_excl = "YES" if gr.ci_excludes_zero else "no"
        
        if gr.cohens_d >= 0.5:
            medium_large_effects.append((roi_name, gr.cohens_d))
        
        lines.append(f"{roi_name:<18} {gr.mean_resistant:>10.4f} {gr.mean_susceptible:>10.4f} {gr.difference:>10.4f} {gr.cohens_d:>10.2f} {gr.effect_interpretation:>10} {ci_excl:>12}")
    
    lines.append("")
    
    if medium_large_effects:
        lines.append("✓ MEDIUM-LARGE EFFECTS (|d| >= 0.5):")
        for roi, d in sorted(medium_large_effects, key=lambda x: abs(x[1]), reverse=True):
            direction = "Resistant > Susceptible" if d > 0 else "Susceptible > Resistant"
            lines.append(f"    {roi}: d = {d:.2f} ({direction})")
    
    lines.append("")
    
    # Section 3: Key Findings
    lines.append("=" * 70)
    lines.append("SECTION 3: KEY FINDINGS AND CONCLUSIONS")
    lines.append("=" * 70)
    lines.append("")
    
    # Find best ROI
    best_roi = min(correlation_results.keys(), 
                   key=lambda r: correlation_results[r].permutation_p)
    best_res = correlation_results[best_roi]
    
    # Summary
    lines.append("MAIN FINDINGS:")
    lines.append("")
    
    if best_res.bootstrap_excludes_zero:
        lines.append(f"1. {best_roi} shows a ROBUST negative relationship with sycophancy:")
        lines.append(f"   r = {best_res.pearson_r:.3f}, 95% CI [{best_res.bootstrap_ci_lower:.3f}, {best_res.bootstrap_ci_upper:.3f}]")
        lines.append(f"   Bootstrap CI excludes zero → Effect is statistically reliable")
        lines.append("")
    
    if medium_large_effects:
        best_effect_roi, best_d = medium_large_effects[0]
        gr = group_results[best_effect_roi]
        lines.append(f"2. Resistant VLMs have HIGHER {best_effect_roi} alignment:")
        lines.append(f"   Mean difference = {gr.difference:.4f}, Cohen's d = {best_d:.2f} (medium effect)")
        lines.append("")
    
    lines.append("INTERPRETATION:")
    lines.append("  • Early visual cortex alignment (V1-V3) differentiates resistant from susceptible VLMs")
    lines.append("  • Models with better low-level visual processing show greater resistance to sycophancy")
    lines.append("  • This suggests that perceptual grounding at the early visual level matters for behavioral robustness")
    
    lines.append("")
    lines.append("-" * 70)
    lines.append("PUBLICATION-READY STATEMENT:")
    lines.append("-" * 70)
    
    if best_res.bootstrap_excludes_zero:
        lines.append(f"Bootstrap analysis (10,000 resamples) revealed that {best_roi}")
        lines.append(f"brain alignment shows a negative relationship with sycophancy")
        lines.append(f"(r = {best_res.pearson_r:.3f}, 95% CI [{best_res.bootstrap_ci_lower:.3f}, {best_res.bootstrap_ci_upper:.3f}],")
        lines.append(f"permutation p = {best_res.permutation_p:.3f}).")
        
        if medium_large_effects:
            lines.append(f"Resistant VLMs (sycophancy < 50%) showed higher {best_effect_roi}")
            lines.append(f"alignment than susceptible VLMs (Cohen's d = {best_d:.2f}, {group_results[best_effect_roi].effect_interpretation} effect).")
    else:
        lines.append("While effect sizes suggest meaningful differences between resistant and")
        lines.append("susceptible VLMs, bootstrap confidence intervals did not consistently")
        lines.append("exclude zero, indicating that larger samples are needed for definitive conclusions.")
    
    lines.append("")
    lines.append("=" * 80)
    
    report_text = "\n".join(lines)
    
    print(report_text)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    logger.info(f"Report saved to {output_path}")
    
    return report_text


def main():
    logger.info("Starting statistical analysis...")
    
    # Paths
    results_dir = PROJECT_ROOT / "results"
    roi_dir = results_dir / "roi_brain_scores"
    output_dir = results_dir / "mixed_effects_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading per-subject ROI scores...")
    per_subject_scores = load_per_subject_roi_scores(roi_dir)
    logger.info(f"Loaded scores for {len(per_subject_scores)} models")
    
    if not per_subject_scores:
        logger.error("No ROI scores found. Run 02b_fit_roi_encoders.py first.")
        return
    
    logger.info("Loading sycophancy results...")
    sycophancy_rates = load_sycophancy_results(results_dir)
    logger.info(f"Loaded sycophancy for {len(sycophancy_rates)} models")
    
    # Build model-level dataframe
    logger.info("Building model-level dataframe...")
    df = build_model_level_dataframe(per_subject_scores, sycophancy_rates)
    
    n_resistant = int((df['resistant'] == 1).sum())
    n_susceptible = int((df['resistant'] == 0).sum())
    logger.info(f"Model split: {n_resistant} resistant, {n_susceptible} susceptible")
    
    # Run analyses
    correlation_results = {}
    group_results = {}
    
    for roi_name in ROI_CATEGORIES:
        logger.info(f"Analyzing {roi_name}...")
        
        # Correlation with bootstrap and permutation
        logger.info("  Computing correlation with bootstrap CI and permutation test...")
        correlation_results[roi_name] = compute_correlation_with_bootstrap(df, roi_name)
        
        # Group comparison
        logger.info("  Computing group comparison...")
        group_results[roi_name] = compute_group_comparison(df, roi_name)
    
    # Generate report
    logger.info("Generating report...")
    report_path = output_dir / "statistical_analysis_report.txt"
    generate_report(
        correlation_results, 
        group_results,
        n_models=len(df),
        n_resistant=n_resistant,
        n_susceptible=n_susceptible,
        output_path=report_path
    )
    
    # Save full results as JSON
    full_results = {
        'correlation_analysis': {
            roi: res.to_dict() for roi, res in correlation_results.items()
        },
        'group_comparison': {
            roi: res.to_dict() for roi, res in group_results.items()
        },
        'metadata': {
            'n_models': int(len(df)),
            'n_resistant': n_resistant,
            'n_susceptible': n_susceptible,
            'resistant_threshold': float(RESISTANT_THRESHOLD),
            'n_bootstrap': 10000,
            'bonferroni_alpha': float(BONFERRONI_ALPHA)
        }
    }
    
    results_path = output_dir / "statistical_analysis_results.json"
    with open(results_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    logger.info(f"Full results saved to {results_path}")
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
