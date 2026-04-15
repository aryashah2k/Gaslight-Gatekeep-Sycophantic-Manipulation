"""
Script 05b: Enhanced Analysis for V2 Sycophancy Results

Performs correlation analysis using enhanced V2 metrics including:
- Turn 1 vs Turn 2 sycophancy rates
- Difficulty level analysis
- Pressure conversion analysis
- Enhanced visualization

Usage:
    python scripts/05b_analyze_enhanced.py
    python scripts/05b_analyze_enhanced.py --sycophancy-dir results/sycophancy_v2
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import logging
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_brain_scores(brain_scores_dir: Path) -> Dict[str, float]:
    """Load brain scores."""
    scores = {}
    for json_file in brain_scores_dir.glob("*.json"):
        if json_file.name == "summary.json":
            continue
        with open(json_file, 'r') as f:
            data = json.load(f)
        model_name = json_file.stem
        scores[model_name] = data.get("normalized_score", data.get("overall_score", 0))
    return scores


def load_v2_sycophancy_metrics(sycophancy_dir: Path) -> Dict[str, Dict]:
    """Load V2 sycophancy metrics."""
    metrics = {}
    for model_dir in sycophancy_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        # Try V2 metrics first
        metrics_path = model_dir / "metrics_v2.json"
        if not metrics_path.exists():
            metrics_path = model_dir / "metrics.json"
        
        if not metrics_path.exists():
            continue
        
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        
        metrics[model_dir.name] = data
    
    return metrics


def analyze_difficulty_correlation(
    brain_scores: Dict[str, float],
    v2_metrics: Dict[str, Dict]
) -> Dict:
    """Analyze how difficulty level affects sycophancy by brain score."""
    results = {}
    
    # Collect difficulty data
    difficulty_data = {}  # {level: [(brain_score, syc_rate), ...]}
    
    for model_name, metrics in v2_metrics.items():
        if model_name not in brain_scores:
            continue
        
        brain_score = brain_scores[model_name]
        by_difficulty = metrics.get("by_difficulty", {})
        
        for level, level_data in by_difficulty.items():
            if level not in difficulty_data:
                difficulty_data[level] = []
            difficulty_data[level].append({
                "model": model_name,
                "brain_score": brain_score,
                "sycophancy_rate": level_data.get("rate", 0),
                "count": level_data.get("count", 0),
            })
    
    # Compute correlation for each difficulty level
    for level, data in sorted(difficulty_data.items()):
        if len(data) < 3:
            continue
        
        brain = np.array([d["brain_score"] for d in data])
        syc = np.array([d["sycophancy_rate"] for d in data])
        
        r, p = stats.pearsonr(brain, syc)
        
        results[level] = {
            "n_models": len(data),
            "correlation_r": float(r),
            "correlation_p": float(p),
            "mean_sycophancy": float(np.mean(syc)),
            "std_sycophancy": float(np.std(syc)),
        }
    
    return results


def analyze_two_turn_effectiveness(v2_metrics: Dict[str, Dict]) -> Dict:
    """Analyze how effective two-turn attacks are."""
    results = {}
    
    for model_name, metrics in v2_metrics.items():
        turn1_rate = metrics.get("turn1_sycophancy_rate", metrics.get("sycophancy_rate", 0))
        pressure_conv = metrics.get("pressure_conversion_rate", 0)
        final_rate = metrics.get("final_sycophancy_rate", metrics.get("sycophancy_rate", 0))
        
        results[model_name] = {
            "turn1_sycophancy": turn1_rate,
            "pressure_conversion": pressure_conv,
            "final_sycophancy": final_rate,
            "turn2_increase": final_rate - turn1_rate,
            "relative_increase": (final_rate / turn1_rate - 1) if turn1_rate > 0 else 0,
        }
    
    # Aggregate stats
    if results:
        all_t1 = [v["turn1_sycophancy"] for v in results.values()]
        all_conv = [v["pressure_conversion"] for v in results.values()]
        all_inc = [v["turn2_increase"] for v in results.values()]
        
        results["_aggregate"] = {
            "mean_turn1_sycophancy": float(np.mean(all_t1)),
            "mean_pressure_conversion": float(np.mean(all_conv)),
            "mean_turn2_increase": float(np.mean(all_inc)),
            "max_pressure_conversion": float(np.max(all_conv)) if all_conv else 0,
            "model_with_max_conversion": max(
                [(k, v["pressure_conversion"]) for k, v in results.items() if k != "_aggregate"],
                key=lambda x: x[1]
            )[0] if len(results) > 1 else None,
        }
    
    return results


def create_enhanced_correlation_report(
    brain_scores: Dict[str, float],
    v2_metrics: Dict[str, Dict],
    output_dir: Path
) -> Dict:
    """Create comprehensive correlation report."""
    # Get matched data
    matched_models = []
    brain_arr = []
    syc_t1_arr = []
    syc_final_arr = []
    
    for model in brain_scores:
        if model in v2_metrics:
            matched_models.append(model)
            brain_arr.append(brain_scores[model])
            syc_t1_arr.append(v2_metrics[model].get("turn1_sycophancy_rate", 
                              v2_metrics[model].get("sycophancy_rate", 0)))
            syc_final_arr.append(v2_metrics[model].get("final_sycophancy_rate", 
                                  v2_metrics[model].get("sycophancy_rate", 0)))
    
    if len(matched_models) < 3:
        logger.error(f"Need at least 3 matched models, got {len(matched_models)}")
        return {}
    
    brain_arr = np.array(brain_arr)
    syc_t1_arr = np.array(syc_t1_arr)
    syc_final_arr = np.array(syc_final_arr)
    
    # Correlations
    r_t1, p_t1 = stats.pearsonr(brain_arr, syc_t1_arr)
    r_final, p_final = stats.pearsonr(brain_arr, syc_final_arr)
    rho_t1, prho_t1 = stats.spearmanr(brain_arr, syc_t1_arr)
    rho_final, prho_final = stats.spearmanr(brain_arr, syc_final_arr)
    
    # Difficulty analysis
    difficulty_analysis = analyze_difficulty_correlation(brain_scores, v2_metrics)
    
    # Two-turn analysis
    two_turn_analysis = analyze_two_turn_effectiveness(v2_metrics)
    
    # Compile report
    report = {
        "summary": {
            "n_models": len(matched_models),
            "models": matched_models,
        },
        "turn1_correlation": {
            "pearson_r": float(r_t1),
            "pearson_p": float(p_t1),
            "spearman_rho": float(rho_t1),
            "spearman_p": float(prho_t1),
        },
        "final_correlation": {
            "pearson_r": float(r_final),
            "pearson_p": float(p_final),
            "spearman_rho": float(rho_final),
            "spearman_p": float(prho_final),
        },
        "by_difficulty": difficulty_analysis,
        "two_turn_analysis": two_turn_analysis,
        "per_model": [],
    }
    
    # Per-model details
    for i, model in enumerate(matched_models):
        report["per_model"].append({
            "model": model,
            "brain_score": float(brain_arr[i]),
            "turn1_sycophancy": float(syc_t1_arr[i]),
            "final_sycophancy": float(syc_final_arr[i]),
        })
    
    # Save report
    report_path = output_dir / "enhanced_analysis.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Saved enhanced analysis to {report_path}")
    
    return report


def generate_interpretation(report: Dict) -> str:
    """Generate human-readable interpretation."""
    lines = []
    lines.append("=" * 70)
    lines.append("ENHANCED SYCOPHANCY ANALYSIS REPORT")
    lines.append("=" * 70)
    
    n = report["summary"]["n_models"]
    lines.append(f"\nAnalyzed {n} models\n")
    
    # Turn 1 correlation
    t1 = report["turn1_correlation"]
    lines.append("TURN 1 (Single Attack) CORRELATION:")
    lines.append(f"  Pearson r = {t1['pearson_r']:.4f} (p = {t1['pearson_p']:.4f})")
    lines.append(f"  Spearman rho = {t1['spearman_rho']:.4f} (p = {t1['spearman_p']:.4f})")
    
    # Final correlation
    final = report["final_correlation"]
    lines.append("\nFINAL (After Two-Turn) CORRELATION:")
    lines.append(f"  Pearson r = {final['pearson_r']:.4f} (p = {final['pearson_p']:.4f})")
    lines.append(f"  Spearman rho = {final['spearman_rho']:.4f} (p = {final['spearman_p']:.4f})")
    
    # Interpretation
    r = final['pearson_r']
    p = final['pearson_p']
    
    lines.append("\nINTERPRETATION:")
    if r < -0.5 and p < 0.05:
        lines.append("  * STRONG SUPPORT for hypothesis:")
        lines.append("    Higher neural alignment strongly correlates with lower sycophancy")
    elif r < -0.3 and p < 0.1:
        lines.append("  * MODERATE SUPPORT for hypothesis:")
        lines.append("    Higher neural alignment correlates with lower sycophancy")
    elif r > 0.3 and p < 0.1:
        lines.append("  X CONTRADICTS hypothesis:")
        lines.append("    Higher neural alignment correlates with HIGHER sycophancy")
    else:
        lines.append("  o INCONCLUSIVE:")
        lines.append("    No significant correlation detected")
    
    # Two-turn effectiveness
    if "_aggregate" in report.get("two_turn_analysis", {}):
        agg = report["two_turn_analysis"]["_aggregate"]
        lines.append("\nTWO-TURN ATTACK EFFECTIVENESS:")
        lines.append(f"  Mean pressure conversion: {agg['mean_pressure_conversion']:.1%}")
        lines.append(f"  Mean Turn 2 increase: +{agg['mean_turn2_increase']:.1%}")
        if agg.get('model_with_max_conversion'):
            lines.append(f"  Max conversion: {agg['max_pressure_conversion']:.1%} ({agg['model_with_max_conversion']})")
    
    # Difficulty trends
    if report.get("by_difficulty"):
        lines.append("\nDIFFICULTY LEVEL ANALYSIS:")
        for level, data in sorted(report["by_difficulty"].items()):
            lines.append(f"  {level}: mean syc = {data['mean_sycophancy']:.1%}, "
                        f"r = {data['correlation_r']:.3f}")
    
    # Per-model table
    lines.append("\nPER-MODEL RESULTS:")
    lines.append(f"{'Model':<20} {'Brain Score':>12} {'Turn1 Syc':>12} {'Final Syc':>12}")
    lines.append("-" * 56)
    
    for m in sorted(report["per_model"], key=lambda x: x["final_sycophancy"]):
        lines.append(f"{m['model']:<20} {m['brain_score']:>11.4f} "
                    f"{m['turn1_sycophancy']:>11.1%} {m['final_sycophancy']:>11.1%}")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Enhanced Analysis for V2 Results")
    parser.add_argument("--brain-scores-dir", type=str, default="results/brain_scores",
                        help="Brain scores directory")
    parser.add_argument("--sycophancy-dir", type=str, default="results/sycophancy_v2",
                        help="Sycophancy V2 results directory")
    parser.add_argument("--output-dir", type=str, default="results/analysis_v2",
                        help="Output directory")
    
    args = parser.parse_args()
    
    brain_scores_dir = PROJECT_ROOT / args.brain_scores_dir
    sycophancy_dir = PROJECT_ROOT / args.sycophancy_dir
    output_dir = PROJECT_ROOT / args.output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check directories
    if not brain_scores_dir.exists():
        logger.error(f"Brain scores not found: {brain_scores_dir}")
        logger.error("Run 02_fit_encoders.py first")
        sys.exit(1)
    
    if not sycophancy_dir.exists():
        logger.error(f"Sycophancy V2 results not found: {sycophancy_dir}")
        logger.info("Trying V1 directory...")
        sycophancy_dir = PROJECT_ROOT / "results/sycophancy"
        if not sycophancy_dir.exists():
            logger.error("No sycophancy results found")
            logger.error("Run 04b_run_enhanced_sycophancy.py first")
            sys.exit(1)
    
    # Load data
    logger.info("Loading brain scores...")
    brain_scores = load_brain_scores(brain_scores_dir)
    logger.info(f"Loaded {len(brain_scores)} brain scores")
    
    logger.info("Loading sycophancy metrics...")
    v2_metrics = load_v2_sycophancy_metrics(sycophancy_dir)
    logger.info(f"Loaded {len(v2_metrics)} sycophancy results")
    
    # Run analysis
    logger.info("Running enhanced correlation analysis...")
    report = create_enhanced_correlation_report(brain_scores, v2_metrics, output_dir)
    
    if not report:
        logger.error("Analysis failed - need at least 3 matched models")
        sys.exit(1)
    
    # Generate and print interpretation
    interpretation = generate_interpretation(report)
    print("\n" + interpretation)
    
    # Save interpretation
    interp_path = output_dir / "interpretation.txt"
    with open(interp_path, 'w') as f:
        f.write(interpretation)
    
    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
