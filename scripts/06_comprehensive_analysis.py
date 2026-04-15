"""
Script 06: Comprehensive Analysis for VLM Brain Alignment Research

Performs all 7 planned analyses:
1. Breakpoint Correlation - Difficulty level where model capitulates vs brain score
2. ROI-Specific Correlations - 32 brain regions × sycophancy
3. Category-Specific Analysis - 5 prompt categories × brain score
4. Architecture Family Comparison - Vision encoder families
5. Two-Turn Resistance Analysis - Pressure conversion metrics
6. Pressure Resistance Curve - AURC and steepness metrics
7. Persuasion Tactic Vulnerability - Which tactics break which models

Usage:
    python scripts/06_comprehensive_analysis.py
    python scripts/06_comprehensive_analysis.py --output-dir results/comprehensive_analysis
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import logging
from scipy import stats
from scipy.integrate import trapezoid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

AVAILABLE_MODELS = [
    "smolvlm_256m", "smolvlm_500m", "gemma3_1b", "qwen2vl_2b", "qwen25vl_3b",
    "paligemma2_10b", "phi35_vision", "llava_7b", "idefics2_8b", "blip2_opt27b",
    "lfm2vl_8b", "lfm25vl_1b"
]

# Vision encoder architecture families
ARCHITECTURE_FAMILIES = {
    "SigLIP": ["smolvlm_256m", "smolvlm_500m", "gemma3_1b", "paligemma2_10b"],
    "SigLIP2_NaFlex": ["lfm2vl_8b", "lfm25vl_1b"],
    "CLIP_ViT": ["llava_7b", "phi35_vision"],
    "Qwen_ViT": ["qwen2vl_2b", "qwen25vl_3b"],
    "SigLIP_SO400M": ["idefics2_8b"],
    "ViT_G14": ["blip2_opt27b"],
}

# Reverse mapping: model -> family
MODEL_TO_FAMILY = {}
for family, models in ARCHITECTURE_FAMILIES.items():
    for model in models:
        MODEL_TO_FAMILY[model] = family

# ROI categories in Algonauts dataset
ROI_CATEGORIES = {
    "prf-visualrois": ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"],
    "floc-bodies": ["EBA", "FBA-1", "FBA-2", "mTL-bodies"],
    "floc-faces": ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"],
    "floc-places": ["OPA", "PPA", "RSC"],
    "floc-words": ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"],
    "streams": ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"],
}

# Difficulty levels in V2 dataset
DIFFICULTY_LEVELS = [f"level_{i}" for i in range(1, 11)]

# Categories in V2 dataset
CATEGORIES = ["CATEGORY_1", "CATEGORY_2", "CATEGORY_3", "CATEGORY_4", "CATEGORY_5"]


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_brain_scores(brain_scores_dir: Path) -> Dict[str, Dict[str, float]]:
    """
    Load brain scores from JSON files.
    
    Returns:
        Dict mapping model_name -> {
            'overall_score': float,
            'normalized_score': float (if available),
            'per_subject': Dict[str, Dict[str, float]]
        }
    """
    scores = {}
    
    for json_file in brain_scores_dir.glob("*.json"):
        if json_file.name == "summary.json":
            continue
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        model_name = json_file.stem
        scores[model_name] = {
            'overall_score': data.get('overall_score', 0.0),
            'normalized_score': data.get('overall_normalized', data.get('normalized_score', None)),
            'per_subject': data.get('per_subject', {}),
        }
    
    logger.info(f"Loaded brain scores for {len(scores)} models")
    return scores


def load_sycophancy_results(sycophancy_dir: Path) -> Dict[str, Dict]:
    """
    Load sycophancy results from V2 evaluation.
    
    Returns:
        Dict mapping model_name -> full metrics dict including:
        - turn1_sycophancy_rate
        - final_sycophancy_rate
        - pressure_conversion_rate
        - by_difficulty
        - by_category
    """
    results = {}
    
    for model_dir in sycophancy_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        # Load metrics file
        metrics_path = model_dir / "metrics_v2.json"
        if not metrics_path.exists():
            metrics_path = model_dir / "metrics.json"
        
        if not metrics_path.exists():
            logger.warning(f"No metrics file found for {model_dir.name}")
            continue
        
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        
        results[model_dir.name] = data
        
        # Also load detailed results if available
        detailed_path = model_dir / "results_v2.json"
        if not detailed_path.exists():
            detailed_path = model_dir / "results.json"
        
        if detailed_path.exists():
            with open(detailed_path, 'r') as f:
                detailed = json.load(f)
            # Extract the 'results' list from the file structure
            # results_v2.json format: {"model_name": ..., "metrics": ..., "results": [...]}
            if isinstance(detailed, dict) and 'results' in detailed:
                results[model_dir.name]['_detailed_results'] = detailed['results']
            elif isinstance(detailed, list):
                results[model_dir.name]['_detailed_results'] = detailed
            else:
                results[model_dir.name]['_detailed_results'] = detailed
    
    logger.info(f"Loaded sycophancy results for {len(results)} models")
    return results


def load_roi_masks(data_dir: Path, subject_id: int = 1) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load ROI masks for a subject.
    
    Returns:
        Dict mapping hemisphere -> roi_category -> mask array
    """
    masks = {'lh': {}, 'rh': {}}
    roi_dir = data_dir / f"subj{subject_id:02d}" / "roi_masks"
    
    if not roi_dir.exists():
        logger.warning(f"ROI directory not found: {roi_dir}")
        return masks
    
    for hemi in ['lh', 'rh']:
        for roi_cat in ROI_CATEGORIES.keys():
            mask_file = roi_dir / f"{hemi}.{roi_cat}_challenge_space.npy"
            if mask_file.exists():
                masks[hemi][roi_cat] = np.load(mask_file)
            else:
                logger.debug(f"Mask not found: {mask_file}")
    
    return masks


def load_roi_mapping(data_dir: Path, subject_id: int = 1) -> Dict[str, Dict[int, str]]:
    """
    Load ROI value to name mapping.
    
    Returns:
        Dict mapping roi_category -> {value: name}
    """
    mappings = {}
    roi_dir = data_dir / f"subj{subject_id:02d}" / "roi_masks"
    
    for roi_cat in ROI_CATEGORIES.keys():
        mapping_file = roi_dir / f"mapping_{roi_cat}.npy"
        if mapping_file.exists():
            # Load the structured array
            mapping_arr = np.load(mapping_file, allow_pickle=True)
            mappings[roi_cat] = {}
            for item in mapping_arr:
                # Handle different possible formats
                if isinstance(item, (tuple, list, np.ndarray)):
                    if len(item) >= 2:
                        mappings[roi_cat][int(item[0])] = str(item[1])
                elif hasattr(item, 'dtype') and item.dtype.names:
                    for name in item.dtype.names:
                        if 'id' in name.lower() or 'value' in name.lower():
                            val = int(item[name])
                        elif 'name' in name.lower() or 'label' in name.lower():
                            label = str(item[name])
                    mappings[roi_cat][val] = label
    
    return mappings


# ============================================================================
# ANALYSIS 1: BREAKPOINT CORRELATION
# ============================================================================

def compute_breakpoints(
    sycophancy_results: Dict[str, Dict],
    threshold: float = 0.5
) -> Dict[str, int]:
    """
    Compute the "breakpoint" for each model - the difficulty level at which
    sycophancy rate first exceeds the threshold.
    
    Args:
        sycophancy_results: Dict of sycophancy metrics per model
        threshold: Sycophancy rate threshold (default 0.5 = 50%)
    
    Returns:
        Dict mapping model_name -> breakpoint (1-10, or 11 if never broke)
    """
    breakpoints = {}
    
    for model_name, metrics in sycophancy_results.items():
        by_difficulty = metrics.get('by_difficulty', {})
        
        if not by_difficulty:
            # If no difficulty breakdown, use overall rate
            final_rate = metrics.get('final_sycophancy_rate', 
                                     metrics.get('sycophancy_rate', 0))
            breakpoints[model_name] = 1 if final_rate > threshold else 11
            continue
        
        breakpoint = 11  # Default: never broke
        
        for level_num in range(1, 11):
            level_key = f"level_{level_num}"
            if level_key in by_difficulty:
                level_data = by_difficulty[level_key]
                rate = level_data.get('rate', level_data.get('sycophancy_rate', 0))
                
                if rate > threshold:
                    breakpoint = level_num
                    break
        
        breakpoints[model_name] = breakpoint
    
    return breakpoints


def analyze_breakpoint_correlation(
    brain_scores: Dict[str, Dict],
    breakpoints: Dict[str, int]
) -> Dict[str, Any]:
    """
    Analyze correlation between brain scores and breakpoints.
    
    Returns comprehensive analysis including:
    - Pearson and Spearman correlations
    - Per-model data
    - Statistical significance
    """
    # Match models
    matched_models = []
    brain_arr = []
    breakpoint_arr = []
    
    for model in brain_scores:
        if model in breakpoints:
            matched_models.append(model)
            # Use normalized score if available, else overall
            score = brain_scores[model].get('normalized_score')
            if score is None:
                score = brain_scores[model].get('overall_score', 0)
            brain_arr.append(score)
            breakpoint_arr.append(breakpoints[model])
    
    if len(matched_models) < 3:
        logger.warning(f"Insufficient models for breakpoint analysis: {len(matched_models)}")
        return {"error": "Insufficient models (need >= 3)"}
    
    brain_arr = np.array(brain_arr)
    breakpoint_arr = np.array(breakpoint_arr)
    
    # Compute correlations
    # Note: Positive correlation expected (higher brain score -> higher breakpoint -> more resistant)
    pearson_r, pearson_p = stats.pearsonr(brain_arr, breakpoint_arr)
    spearman_rho, spearman_p = stats.spearmanr(brain_arr, breakpoint_arr)
    
    # Per-model data
    per_model = []
    for i, model in enumerate(matched_models):
        per_model.append({
            'model': model,
            'brain_score': float(brain_arr[i]),
            'breakpoint': int(breakpoint_arr[i]),
            'architecture': MODEL_TO_FAMILY.get(model, 'unknown'),
        })
    
    # Sort by breakpoint (most resistant first)
    per_model = sorted(per_model, key=lambda x: -x['breakpoint'])
    
    return {
        'analysis': 'breakpoint_correlation',
        'threshold': 0.5,
        'n_models': len(matched_models),
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_rho': float(spearman_rho),
        'spearman_p': float(spearman_p),
        'brain_score_range': [float(brain_arr.min()), float(brain_arr.max())],
        'breakpoint_range': [int(breakpoint_arr.min()), int(breakpoint_arr.max())],
        'per_model': per_model,
        'interpretation': _interpret_correlation(pearson_r, pearson_p, 
                                                  "brain score and breakpoint (higher = more resistant)")
    }


# ============================================================================
# ANALYSIS 2: ROI-SPECIFIC CORRELATIONS
# ============================================================================

def compute_roi_brain_scores(
    features_dir: Path,
    data_dir: Path,
    brain_scores_dir: Path,
    subjects: List[int] = [1, 2, 3, 4, 5, 6, 7, 8]
) -> Dict[str, Dict[str, float]]:
    """
    Compute brain scores per ROI category.
    
    This is a computationally expensive operation that requires:
    1. Loading features for each model
    2. Loading fMRI data
    3. Loading ROI masks
    4. Computing Ridge regression per ROI
    
    For efficiency, we check if pre-computed ROI scores exist first.
    
    Returns:
        Dict mapping model_name -> roi_category -> score
    """
    # Check for pre-computed ROI scores from 02b_fit_roi_encoders.py
    # The script saves summary to results/roi_brain_scores/summary.json
    roi_summary_file = brain_scores_dir.parent / "roi_brain_scores" / "summary.json"
    
    if roi_summary_file.exists():
        logger.info(f"Loading ROI scores from {roi_summary_file}")
        with open(roi_summary_file, 'r') as f:
            return json.load(f)
    
    # Also check older path pattern
    roi_scores_file = brain_scores_dir / "roi_brain_scores.json"
    if roi_scores_file.exists():
        with open(roi_scores_file, 'r') as f:
            return json.load(f)
    
    # ROI scores not computed - return error status
    return {
        "_error": "ROI scores not computed",
        "_instructions": "Run: python scripts/02b_fit_roi_encoders.py --models all --subjects 1 2 3 4 5 6 7 8"
    }


def analyze_roi_correlations(
    roi_brain_scores: Dict[str, Dict[str, float]],
    sycophancy_results: Dict[str, Dict]
) -> Dict[str, Any]:
    """
    Analyze correlation between ROI-specific brain scores and sycophancy.
    
    Computes:
    1. ROI score vs overall sycophancy
    2. ROI score vs category-specific sycophancy (cross-correlation matrix)
    """
    if "_note" in roi_brain_scores or "_error" in roi_brain_scores:
        return {
            'analysis': 'roi_correlations',
            'status': 'requires_computation',
            'note': roi_brain_scores.get('_error', 'ROI-specific brain scores need to be computed separately.'),
            'instructions': roi_brain_scores.get('_instructions', 'Run: python scripts/02b_fit_roi_encoders.py --models all')
        }
    
    # Map ROIs to expected relevant categories
    # This is our hypothesis about which ROI should predict which category
    roi_category_mapping = {
        'floc-faces': 'CATEGORY_5',   # Face/person regions → authority/social attacks
        'floc-places': 'CATEGORY_3',  # Scene regions → presence denial
        'floc-bodies': 'CATEGORY_1',  # Body regions → object misidentification
        'prf-visualrois': 'CATEGORY_2',  # Low-level → color/attribute attacks
        'floc-words': 'CATEGORY_4',   # Word regions → count contradictions
    }
    
    # Get sycophancy rates per category
    model_syc_by_category = {}
    for model, metrics in sycophancy_results.items():
        by_category = metrics.get('by_category', {})
        model_syc_by_category[model] = {
            cat: data.get('rate', data.get('sycophancy_rate', 0))
            for cat, data in by_category.items()
        }
    
    # Get overall sycophancy rates
    overall_syc = {
        model: metrics.get('final_sycophancy_rate', metrics.get('sycophancy_rate', 0))
        for model, metrics in sycophancy_results.items()
    }
    
    # 1. ROI vs Overall Sycophancy
    roi_overall_results = {}
    for roi_cat in list(ROI_CATEGORIES.keys()):
        matched_models = []
        roi_scores = []
        syc_scores = []
        
        for model in roi_brain_scores:
            if model in overall_syc and roi_cat in roi_brain_scores[model]:
                matched_models.append(model)
                roi_scores.append(roi_brain_scores[model][roi_cat])
                syc_scores.append(overall_syc[model])
        
        if len(matched_models) >= 3:
            roi_scores = np.array(roi_scores)
            syc_scores = np.array(syc_scores)
            
            r, p = stats.pearsonr(roi_scores, syc_scores)
            
            roi_overall_results[roi_cat] = {
                'n_models': len(matched_models),
                'pearson_r': float(r),
                'pearson_p': float(p),
                'significant': p < 0.05,
            }
    
    # 2. ROI × Category Cross-Correlations
    # Compute correlation matrix: which ROI predicts which category?
    cross_correlations = {}
    categories = ['CATEGORY_1', 'CATEGORY_2', 'CATEGORY_3', 'CATEGORY_4', 'CATEGORY_5']
    
    for roi_cat in list(ROI_CATEGORIES.keys()):
        cross_correlations[roi_cat] = {}
        
        for cat in categories:
            matched_models = []
            roi_scores = []
            cat_syc_scores = []
            
            for model in roi_brain_scores:
                if (model in model_syc_by_category and 
                    roi_cat in roi_brain_scores[model] and
                    cat in model_syc_by_category[model]):
                    matched_models.append(model)
                    roi_scores.append(roi_brain_scores[model][roi_cat])
                    cat_syc_scores.append(model_syc_by_category[model][cat])
            
            if len(matched_models) >= 3:
                roi_scores = np.array(roi_scores)
                cat_syc_scores = np.array(cat_syc_scores)
                
                r, p = stats.pearsonr(roi_scores, cat_syc_scores)
                
                cross_correlations[roi_cat][cat] = {
                    'pearson_r': float(r),
                    'pearson_p': float(p),
                    'significant': p < 0.05,
                    'n_models': len(matched_models),
                }
    
    # 3. Test hypothesis: Does matched ROI-category pair show stronger correlation?
    hypothesis_tests = {}
    for roi, expected_cat in roi_category_mapping.items():
        if roi in cross_correlations and expected_cat in cross_correlations.get(roi, {}):
            matched_r = abs(cross_correlations[roi][expected_cat]['pearson_r'])
            
            # Compare to other categories for same ROI
            other_rs = [
                abs(cross_correlations[roi][cat]['pearson_r'])
                for cat in categories if cat != expected_cat and cat in cross_correlations.get(roi, {})
            ]
            
            if other_rs:
                mean_other_r = np.mean(other_rs)
                hypothesis_tests[roi] = {
                    'matched_category': expected_cat,
                    'matched_r': matched_r,
                    'other_categories_mean_r': mean_other_r,
                    'hypothesis_supported': matched_r > mean_other_r,
                }
    
    return {
        'analysis': 'roi_correlations',
        'n_roi_categories': len(roi_overall_results),
        'roi_vs_overall': roi_overall_results,
        'roi_category_cross_correlations': cross_correlations,
        'hypothesis_tests': hypothesis_tests,
        'most_predictive_roi': max(roi_overall_results.items(), 
                                    key=lambda x: abs(x[1]['pearson_r']))[0] if roi_overall_results else None,
    }


# ============================================================================
# ANALYSIS 3: CATEGORY-SPECIFIC SYCOPHANCY
# ============================================================================

def analyze_category_correlations(
    brain_scores: Dict[str, Dict],
    sycophancy_results: Dict[str, Dict]
) -> Dict[str, Any]:
    """
    Analyze correlation between brain scores and sycophancy per category.
    
    Categories in V2 dataset:
    - CATEGORY_1: Object misidentification
    - CATEGORY_2: Color/attribute attacks
    - CATEGORY_3: Presence denial
    - CATEGORY_4: Count contradictions
    - CATEGORY_5: Authority-based misidentification
    """
    category_results = {}
    
    for category in CATEGORIES:
        matched_models = []
        brain_arr = []
        syc_arr = []
        
        for model in brain_scores:
            if model not in sycophancy_results:
                continue
            
            by_category = sycophancy_results[model].get('by_category', {})
            if category not in by_category:
                continue
            
            matched_models.append(model)
            
            score = brain_scores[model].get('normalized_score')
            if score is None:
                score = brain_scores[model].get('overall_score', 0)
            brain_arr.append(score)
            
            cat_data = by_category[category]
            rate = cat_data.get('rate', cat_data.get('sycophancy_rate', 0))
            syc_arr.append(rate)
        
        if len(matched_models) >= 3:
            brain_arr = np.array(brain_arr)
            syc_arr = np.array(syc_arr)
            
            r, p = stats.pearsonr(brain_arr, syc_arr)
            rho, rho_p = stats.spearmanr(brain_arr, syc_arr)
            
            category_results[category] = {
                'n_models': len(matched_models),
                'mean_sycophancy': float(np.mean(syc_arr)),
                'std_sycophancy': float(np.std(syc_arr)),
                'pearson_r': float(r),
                'pearson_p': float(p),
                'spearman_rho': float(rho),
                'spearman_p': float(rho_p),
                'significant': p < 0.05,
            }
    
    # Category descriptions
    category_descriptions = {
        'CATEGORY_1': 'Object misidentification',
        'CATEGORY_2': 'Color/attribute attacks',
        'CATEGORY_3': 'Presence denial',
        'CATEGORY_4': 'Count contradictions',
        'CATEGORY_5': 'Authority-based misidentification',
    }
    
    # Find strongest correlations
    if category_results:
        strongest_visual = None
        strongest_social = None
        
        visual_cats = ['CATEGORY_1', 'CATEGORY_2', 'CATEGORY_3', 'CATEGORY_4']
        social_cats = ['CATEGORY_5']
        
        for cat in visual_cats:
            if cat in category_results:
                if strongest_visual is None or \
                   abs(category_results[cat]['pearson_r']) > abs(category_results[strongest_visual]['pearson_r']):
                    strongest_visual = cat
        
        for cat in social_cats:
            if cat in category_results:
                strongest_social = cat
    
    return {
        'analysis': 'category_correlations',
        'category_descriptions': category_descriptions,
        'n_categories': len(category_results),
        'results': category_results,
        'strongest_visual_correlation': strongest_visual,
        'strongest_social_correlation': strongest_social,
        'hypothesis_test': _test_visual_vs_social_hypothesis(category_results),
    }


def _test_visual_vs_social_hypothesis(category_results: Dict) -> Dict:
    """
    Test hypothesis: Brain alignment predicts resistance to visual attacks
    better than social attacks.
    """
    visual_cats = ['CATEGORY_1', 'CATEGORY_2', 'CATEGORY_3', 'CATEGORY_4']
    social_cats = ['CATEGORY_5']
    
    visual_rs = [abs(category_results[c]['pearson_r']) 
                 for c in visual_cats if c in category_results]
    social_rs = [abs(category_results[c]['pearson_r']) 
                 for c in social_cats if c in category_results]
    
    if not visual_rs or not social_rs:
        return {'status': 'insufficient_data'}
    
    mean_visual_r = np.mean(visual_rs)
    mean_social_r = np.mean(social_rs)
    
    return {
        'mean_visual_correlation': float(mean_visual_r),
        'mean_social_correlation': float(mean_social_r),
        'visual_stronger': mean_visual_r > mean_social_r,
        'interpretation': (
            "Brain alignment shows STRONGER correlation with visual-domain sycophancy"
            if mean_visual_r > mean_social_r else
            "Brain alignment shows STRONGER correlation with social-domain sycophancy"
        )
    }


# ============================================================================
# ANALYSIS 4: ARCHITECTURE FAMILY COMPARISON
# ============================================================================

def analyze_architecture_families(
    brain_scores: Dict[str, Dict],
    sycophancy_results: Dict[str, Dict]
) -> Dict[str, Any]:
    """
    Compare brain scores and sycophancy across vision encoder architecture families.
    """
    family_data = {}
    
    for family, models in ARCHITECTURE_FAMILIES.items():
        family_brain = []
        family_syc = []
        family_models = []
        
        for model in models:
            if model in brain_scores and model in sycophancy_results:
                score = brain_scores[model].get('normalized_score')
                if score is None:
                    score = brain_scores[model].get('overall_score', 0)
                
                final_syc = sycophancy_results[model].get('final_sycophancy_rate',
                            sycophancy_results[model].get('sycophancy_rate', 0))
                
                family_brain.append(score)
                family_syc.append(final_syc)
                family_models.append(model)
        
        if family_models:
            family_data[family] = {
                'models': family_models,
                'n_models': len(family_models),
                'mean_brain_score': float(np.mean(family_brain)),
                'std_brain_score': float(np.std(family_brain)) if len(family_brain) > 1 else 0.0,
                'mean_sycophancy': float(np.mean(family_syc)),
                'std_sycophancy': float(np.std(family_syc)) if len(family_syc) > 1 else 0.0,
            }
    
    # Rank families
    if family_data:
        # Best brain alignment
        best_brain = max(family_data.items(), 
                         key=lambda x: x[1]['mean_brain_score'])
        # Most resistant (lowest sycophancy)
        most_resistant = min(family_data.items(), 
                             key=lambda x: x[1]['mean_sycophancy'])
    else:
        best_brain = (None, {})
        most_resistant = (None, {})
    
    return {
        'analysis': 'architecture_families',
        'n_families': len(family_data),
        'families': family_data,
        'best_brain_alignment': best_brain[0],
        'best_brain_score': best_brain[1].get('mean_brain_score', 0) if best_brain[1] else 0,
        'most_resistant_family': most_resistant[0],
        'lowest_sycophancy': most_resistant[1].get('mean_sycophancy', 0) if most_resistant[1] else 0,
    }


# ============================================================================
# ANALYSIS 5: TWO-TURN RESISTANCE ANALYSIS
# ============================================================================

def analyze_two_turn_resistance(
    brain_scores: Dict[str, Dict],
    sycophancy_results: Dict[str, Dict]
) -> Dict[str, Any]:
    """
    Analyze two-turn attack resistance and its correlation with brain alignment.
    
    Metrics:
    - Turn 1 Resistance: 1 - Turn1_Sycophancy
    - Pressure Conversion Rate: (Final - Turn1) / (1 - Turn1)
    """
    model_data = []
    
    for model in brain_scores:
        if model not in sycophancy_results:
            continue
        
        metrics = sycophancy_results[model]
        
        turn1_rate = metrics.get('turn1_sycophancy_rate', 
                                 metrics.get('sycophancy_rate', 0))
        final_rate = metrics.get('final_sycophancy_rate', 
                                 metrics.get('sycophancy_rate', 0))
        
        # Compute derived metrics
        turn1_resistance = 1.0 - turn1_rate
        
        # Pressure conversion: how many that resisted Turn 1 capitulated in Turn 2
        if turn1_resistance > 0.01:  # Avoid division by near-zero
            pressure_conversion = (final_rate - turn1_rate) / turn1_resistance
        else:
            pressure_conversion = 0.0  # Already sycophantic
        
        score = brain_scores[model].get('normalized_score')
        if score is None:
            score = brain_scores[model].get('overall_score', 0)
        
        model_data.append({
            'model': model,
            'brain_score': score,
            'turn1_sycophancy': turn1_rate,
            'turn1_resistance': turn1_resistance,
            'final_sycophancy': final_rate,
            'pressure_conversion': pressure_conversion,
            'turn2_increase': final_rate - turn1_rate,
        })
    
    if len(model_data) < 3:
        return {'analysis': 'two_turn_resistance', 'error': 'Insufficient models'}
    
    # Compute correlations
    brain_arr = np.array([d['brain_score'] for d in model_data])
    t1_resist_arr = np.array([d['turn1_resistance'] for d in model_data])
    pressure_conv_arr = np.array([d['pressure_conversion'] for d in model_data])
    
    # Correlation: Brain score vs Turn 1 resistance
    r_t1, p_t1 = stats.pearsonr(brain_arr, t1_resist_arr)
    
    # Correlation: Brain score vs Pressure conversion (lower = more resistant)
    r_conv, p_conv = stats.pearsonr(brain_arr, pressure_conv_arr)
    
    # Aggregate stats
    mean_t1_resist = float(np.mean(t1_resist_arr))
    mean_pressure_conv = float(np.mean(pressure_conv_arr))
    
    return {
        'analysis': 'two_turn_resistance',
        'n_models': len(model_data),
        'turn1_resistance_correlation': {
            'pearson_r': float(r_t1),
            'pearson_p': float(p_t1),
            'interpretation': _interpret_correlation(r_t1, p_t1, 
                              "brain score and Turn 1 resistance")
        },
        'pressure_conversion_correlation': {
            'pearson_r': float(r_conv),
            'pearson_p': float(p_conv),
            'interpretation': _interpret_correlation(-r_conv, p_conv, 
                              "brain score and pressure resistance (negative = more resistant)")
        },
        'aggregate': {
            'mean_turn1_resistance': mean_t1_resist,
            'mean_pressure_conversion': mean_pressure_conv,
        },
        'per_model': sorted(model_data, key=lambda x: -x['turn1_resistance']),
    }


# ============================================================================
# ANALYSIS 6: PRESSURE RESISTANCE CURVE
# ============================================================================

def analyze_resistance_curves(
    brain_scores: Dict[str, Dict],
    sycophancy_results: Dict[str, Dict]
) -> Dict[str, Any]:
    """
    Analyze the pressure resistance curve for each model.
    
    Metrics:
    - AURC: Area Under Resistance Curve (higher = more resistant across all levels)
    - Curve steepness: How quickly does resistance drop?
    - Initial resistance: Performance at levels 1-3
    """
    model_curves = {}
    
    for model in brain_scores:
        if model not in sycophancy_results:
            continue
        
        by_difficulty = sycophancy_results[model].get('by_difficulty', {})
        if not by_difficulty:
            continue
        
        # Extract resistance (1 - sycophancy) at each level
        levels = []
        resistance = []
        
        for level_num in range(1, 11):
            level_key = f"level_{level_num}"
            if level_key in by_difficulty:
                levels.append(level_num)
                rate = by_difficulty[level_key].get('rate', 
                       by_difficulty[level_key].get('sycophancy_rate', 0))
                resistance.append(1.0 - rate)
        
        if len(levels) < 3:
            continue
        
        levels = np.array(levels)
        resistance = np.array(resistance)
        
        # Compute AURC (normalized to [0, 1] by dividing by max possible area)
        aurc = trapezoid(resistance, levels) / (levels[-1] - levels[0])
        
        # Compute curve steepness (slope of linear fit)
        slope, intercept = np.polyfit(levels, resistance, 1)
        
        # Initial resistance (mean of levels 1-3)
        initial_mask = levels <= 3
        initial_resistance = float(np.mean(resistance[initial_mask])) if initial_mask.any() else 0.0
        
        score = brain_scores[model].get('normalized_score')
        if score is None:
            score = brain_scores[model].get('overall_score', 0)
        
        model_curves[model] = {
            'brain_score': score,
            'levels': levels.tolist(),
            'resistance': resistance.tolist(),
            'aurc': float(aurc),
            'slope': float(slope),  # Negative = resistance decreases with difficulty
            'intercept': float(intercept),
            'initial_resistance': initial_resistance,
        }
    
    if len(model_curves) < 3:
        return {'analysis': 'resistance_curves', 'error': 'Insufficient models'}
    
    # Compute correlations with brain score
    brain_arr = np.array([d['brain_score'] for d in model_curves.values()])
    aurc_arr = np.array([d['aurc'] for d in model_curves.values()])
    slope_arr = np.array([d['slope'] for d in model_curves.values()])
    initial_arr = np.array([d['initial_resistance'] for d in model_curves.values()])
    
    r_aurc, p_aurc = stats.pearsonr(brain_arr, aurc_arr)
    r_slope, p_slope = stats.pearsonr(brain_arr, slope_arr)
    r_initial, p_initial = stats.pearsonr(brain_arr, initial_arr)
    
    return {
        'analysis': 'resistance_curves',
        'n_models': len(model_curves),
        'aurc_correlation': {
            'pearson_r': float(r_aurc),
            'pearson_p': float(p_aurc),
            'interpretation': _interpret_correlation(r_aurc, p_aurc, 
                              "brain score and AURC (higher = more resistant)")
        },
        'slope_correlation': {
            'pearson_r': float(r_slope),
            'pearson_p': float(p_slope),
            'interpretation': "Positive slope correlation means brain-aligned models maintain resistance longer"
        },
        'initial_resistance_correlation': {
            'pearson_r': float(r_initial),
            'pearson_p': float(p_initial),
        },
        'per_model': dict(sorted(model_curves.items(), 
                                  key=lambda x: -x[1]['aurc'])),
    }


# ============================================================================
# ANALYSIS 7: PERSUASION TACTIC VULNERABILITY
# ============================================================================

def analyze_tactic_vulnerability(
    sycophancy_results: Dict[str, Dict],
    gaslighting_prompts_path: Path
) -> Dict[str, Any]:
    """
    Analyze which persuasion tactics are most effective against each model.
    
    This requires parsing the detailed results to map prompts to their tactics.
    """
    # Load gaslighting prompts to get tactic mapping
    if not gaslighting_prompts_path.exists():
        logger.warning(f"Gaslighting prompts not found: {gaslighting_prompts_path}")
        return {
            'analysis': 'tactic_vulnerability',
            'status': 'requires_prompts_file',
            'note': f'Gaslighting prompts file not found at {gaslighting_prompts_path}'
        }
    
    with open(gaslighting_prompts_path, 'r') as f:
        prompts_data = json.load(f)
    
    # Build prompt_id -> tactics mapping
    prompt_tactics = {}
    for prompt in prompts_data.get('prompts', []):
        prompt_id = prompt.get('prompt_id', '')
        tactics = prompt.get('persuasion_tactics', [])
        prompt_tactics[prompt_id] = tactics
    
    # Count all unique tactics
    all_tactics = set()
    for tactics in prompt_tactics.values():
        all_tactics.update(tactics)
    
    logger.info(f"Found {len(all_tactics)} unique persuasion tactics")
    
    # Analyze per-model vulnerability to each tactic
    model_tactic_vulnerability = {}
    
    for model, metrics in sycophancy_results.items():
        detailed = metrics.get('_detailed_results', [])
        if not detailed:
            continue
        
        tactic_counts = {tactic: {'total': 0, 'sycophantic': 0} for tactic in all_tactics}
        
        for result in detailed:
            prompt_id = result.get('prompt_id', '')
            is_syc = result.get('final_is_sycophantic', result.get('is_sycophantic', False))
            
            tactics = prompt_tactics.get(prompt_id, [])
            for tactic in tactics:
                if tactic in tactic_counts:
                    tactic_counts[tactic]['total'] += 1
                    if is_syc:
                        tactic_counts[tactic]['sycophantic'] += 1
        
        # Compute vulnerability rates
        vulnerabilities = {}
        for tactic, counts in tactic_counts.items():
            if counts['total'] > 0:
                vulnerabilities[tactic] = counts['sycophantic'] / counts['total']
        
        if vulnerabilities:
            # Find most and least vulnerable tactics
            most_vulnerable = max(vulnerabilities.items(), key=lambda x: x[1])
            least_vulnerable = min(vulnerabilities.items(), key=lambda x: x[1])
            
            model_tactic_vulnerability[model] = {
                'vulnerabilities': vulnerabilities,
                'most_vulnerable_tactic': most_vulnerable[0],
                'most_vulnerable_rate': most_vulnerable[1],
                'least_vulnerable_tactic': least_vulnerable[0],
                'least_vulnerable_rate': least_vulnerable[1],
            }
    
    # Aggregate tactic effectiveness across all models
    tactic_effectiveness = {}
    for tactic in all_tactics:
        rates = []
        for model_data in model_tactic_vulnerability.values():
            if tactic in model_data['vulnerabilities']:
                rates.append(model_data['vulnerabilities'][tactic])
        
        if rates:
            tactic_effectiveness[tactic] = {
                'mean_sycophancy_rate': float(np.mean(rates)),
                'std_sycophancy_rate': float(np.std(rates)),
                'n_models': len(rates),
            }
    
    # Rank tactics by effectiveness
    ranked_tactics = sorted(tactic_effectiveness.items(),
                            key=lambda x: -x[1]['mean_sycophancy_rate'])
    
    return {
        'analysis': 'tactic_vulnerability',
        'n_tactics': len(all_tactics),
        'n_models_analyzed': len(model_tactic_vulnerability),
        'tactic_effectiveness': tactic_effectiveness,
        'ranked_tactics': [t[0] for t in ranked_tactics[:10]],  # Top 10 most effective
        'per_model': model_tactic_vulnerability,
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _interpret_correlation(r: float, p: float, description: str) -> str:
    """Generate human-readable interpretation of correlation."""
    if p >= 0.05:
        return f"No significant correlation between {description} (r={r:.3f}, p={p:.3f})"
    
    strength = "strong" if abs(r) > 0.5 else "moderate" if abs(r) > 0.3 else "weak"
    direction = "positive" if r > 0 else "negative"
    
    return f"Significant {strength} {direction} correlation between {description} (r={r:.3f}, p={p:.3f})"


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_all_analyses(
    brain_scores_dir: Path,
    sycophancy_dir: Path,
    data_dir: Path,
    features_dir: Path,
    gaslighting_prompts_path: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Run all 7 analyses and compile results.
    """
    results = {
        'timestamp': str(np.datetime64('now')),
        'analyses': {},
    }
    
    # Load data
    logger.info("Loading brain scores...")
    brain_scores = load_brain_scores(brain_scores_dir)
    
    logger.info("Loading sycophancy results...")
    sycophancy_results = load_sycophancy_results(sycophancy_dir)
    
    # Check we have data
    if not brain_scores:
        logger.error("No brain scores found")
        return {'error': 'No brain scores found'}
    
    if not sycophancy_results:
        logger.error("No sycophancy results found")
        return {'error': 'No sycophancy results found'}
    
    logger.info(f"Matched models: {set(brain_scores.keys()) & set(sycophancy_results.keys())}")
    
    # Analysis 1: Breakpoint Correlation
    logger.info("Analysis 1: Computing breakpoints...")
    breakpoints = compute_breakpoints(sycophancy_results)
    results['analyses']['breakpoint'] = analyze_breakpoint_correlation(brain_scores, breakpoints)
    
    # Analysis 2: ROI-Specific Correlations
    logger.info("Analysis 2: ROI-specific correlations...")
    roi_scores = compute_roi_brain_scores(features_dir, data_dir, brain_scores_dir)
    results['analyses']['roi'] = analyze_roi_correlations(roi_scores, sycophancy_results)
    
    # Analysis 3: Category-Specific
    logger.info("Analysis 3: Category-specific correlations...")
    results['analyses']['category'] = analyze_category_correlations(brain_scores, sycophancy_results)
    
    # Analysis 4: Architecture Families
    logger.info("Analysis 4: Architecture family comparison...")
    results['analyses']['architecture'] = analyze_architecture_families(brain_scores, sycophancy_results)
    
    # Analysis 5: Two-Turn Resistance
    logger.info("Analysis 5: Two-turn resistance analysis...")
    results['analyses']['two_turn'] = analyze_two_turn_resistance(brain_scores, sycophancy_results)
    
    # Analysis 6: Resistance Curves
    logger.info("Analysis 6: Resistance curve analysis...")
    results['analyses']['resistance_curves'] = analyze_resistance_curves(brain_scores, sycophancy_results)
    
    # Analysis 7: Tactic Vulnerability
    logger.info("Analysis 7: Persuasion tactic vulnerability...")
    results['analyses']['tactics'] = analyze_tactic_vulnerability(
        sycophancy_results, gaslighting_prompts_path
    )
    
    return results


def generate_summary_report(results: Dict[str, Any]) -> str:
    """Generate human-readable summary report."""
    lines = []
    lines.append("=" * 80)
    lines.append("COMPREHENSIVE VLM BRAIN ALIGNMENT ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    analyses = results.get('analyses', {})
    
    # Analysis 1: Breakpoint
    if 'breakpoint' in analyses:
        bp = analyses['breakpoint']
        lines.append("ANALYSIS 1: BREAKPOINT CORRELATION")
        lines.append("-" * 40)
        if 'error' not in bp:
            lines.append(f"  Models analyzed: {bp['n_models']}")
            lines.append(f"  Pearson r: {bp['pearson_r']:.4f} (p = {bp['pearson_p']:.4f})")
            lines.append(f"  Spearman rho: {bp['spearman_rho']:.4f} (p = {bp['spearman_p']:.4f})")
            lines.append(f"  Interpretation: {bp['interpretation']}")
            lines.append("")
            lines.append("  Per-model breakpoints (most resistant first):")
            for m in bp['per_model'][:5]:
                lines.append(f"    {m['model']}: level {m['breakpoint']}, brain={m['brain_score']:.4f}")
        else:
            lines.append(f"  Error: {bp['error']}")
        lines.append("")
    
    # Analysis 2: ROI
    if 'roi' in analyses:
        roi = analyses['roi']
        lines.append("ANALYSIS 2: ROI-SPECIFIC CORRELATIONS")
        lines.append("-" * 40)
        if roi.get('status') == 'requires_computation':
            lines.append(f"  Status: {roi.get('note', 'Requires computation')}")
            lines.append(f"  Instructions: {roi.get('instructions', 'Run 02b_fit_roi_encoders.py')}")
        else:
            # ROI vs Overall
            lines.append("  ROI vs Overall Sycophancy:")
            for roi_name, data in roi.get('roi_vs_overall', {}).items():
                sig = "*" if data.get('significant', False) else ""
                lines.append(f"    {roi_name}: r={data['pearson_r']:.3f}, p={data['pearson_p']:.3f} {sig}")
            
            lines.append(f"  Most predictive ROI: {roi.get('most_predictive_roi', 'N/A')}")
            
            # Cross-correlations
            if roi.get('roi_category_cross_correlations'):
                lines.append("")
                lines.append("  ROI × Category Cross-Correlations (Pearson r):")
                cross = roi['roi_category_cross_correlations']
                
                # Header
                lines.append("    " + " " * 18 + "CAT1   CAT2   CAT3   CAT4   CAT5")
                
                for roi_name in cross:
                    row = f"    {roi_name[:16]:<16}"
                    for cat in ['CATEGORY_1', 'CATEGORY_2', 'CATEGORY_3', 'CATEGORY_4', 'CATEGORY_5']:
                        if cat in cross[roi_name]:
                            r = cross[roi_name][cat]['pearson_r']
                            row += f" {r:>5.2f} "
                        else:
                            row += "   -   "
                    lines.append(row)
            
            # Hypothesis tests
            if roi.get('hypothesis_tests'):
                lines.append("")
                lines.append("  Hypothesis Tests (matched ROI → category):")
                for roi_name, test in roi['hypothesis_tests'].items():
                    supported = "✓ SUPPORTED" if test['hypothesis_supported'] else "✗ NOT SUPPORTED"
                    lines.append(f"    {roi_name} → {test['matched_category']}: r={test['matched_r']:.3f} vs others={test['other_categories_mean_r']:.3f} {supported}")
        lines.append("")
    
    # Analysis 3: Category
    if 'category' in analyses:
        cat = analyses['category']
        lines.append("ANALYSIS 3: CATEGORY-SPECIFIC CORRELATIONS")
        lines.append("-" * 40)
        for cat_name, data in cat.get('results', {}).items():
            sig = "*" if data.get('significant', False) else ""
            lines.append(f"  {cat_name}: r={data['pearson_r']:.3f}, p={data['pearson_p']:.3f} {sig}")
        
        hyp = cat.get('hypothesis_test', {})
        if 'mean_visual_correlation' in hyp:
            lines.append(f"  Visual vs Social: {hyp['interpretation']}")
        lines.append("")
    
    # Analysis 4: Architecture
    if 'architecture' in analyses:
        arch = analyses['architecture']
        lines.append("ANALYSIS 4: ARCHITECTURE FAMILY COMPARISON")
        lines.append("-" * 40)
        lines.append(f"  Best brain alignment: {arch.get('best_brain_alignment', 'N/A')}")
        lines.append(f"  Most resistant: {arch.get('most_resistant_family', 'N/A')}")
        lines.append("")
        for family, data in arch.get('families', {}).items():
            lines.append(f"  {family}:")
            lines.append(f"    Brain: {data['mean_brain_score']:.4f}, Syc: {data['mean_sycophancy']:.1%}")
        lines.append("")
    
    # Analysis 5: Two-Turn
    if 'two_turn' in analyses:
        tt = analyses['two_turn']
        lines.append("ANALYSIS 5: TWO-TURN RESISTANCE")
        lines.append("-" * 40)
        if 'error' not in tt:
            t1_corr = tt.get('turn1_resistance_correlation', {})
            lines.append(f"  Turn 1 Resistance ~ Brain Score: r={t1_corr.get('pearson_r', 0):.4f}")
            conv_corr = tt.get('pressure_conversion_correlation', {})
            lines.append(f"  Pressure Conversion ~ Brain Score: r={conv_corr.get('pearson_r', 0):.4f}")
            agg = tt.get('aggregate', {})
            lines.append(f"  Mean Turn 1 Resistance: {agg.get('mean_turn1_resistance', 0):.1%}")
            lines.append(f"  Mean Pressure Conversion: {agg.get('mean_pressure_conversion', 0):.1%}")
        lines.append("")
    
    # Analysis 6: Resistance Curves
    if 'resistance_curves' in analyses:
        rc = analyses['resistance_curves']
        lines.append("ANALYSIS 6: RESISTANCE CURVES (AURC)")
        lines.append("-" * 40)
        if 'error' not in rc:
            aurc_corr = rc.get('aurc_correlation', {})
            lines.append(f"  AURC ~ Brain Score: r={aurc_corr.get('pearson_r', 0):.4f} (p={aurc_corr.get('pearson_p', 1):.4f})")
            lines.append(f"  Interpretation: {aurc_corr.get('interpretation', 'N/A')}")
            lines.append("")
            lines.append("  Top 5 models by AURC:")
            for i, (model, data) in enumerate(list(rc.get('per_model', {}).items())[:5]):
                lines.append(f"    {model}: AURC={data['aurc']:.3f}")
        lines.append("")
    
    # Analysis 7: Tactics
    if 'tactics' in analyses:
        tac = analyses['tactics']
        lines.append("ANALYSIS 7: PERSUASION TACTIC VULNERABILITY")
        lines.append("-" * 40)
        if tac.get('status') == 'requires_prompts_file':
            lines.append(f"  Status: {tac['note']}")
        else:
            lines.append(f"  Tactics analyzed: {tac.get('n_tactics', 0)}")
            lines.append(f"  Models analyzed: {tac.get('n_models_analyzed', 0)}")
            lines.append("  Most effective tactics:")
            for tactic in tac.get('ranked_tactics', [])[:5]:
                eff = tac.get('tactic_effectiveness', {}).get(tactic, {})
                lines.append(f"    {tactic}: {eff.get('mean_sycophancy_rate', 0):.1%}")
        lines.append("")
    
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive VLM Brain Alignment Analysis")
    parser.add_argument("--brain-scores-dir", type=str, default="results/brain_scores",
                        help="Brain scores directory")
    parser.add_argument("--sycophancy-dir", type=str, default="results/sycophancy_v2",
                        help="Sycophancy V2 results directory")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Algonauts data directory")
    parser.add_argument("--features-dir", type=str, default="results/features",
                        help="Extracted features directory")
    parser.add_argument("--prompts-file", type=str, default="data/gaslighting_prompts_v2.json",
                        help="Gaslighting prompts V2 file")
    parser.add_argument("--output-dir", type=str, default="results/comprehensive_analysis",
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Resolve paths
    brain_scores_dir = PROJECT_ROOT / args.brain_scores_dir
    sycophancy_dir = PROJECT_ROOT / args.sycophancy_dir
    data_dir = PROJECT_ROOT / args.data_dir
    features_dir = PROJECT_ROOT / args.features_dir
    prompts_path = PROJECT_ROOT / args.prompts_file
    output_dir = PROJECT_ROOT / args.output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate directories
    if not brain_scores_dir.exists():
        logger.error(f"Brain scores directory not found: {brain_scores_dir}")
        sys.exit(1)
    
    if not sycophancy_dir.exists():
        logger.error(f"Sycophancy directory not found: {sycophancy_dir}")
        # Try fallback
        sycophancy_dir = PROJECT_ROOT / "results/sycophancy"
        if not sycophancy_dir.exists():
            logger.error("No sycophancy results found")
            sys.exit(1)
        logger.info(f"Using fallback sycophancy directory: {sycophancy_dir}")
    
    # Run all analyses
    logger.info("Starting comprehensive analysis...")
    results = run_all_analyses(
        brain_scores_dir=brain_scores_dir,
        sycophancy_dir=sycophancy_dir,
        data_dir=data_dir,
        features_dir=features_dir,
        gaslighting_prompts_path=prompts_path,
        output_dir=output_dir
    )
    
    # Save full results
    results_path = output_dir / "comprehensive_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved full results to {results_path}")
    
    # Generate and save summary report
    report = generate_summary_report(results)
    print("\n" + report)
    
    report_path = output_dir / "summary_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Saved summary report to {report_path}")
    
    logger.info(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
