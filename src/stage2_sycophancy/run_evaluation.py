"""
Sycophancy Evaluation Module

This module orchestrates the complete sycophancy evaluation pipeline:
1. Load gaslighting prompts
2. Run VLMs on prompts with structured output format
3. Parse responses using multi-layer parser
4. Compute metrics and save results
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from PIL import Image
from tqdm import tqdm
import logging

from .prompt_generator import GaslightingPrompt, GaslightingPromptGenerator
from .response_parser import (
    ResponseParser, ParsedResponse, Assessment, Confidence,
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result for a single prompt evaluation."""
    prompt_id: str
    image_id: str
    category: str
    gaslighting_prompt: str
    false_claim: str
    ground_truth: str
    raw_response: str
    assessment: str
    confidence: str
    is_sycophantic: Optional[bool]
    parse_method: str
    parse_layer: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompt_id": self.prompt_id,
            "image_id": self.image_id,
            "category": self.category,
            "gaslighting_prompt": self.gaslighting_prompt,
            "false_claim": self.false_claim,
            "ground_truth": self.ground_truth,
            "raw_response": self.raw_response,
            "assessment": self.assessment,
            "confidence": self.confidence,
            "is_sycophantic": self.is_sycophantic,
            "parse_method": self.parse_method,
            "parse_layer": self.parse_layer
        }


@dataclass
class SycophancyMetrics:
    """Aggregated metrics for sycophancy evaluation."""
    model_name: str
    total_prompts: int
    valid_responses: int
    unclear_responses: int
    sycophantic_count: int
    sycophancy_rate: float
    resistance_score: float
    parse_layer_distribution: Dict[int, int]
    by_category: Dict[str, Dict[str, float]]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "total_prompts": self.total_prompts,
            "valid_responses": self.valid_responses,
            "unclear_responses": self.unclear_responses,
            "sycophantic_count": self.sycophantic_count,
            "sycophancy_rate": self.sycophancy_rate,
            "resistance_score": self.resistance_score,
            "parse_layer_distribution": self.parse_layer_distribution,
            "by_category": self.by_category
        }


class SycophancyEvaluator:
    """
    Evaluates VLM sycophancy using gaslighting prompts.
    
    Runs the complete evaluation pipeline:
    1. Loads pre-generated gaslighting prompts
    2. Presents prompts to VLM with structured output requirement
    3. Parses responses using multi-layer parser
    4. Computes sycophancy metrics
    
    Example:
        >>> evaluator = SycophancyEvaluator(prompts_path="gaslighting_prompts.json")
        >>> metrics = evaluator.evaluate_model(vlm, "llava_7b")
        >>> print(f"Sycophancy rate: {metrics.sycophancy_rate:.2%}")
    """
    
    def __init__(
        self,
        prompts_path: Union[str, Path],
        output_dir: Union[str, Path] = "results/sycophancy"
    ):
        """
        Initialize the evaluator.
        
        Args:
            prompts_path: Path to gaslighting prompts JSON
            output_dir: Directory to save results
        """
        self.prompts_path = Path(prompts_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load prompts
        self.prompts, self.metadata = GaslightingPromptGenerator.load_prompts(
            str(self.prompts_path)
        )
        
        logger.info(f"Loaded {len(self.prompts)} prompts from {self.prompts_path}")
        
        # Initialize parser
        self.parser = ResponseParser()
    
    def evaluate_model(
        self,
        model,
        model_name: str,
        max_prompts: Optional[int] = None,
        show_progress: bool = True
    ) -> SycophancyMetrics:
        """
        Run sycophancy evaluation for a VLM.
        
        Args:
            model: VLM model object with generate() method
            model_name: Name identifier for the model
            max_prompts: Optional limit on number of prompts
            show_progress: Whether to show progress bar
            
        Returns:
            SycophancyMetrics with aggregated results
        """
        prompts_to_use = self.prompts
        if max_prompts:
            prompts_to_use = self.prompts[:max_prompts]
        
        logger.info(f"Evaluating {model_name} on {len(prompts_to_use)} prompts")
        
        results = []
        self.parser.reset_stats()
        
        iterator = prompts_to_use
        if show_progress:
            iterator = tqdm(prompts_to_use, desc=f"Evaluating {model_name}")
        
        for prompt in iterator:
            result = self._evaluate_single(model, prompt)
            results.append(result)
        
        # Compute metrics
        metrics = self._compute_metrics(model_name, results)
        
        # Save results
        self._save_results(model_name, metrics, results)
        
        logger.info(f"Evaluation complete for {model_name}")
        logger.info(f"  Sycophancy rate: {metrics.sycophancy_rate:.2%}")
        logger.info(f"  Parse stats: {self.parser.parse_stats}")
        
        return metrics
    
    def _evaluate_single(
        self,
        model,
        prompt: GaslightingPrompt
    ) -> EvaluationResult:
        """
        Evaluate a single prompt.
        
        Args:
            model: VLM model
            prompt: GaslightingPrompt to evaluate
            
        Returns:
            EvaluationResult for this prompt
        """
        # Load image
        try:
            image = Image.open(prompt.image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image {prompt.image_path}: {e}")
            return self._create_error_result(prompt, str(e))
        
        # Format prompt with template
        user_prompt = USER_PROMPT_TEMPLATE.format(
            gaslighting_prompt=prompt.gaslighting_prompt
        )
        
        # Generate response
        try:
            raw_response = model.generate(
                image=image,
                prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT,
                max_tokens=256
            )
        except Exception as e:
            logger.warning(f"Generation failed for {prompt.prompt_id}: {e}")
            return self._create_error_result(prompt, str(e))
        
        # Parse response
        parsed = self.parser.parse(raw_response)
        
        return EvaluationResult(
            prompt_id=prompt.prompt_id,
            image_id=prompt.image_id,
            category=prompt.category,
            gaslighting_prompt=prompt.gaslighting_prompt,
            false_claim=prompt.false_claim,
            ground_truth=prompt.ground_truth,
            raw_response=raw_response,
            assessment=parsed.assessment.value,
            confidence=parsed.confidence.value,
            is_sycophantic=parsed.is_sycophantic,
            parse_method=parsed.parse_method,
            parse_layer=parsed.parse_layer
        )
    
    def _create_error_result(
        self,
        prompt: GaslightingPrompt,
        error: str
    ) -> EvaluationResult:
        """Create an error result for failed evaluations."""
        return EvaluationResult(
            prompt_id=prompt.prompt_id,
            image_id=prompt.image_id,
            category=prompt.category,
            gaslighting_prompt=prompt.gaslighting_prompt,
            false_claim=prompt.false_claim,
            ground_truth=prompt.ground_truth,
            raw_response=f"ERROR: {error}",
            assessment="UNCLEAR",
            confidence="UNKNOWN",
            is_sycophantic=None,
            parse_method="error",
            parse_layer=0
        )
    
    def _compute_metrics(
        self,
        model_name: str,
        results: List[EvaluationResult]
    ) -> SycophancyMetrics:
        """
        Compute aggregated metrics from results.
        
        Args:
            model_name: Model identifier
            results: List of evaluation results
            
        Returns:
            SycophancyMetrics object
        """
        total = len(results)
        
        # Filter valid results (not UNCLEAR)
        valid_results = [r for r in results if r.is_sycophantic is not None]
        valid_count = len(valid_results)
        unclear_count = total - valid_count
        
        # Count sycophantic responses
        sycophantic_count = sum(r.is_sycophantic for r in valid_results)
        
        # Compute rates
        sycophancy_rate = sycophantic_count / valid_count if valid_count > 0 else 0.0
        resistance_score = 1.0 - sycophancy_rate
        
        # Parse layer distribution
        layer_dist = {}
        for r in results:
            layer = r.parse_layer
            layer_dist[layer] = layer_dist.get(layer, 0) + 1
        
        # Per-category breakdown
        categories = ["CATEGORY_1", "CATEGORY_2", "CATEGORY_3", "CATEGORY_4", "CATEGORY_5"]
        by_category = {}
        
        for cat in categories:
            cat_results = [r for r in valid_results if r.category == cat]
            if cat_results:
                cat_syc = sum(r.is_sycophantic for r in cat_results)
                by_category[cat] = {
                    "count": len(cat_results),
                    "sycophantic": cat_syc,
                    "rate": cat_syc / len(cat_results)
                }
        
        return SycophancyMetrics(
            model_name=model_name,
            total_prompts=total,
            valid_responses=valid_count,
            unclear_responses=unclear_count,
            sycophantic_count=sycophantic_count,
            sycophancy_rate=sycophancy_rate,
            resistance_score=resistance_score,
            parse_layer_distribution=layer_dist,
            by_category=by_category
        )
    
    def _save_results(
        self,
        model_name: str,
        metrics: SycophancyMetrics,
        results: List[EvaluationResult]
    ) -> Tuple[Path, Path]:
        """
        Save evaluation results and metrics.
        
        Args:
            model_name: Model identifier
            metrics: Computed metrics
            results: Individual results
            
        Returns:
            Tuple of (metrics_path, results_path)
        """
        model_dir = self.output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_path = model_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        
        # Save detailed results
        results_path = model_dir / "results.json"
        results_data = {
            "model_name": model_name,
            "metrics": metrics.to_dict(),
            "results": [r.to_dict() for r in results]
        }
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Saved metrics to {metrics_path}")
        logger.info(f"Saved results to {results_path}")
        
        return metrics_path, results_path
    
    @staticmethod
    def load_metrics(path: Union[str, Path]) -> SycophancyMetrics:
        """Load metrics from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        return SycophancyMetrics(
            model_name=data["model_name"],
            total_prompts=data["total_prompts"],
            valid_responses=data["valid_responses"],
            unclear_responses=data["unclear_responses"],
            sycophantic_count=data["sycophantic_count"],
            sycophancy_rate=data["sycophancy_rate"],
            resistance_score=data["resistance_score"],
            parse_layer_distribution=data["parse_layer_distribution"],
            by_category=data["by_category"]
        )
    
    def create_summary(
        self,
        model_names: List[str]
    ) -> Dict[str, Any]:
        """
        Create summary of all evaluated models.
        
        Args:
            model_names: List of model names to include
            
        Returns:
            Summary dictionary
        """
        summary = {
            "models": [],
            "best_resistance": None,
            "worst_resistance": None
        }
        
        best_resistance = -1
        worst_resistance = 2
        
        for model_name in model_names:
            metrics_path = self.output_dir / model_name / "metrics.json"
            if not metrics_path.exists():
                continue
            
            metrics = self.load_metrics(metrics_path)
            
            summary["models"].append({
                "model": model_name,
                "sycophancy_rate": metrics.sycophancy_rate,
                "resistance_score": metrics.resistance_score,
                "valid_responses": metrics.valid_responses,
                "unclear_responses": metrics.unclear_responses
            })
            
            if metrics.resistance_score > best_resistance:
                best_resistance = metrics.resistance_score
                summary["best_resistance"] = model_name
            
            if metrics.resistance_score < worst_resistance:
                worst_resistance = metrics.resistance_score
                summary["worst_resistance"] = model_name
        
        # Sort by resistance score (descending)
        summary["models"].sort(key=lambda x: x["resistance_score"], reverse=True)
        
        # Save summary
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also save as CSV
        import pandas as pd
        df = pd.DataFrame(summary["models"])
        csv_path = self.output_dir / "summary.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Saved summary to {summary_path}")
        
        return summary
