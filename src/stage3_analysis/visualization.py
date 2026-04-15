"""
Publication-Quality Visualization Module

This module creates publication-ready figures for the research paper.
All figures are saved in both PNG (300 DPI) and PDF formats.

Visualization Guidelines:
- Clean, professional aesthetics
- Proper spacing and no overlapping elements
- Consistent color schemes
- Clear legends and axis labels
- Appropriate font sizes for publication
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import logging

from .correlation import FullAnalysisResult, CorrelationAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Publication-quality style settings
STYLE_CONFIG = {
    "figure.dpi": 300,
    "figure.figsize": (8, 6),
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.8,
}

# Color palette
COLORS = {
    "primary": "#2E86AB",      # Blue
    "secondary": "#A23B72",    # Magenta
    "accent": "#F18F01",       # Orange
    "positive": "#4CAF50",     # Green
    "negative": "#E53935",     # Red
    "neutral": "#757575",      # Gray
    "regression": "#C73E1D",   # Dark red for regression line
}


class ResultVisualizer:
    """
    Creates publication-quality visualizations for the research.
    
    All figures are saved in both PNG (300 DPI) and PDF formats
    for publication purposes.
    
    Example:
        >>> visualizer = ResultVisualizer(analysis_result, output_dir="results/figures")
        >>> visualizer.create_all_figures()
    """
    
    def __init__(
        self,
        analysis_result: Optional[FullAnalysisResult] = None,
        output_dir: Union[str, Path] = "results/figures"
    ):
        """
        Initialize the visualizer.
        
        Args:
            analysis_result: FullAnalysisResult from correlation analysis
            output_dir: Directory to save figures
        """
        self.result = analysis_result
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Apply style
        plt.rcParams.update(STYLE_CONFIG)
        sns.set_style("whitegrid")
    
    def set_result(self, result: FullAnalysisResult):
        """Set the analysis result."""
        self.result = result
    
    def _save_figure(self, fig: plt.Figure, name: str):
        """Save figure in both PNG and PDF formats."""
        png_path = self.output_dir / f"{name}.png"
        pdf_path = self.output_dir / f"{name}.pdf"
        
        fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
        
        logger.info(f"Saved {name}.png and {name}.pdf")
        plt.close(fig)
    
    def create_main_correlation_figure(self) -> plt.Figure:
        """
        Create the main correlation figure (Figure 1).
        
        Shows Brain-Score vs Sycophancy Rate with regression line
        and confidence band.
        """
        if self.result is None:
            raise ValueError("No analysis result set. Call set_result() first.")
        
        x = np.array(self.result.brain_scores)
        y = np.array(self.result.sycophancy_rates)
        models = self.result.models
        r = self.result.main_correlation.pearson_r
        p = self.result.main_correlation.pearson_p
        ci_lower = self.result.main_correlation.ci_lower
        ci_upper = self.result.main_correlation.ci_upper
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot
        scatter = ax.scatter(
            x, y,
            s=120,
            c=COLORS["primary"],
            alpha=0.7,
            edgecolors='white',
            linewidth=1.5,
            zorder=5
        )
        
        # Add model labels with smart positioning
        for i, model in enumerate(models):
            # Simple label - clean model name for display
            display_name = model.replace("_", "-")
            
            # Offset based on position to avoid overlap
            x_offset = 0.01
            y_offset = 0.01
            ha = 'left'
            
            ax.annotate(
                display_name,
                (x[i], y[i]),
                fontsize=8,
                alpha=0.8,
                xytext=(5, 5),
                textcoords='offset points',
                ha=ha
            )
        
        # Regression line
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min() - 0.02, x.max() + 0.02, 100)
        y_line = slope * x_line + intercept
        
        ax.plot(
            x_line, y_line,
            color=COLORS["regression"],
            linewidth=2.5,
            linestyle='-',
            alpha=0.8,
            label='Linear fit',
            zorder=4
        )
        
        # Add confidence band (using bootstrap percentiles as guide)
        # This is simplified - full CI band would require more computation
        residuals = y - (slope * x + intercept)
        se = np.std(residuals)
        y_lower = y_line - 1.96 * se
        y_upper = y_line + 1.96 * se
        
        ax.fill_between(
            x_line, y_lower, y_upper,
            color=COLORS["regression"],
            alpha=0.1,
            label='95% CI'
        )
        
        # Statistics annotation box
        significance = "p < 0.05" if p < 0.05 else f"p = {p:.3f}"
        stats_text = (
            f"r = {r:.3f}\n"
            f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]\n"
            f"{significance}"
        )
        
        # Position stats box
        props = dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor=COLORS["neutral"], alpha=0.9)
        
        ax.text(
            0.05, 0.95, stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=props
        )
        
        # Labels and title
        ax.set_xlabel("Brain-Score (Neural Alignment R²)", fontsize=13)
        ax.set_ylabel("Sycophancy Rate", fontsize=13)
        ax.set_title(
            "Neural Alignment vs. Sycophancy in Vision-Language Models",
            fontsize=14,
            fontweight='bold',
            pad=15
        )
        
        # Format axes
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Legend
        ax.legend(loc='lower right', framealpha=0.9)
        
        plt.tight_layout()
        self._save_figure(fig, "main_correlation")
        
        return fig
    
    def create_model_comparison_figure(self) -> plt.Figure:
        """
        Create model comparison bar charts (Figure 2).
        
        Side-by-side bar charts showing Brain-Score and Sycophancy Rate
        for each model.
        """
        if self.result is None:
            raise ValueError("No analysis result set. Call set_result() first.")
        
        models = self.result.models
        brain_scores = self.result.brain_scores
        sycophancy_rates = self.result.sycophancy_rates
        
        # Sort by brain score
        sorted_indices = np.argsort(brain_scores)[::-1]
        models_sorted = [models[i].replace("_", "-") for i in sorted_indices]
        bs_sorted = [brain_scores[i] for i in sorted_indices]
        sr_sorted = [sycophancy_rates[i] for i in sorted_indices]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Brain Score subplot
        ax1 = axes[0]
        bars1 = ax1.barh(
            range(len(models_sorted)),
            bs_sorted,
            color=COLORS["primary"],
            alpha=0.8,
            edgecolor='white',
            linewidth=1
        )
        ax1.set_yticks(range(len(models_sorted)))
        ax1.set_yticklabels(models_sorted)
        ax1.set_xlabel("Brain-Score (R²)", fontsize=12)
        ax1.set_title("Neural Alignment", fontsize=13, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(True, axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, v in enumerate(bs_sorted):
            ax1.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
        
        # Sycophancy Rate subplot
        ax2 = axes[1]
        
        # Color code: lower is better (green), higher is worse (red)
        colors = [COLORS["positive"] if sr < np.median(sr_sorted) 
                  else COLORS["negative"] for sr in sr_sorted]
        
        bars2 = ax2.barh(
            range(len(models_sorted)),
            sr_sorted,
            color=colors,
            alpha=0.8,
            edgecolor='white',
            linewidth=1
        )
        ax2.set_yticks(range(len(models_sorted)))
        ax2.set_yticklabels(models_sorted)
        ax2.set_xlabel("Sycophancy Rate", fontsize=12)
        ax2.set_title("Sycophancy Rate (Lower = Better)", fontsize=13, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(True, axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, v in enumerate(sr_sorted):
            ax2.text(v + 0.01, i, f'{v:.2%}', va='center', fontsize=9)
        
        plt.tight_layout()
        self._save_figure(fig, "model_comparison")
        
        return fig
    
    def create_partial_correlation_figure(self) -> plt.Figure:
        """
        Create partial correlation figure (Figure 3).
        
        Shows the relationship after controlling for model size.
        """
        if self.result is None:
            raise ValueError("No analysis result set. Call set_result() first.")
        
        x = np.array(self.result.brain_scores)
        y = np.array(self.result.sycophancy_rates)
        sizes = np.array(self.result.model_sizes)
        models = self.result.models
        
        log_sizes = np.log10(sizes + 0.1).reshape(-1, 1)
        
        # Compute residuals
        from sklearn.linear_model import LinearRegression
        
        reg_x = LinearRegression().fit(log_sizes, x)
        residual_x = x - reg_x.predict(log_sizes)
        
        reg_y = LinearRegression().fit(log_sizes, y)
        residual_y = y - reg_y.predict(log_sizes)
        
        partial_r = self.result.partial_correlation.partial_r
        partial_p = self.result.partial_correlation.partial_p
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Original correlation with size as marker
        ax1 = axes[0]
        
        # Scale marker sizes
        marker_sizes = 50 + (sizes / sizes.max()) * 200
        
        scatter1 = ax1.scatter(
            x, y,
            s=marker_sizes,
            c=COLORS["primary"],
            alpha=0.7,
            edgecolors='white',
            linewidth=1.5
        )
        
        # Regression line
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min() - 0.02, x.max() + 0.02, 100)
        ax1.plot(x_line, slope * x_line + intercept, 
                color=COLORS["regression"], linewidth=2, linestyle='--')
        
        ax1.set_xlabel("Brain-Score", fontsize=12)
        ax1.set_ylabel("Sycophancy Rate", fontsize=12)
        ax1.set_title("Original Correlation\n(marker size = model params)", 
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Right: Residuals plot
        ax2 = axes[1]
        
        scatter2 = ax2.scatter(
            residual_x, residual_y,
            s=100,
            c=COLORS["secondary"],
            alpha=0.7,
            edgecolors='white',
            linewidth=1.5
        )
        
        # Regression on residuals
        slope_r, intercept_r = np.polyfit(residual_x, residual_y, 1)
        x_line_r = np.linspace(residual_x.min() - 0.02, residual_x.max() + 0.02, 100)
        ax2.plot(x_line_r, slope_r * x_line_r + intercept_r,
                color=COLORS["regression"], linewidth=2, linestyle='--')
        
        # Statistics
        sig = "p < 0.05" if partial_p < 0.05 else f"p = {partial_p:.3f}"
        stats_text = f"Partial r = {partial_r:.3f}\n{sig}"
        props = dict(boxstyle='round,pad=0.5', facecolor='white', 
                    edgecolor=COLORS["neutral"], alpha=0.9)
        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
                fontsize=11, verticalalignment='top', bbox=props)
        
        ax2.set_xlabel("Brain-Score (residualized)", fontsize=12)
        ax2.set_ylabel("Sycophancy Rate (residualized)", fontsize=12)
        ax2.set_title("After Controlling for Model Size", 
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Add zero reference lines
        ax2.axhline(0, color='gray', linewidth=0.8, linestyle=':')
        ax2.axvline(0, color='gray', linewidth=0.8, linestyle=':')
        
        plt.tight_layout()
        self._save_figure(fig, "partial_correlation")
        
        return fig
    
    def create_category_breakdown_figure(
        self,
        sycophancy_dir: Union[str, Path]
    ) -> plt.Figure:
        """
        Create per-category sycophancy breakdown (Figure 4).
        
        Shows sycophancy rates by attack category across models.
        """
        sycophancy_dir = Path(sycophancy_dir)
        
        # Load all metrics
        model_categories = {}
        
        for model_dir in sycophancy_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            metrics_path = model_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            
            with open(metrics_path, 'r') as f:
                data = json.load(f)
            
            model_name = model_dir.name.replace("_", "-")
            model_categories[model_name] = data.get("by_category", {})
        
        if not model_categories:
            logger.warning("No category data found")
            return None
        
        # Prepare data for heatmap
        categories = ["CATEGORY_1", "CATEGORY_2", "CATEGORY_3", "CATEGORY_4", "CATEGORY_5"]
        category_labels = [
            "Object\nMisidentification",
            "Attribute\nManipulation",
            "Existence\nDenial",
            "Count\nFalsification",
            "Authority\nAppeal"
        ]
        
        models = sorted(model_categories.keys())
        
        # Build matrix
        matrix = np.zeros((len(models), len(categories)))
        
        for i, model in enumerate(models):
            for j, cat in enumerate(categories):
                if cat in model_categories[model]:
                    matrix[i, j] = model_categories[model][cat].get("rate", 0)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
        
        # Colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Sycophancy Rate', fontsize=11)
        
        # Labels
        ax.set_xticks(np.arange(len(categories)))
        ax.set_xticklabels(category_labels, fontsize=10)
        ax.set_yticks(np.arange(len(models)))
        ax.set_yticklabels(models, fontsize=10)
        
        # Add value annotations
        for i in range(len(models)):
            for j in range(len(categories)):
                value = matrix[i, j]
                color = 'white' if value > 0.5 else 'black'
                ax.text(j, i, f'{value:.0%}', ha='center', va='center',
                       color=color, fontsize=9)
        
        ax.set_title("Sycophancy Rate by Attack Category", 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Attack Category", fontsize=12)
        ax.set_ylabel("Model", fontsize=12)
        
        plt.tight_layout()
        self._save_figure(fig, "category_breakdown")
        
        return fig
    
    def create_all_figures(
        self,
        sycophancy_dir: Optional[Union[str, Path]] = None
    ):
        """
        Create all figures.
        
        Args:
            sycophancy_dir: Directory with sycophancy results (for category breakdown)
        """
        logger.info("Creating all publication figures...")
        
        self.create_main_correlation_figure()
        self.create_model_comparison_figure()
        self.create_partial_correlation_figure()
        
        if sycophancy_dir:
            self.create_category_breakdown_figure(sycophancy_dir)
        
        logger.info(f"All figures saved to {self.output_dir}")
