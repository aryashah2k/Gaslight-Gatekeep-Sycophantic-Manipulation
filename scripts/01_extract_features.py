"""
Script 01: Feature Extraction with Checkpointing

Extracts frozen vision encoder features from all VLMs for all subjects.
Features are saved to results/features/ directory.

FEATURES:
- Periodic checkpointing (saves progress every N batches)
- Resume capability (detects existing checkpoints and continues)
- Memory management (clears CUDA cache periodically)
- Graceful shutdown handling (saves on interrupt)
- Skip already completed subject/model combinations

Usage:
    python scripts/01_extract_features.py --models all --subjects 1 2 3 4
    python scripts/01_extract_features.py --models smolvlm_256m --subjects 1 --resume
    python scripts/01_extract_features.py --models llava_7b --batch-size 8 --checkpoint-every 50
"""

import os
import sys
import gc
import signal
import argparse
from pathlib import Path
from typing import List, Optional
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Available models
AVAILABLE_MODELS = [
    "smolvlm_256m",
    "smolvlm_500m",
    "qwen2vl_2b",
    "qwen25vl_3b",
    "gemma3_1b",
    "llava_7b",
    "paligemma2_10b",  # Replaces moondream2/internvl2_2b
    "blip2_opt27b",    # Replaces minicpm_v26
    "lfm2vl_8b",
    "lfm25vl_1b",
    "idefics2_8b",
    "phi35_vision",
]


class CheckpointManager:
    """Manages checkpoints for feature extraction with resume capability."""
    
    def __init__(self, output_dir: Path, model_name: str, subject_id: int):
        self.output_dir = output_dir / model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.subject_id = subject_id
        
        self.checkpoint_file = self.output_dir / f"checkpoint_subj{subject_id:02d}.json"
        self.features_file = self.output_dir / f"{model_name}_subj{subject_id:02d}_features.npz"
        self.partial_file = self.output_dir / f"{model_name}_subj{subject_id:02d}_partial.npz"
    
    def is_complete(self) -> bool:
        """Check if extraction is already complete for this subject."""
        return self.features_file.exists()
    
    def get_resume_index(self) -> int:
        """Get the index to resume from (0 if no checkpoint exists)."""
        if not self.checkpoint_file.exists():
            return 0
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
            return data.get('last_completed_index', 0)
        except Exception as e:
            logger.warning(f"Could not read checkpoint: {e}")
            return 0
    
    def load_partial_features(self) -> Optional[np.ndarray]:
        """Load partially extracted features if they exist."""
        if not self.partial_file.exists():
            return None
        
        try:
            data = np.load(self.partial_file)
            return data['features']
        except Exception as e:
            logger.warning(f"Could not load partial features: {e}")
            return None
    
    def save_checkpoint(self, features: np.ndarray, last_index: int, total: int):
        """Save checkpoint with current progress."""
        # Save partial features
        np.savez_compressed(self.partial_file, features=features)
        
        # Save metadata
        checkpoint_data = {
            'last_completed_index': last_index,
            'total_images': total,
            'model_name': self.model_name,
            'subject_id': self.subject_id,
            'feature_shape': list(features.shape)
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"Checkpoint saved: {last_index}/{total} images")
    
    def finalize(self, features: np.ndarray, feature_dim: int, n_images: int):
        """Save final features and clean up checkpoints."""
        # Save final features
        np.savez_compressed(
            self.features_file,
            features=features,
            model_name=self.model_name,
            feature_dim=feature_dim,
            n_images=n_images,
            pooling_method="model_specific"
        )
        
        # Clean up checkpoint files
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        if self.partial_file.exists():
            self.partial_file.unlink()
        
        logger.info(f"Saved final features to {self.features_file}")
    
    def cleanup_partial(self):
        """Remove partial files without saving final."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        if self.partial_file.exists():
            self.partial_file.unlink()


def clear_memory():
    """Clear GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_vlm(model_name: str):
    """Dynamically load a VLM wrapper."""
    module_name = f"src.vlm_models.vlm_{model_name}"
    
    import importlib
    module = importlib.import_module(module_name)
    
    return module.load_model(device="cuda" if torch.cuda.is_available() else "cpu")


def extract_features_with_checkpointing(
    vlm,
    image_paths: List[Path],
    checkpoint_mgr: CheckpointManager,
    batch_size: int = 8,
    checkpoint_every: int = 100,
    memory_clear_every: int = 50
) -> np.ndarray:
    """
    Extract features with checkpointing and resume capability.
    
    Args:
        vlm: Loaded VLM model
        image_paths: List of image paths
        checkpoint_mgr: Checkpoint manager for saving progress
        batch_size: Batch size for processing
        checkpoint_every: Save checkpoint every N batches
        memory_clear_every: Clear memory every N batches
    
    Returns:
        Extracted features array
    """
    total_images = len(image_paths)
    
    # Check for resume
    resume_index = checkpoint_mgr.get_resume_index()
    existing_features = checkpoint_mgr.load_partial_features()
    
    if resume_index > 0 and existing_features is not None:
        logger.info(f"Resuming from image {resume_index}/{total_images}")
        all_features = [existing_features]
    else:
        resume_index = 0
        all_features = []
    
    # Setup interrupt handler
    interrupted = [False]
    
    def signal_handler(signum, frame):
        logger.warning("Interrupt received! Saving checkpoint...")
        interrupted[0] = True
    
    # Register signal handlers
    original_sigint = signal.signal(signal.SIGINT, signal_handler)
    original_sigterm = signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Process remaining images
        remaining_paths = image_paths[resume_index:]
        n_batches = (len(remaining_paths) + batch_size - 1) // batch_size
        
        batch_count = 0
        processed_since_checkpoint = 0
        
        pbar = tqdm(
            range(0, len(remaining_paths), batch_size),
            desc=f"Extracting (from {resume_index})",
            total=n_batches
        )
        
        for batch_start in pbar:
            if interrupted[0]:
                break
            
            batch_end = min(batch_start + batch_size, len(remaining_paths))
            batch_paths = remaining_paths[batch_start:batch_end]
            
            try:
                # Load and process batch
                images = [Image.open(p).convert("RGB") for p in batch_paths]
                
                with torch.no_grad():
                    features = vlm.get_vision_features(images)
                
                # Ensure features are numpy array
                if isinstance(features, torch.Tensor):
                    features = features.cpu().float().numpy()
                
                all_features.append(features)
                
                # Cleanup images
                for img in images:
                    img.close()
                del images
                
            except Exception as e:
                logger.error(f"Error processing batch at index {resume_index + batch_start}: {e}")
                # Save checkpoint before raising
                if all_features:
                    combined = np.concatenate(all_features, axis=0)
                    checkpoint_mgr.save_checkpoint(
                        combined,
                        resume_index + batch_start,
                        total_images
                    )
                raise
            
            batch_count += 1
            processed_since_checkpoint += len(batch_paths)
            
            # Update progress bar
            current_total = resume_index + batch_start + len(batch_paths)
            pbar.set_postfix({
                'done': f"{current_total}/{total_images}",
                'batch': batch_count
            })
            
            # Periodic checkpoint
            if batch_count % checkpoint_every == 0:
                combined = np.concatenate(all_features, axis=0)
                checkpoint_mgr.save_checkpoint(
                    combined,
                    resume_index + batch_start + len(batch_paths),
                    total_images
                )
            
            # Periodic memory clear
            if batch_count % memory_clear_every == 0:
                clear_memory()
        
        pbar.close()
        
    finally:
        # Restore signal handlers
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
    
    # Combine all features
    if not all_features:
        raise RuntimeError("No features extracted!")
    
    combined_features = np.concatenate(all_features, axis=0)
    
    # L2 normalize
    norms = np.linalg.norm(combined_features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    combined_features = combined_features / norms
    
    # If interrupted, save checkpoint; otherwise finalize
    if interrupted[0]:
        checkpoint_mgr.save_checkpoint(combined_features, len(combined_features), total_images)
        logger.warning(f"Saved partial progress. Rerun with --resume to continue.")
        raise KeyboardInterrupt("Extraction interrupted by user")
    
    return combined_features


def extract_features_for_model(
    model_name: str,
    subjects: List[int],
    data_dir: Path,
    output_dir: Path,
    batch_size: int = 8,
    checkpoint_every: int = 100,
    resume: bool = True,
    skip_existing: bool = True
):
    """Extract features for a single model across subjects with checkpointing."""
    
    logger.info(f"=" * 60)
    logger.info(f"Extracting features for {model_name}")
    logger.info(f"=" * 60)
    
    # Load data loader
    from src.utils.data_loader import AlgonautsDataLoader
    loader = AlgonautsDataLoader(data_dir)
    
    # Check which subjects need processing
    subjects_to_process = []
    for subj_id in subjects:
        checkpoint_mgr = CheckpointManager(output_dir, model_name, subj_id)
        
        if skip_existing and checkpoint_mgr.is_complete():
            logger.info(f"  Subject {subj_id}: Already complete, skipping")
            continue
        
        subjects_to_process.append(subj_id)
    
    if not subjects_to_process:
        logger.info(f"All subjects already complete for {model_name}")
        return
    
    # Load model only if we have work to do
    logger.info(f"Loading model {model_name}...")
    vlm = load_vlm(model_name)
    
    try:
        for subj_id in subjects_to_process:
            logger.info(f"Processing subject {subj_id}")
            
            checkpoint_mgr = CheckpointManager(output_dir, model_name, subj_id)
            
            # Get image paths
            subj_data = loader.load_subject(subj_id, load_images=False)
            image_paths = subj_data.image_paths
            
            logger.info(f"  Found {len(image_paths)} images")
            
            try:
                # Extract features with checkpointing
                features = extract_features_with_checkpointing(
                    vlm=vlm,
                    image_paths=image_paths,
                    checkpoint_mgr=checkpoint_mgr,
                    batch_size=batch_size,
                    checkpoint_every=checkpoint_every
                )
                
                # Validate feature count
                if features.shape[0] != len(image_paths):
                    raise ValueError(
                        f"Feature count mismatch: got {features.shape[0]}, "
                        f"expected {len(image_paths)}"
                    )
                
                # Finalize
                checkpoint_mgr.finalize(
                    features=features,
                    feature_dim=features.shape[1],
                    n_images=len(image_paths)
                )
                
                logger.info(f"  Subject {subj_id} complete: {features.shape}")
                
            except KeyboardInterrupt:
                logger.warning("Interrupted! Progress saved. Rerun to continue.")
                raise
            
            # Clear memory between subjects
            clear_memory()
    
    finally:
        # Unload model
        vlm.unload()
        clear_memory()
    
    logger.info(f"Completed {model_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract VLM features with checkpointing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all models for all subjects
  python scripts/01_extract_features.py --models all
  
  # Extract single model with small batch size
  python scripts/01_extract_features.py --models smolvlm_256m --batch-size 4
  
  # Resume interrupted extraction
  python scripts/01_extract_features.py --models smolvlm_256m --resume
  
  # Force re-extraction (ignore existing)
  python scripts/01_extract_features.py --models smolvlm_256m --no-skip-existing
"""
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Models to process (default: all)"
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Subject IDs to process"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/features",
        help="Output directory for features"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for feature extraction (default: 8, lower if OOM)"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Save checkpoint every N batches (default: 50)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoints (default: True)"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Don't skip already completed subject/model combinations"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    data_dir = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine models
    if "all" in args.models:
        models = AVAILABLE_MODELS
    else:
        models = [m for m in args.models if m in AVAILABLE_MODELS]
        if len(models) != len(args.models):
            invalid = set(args.models) - set(AVAILABLE_MODELS)
            logger.warning(f"Unknown models (skipping): {invalid}")
    
    logger.info(f"Processing {len(models)} models: {models}")
    logger.info(f"Subjects: {args.subjects}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Checkpoint every: {args.checkpoint_every} batches")
    logger.info(f"Skip existing: {not args.no_skip_existing}")
    
    # Process each model
    for model_name in models:
        try:
            extract_features_for_model(
                model_name=model_name,
                subjects=args.subjects,
                data_dir=data_dir,
                output_dir=output_dir,
                batch_size=args.batch_size,
                checkpoint_every=args.checkpoint_every,
                resume=args.resume,
                skip_existing=not args.no_skip_existing
            )
        except KeyboardInterrupt:
            logger.info("Extraction interrupted. Run again to resume.")
            break
        except Exception as e:
            logger.error(f"Failed to process {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Clear memory between models
        clear_memory()


if __name__ == "__main__":
    main()
