"""
Training execution and monitoring for DialoGPT model.

This module provides comprehensive training functionality including
monitoring, evaluation, and model saving with proper error handling.
"""

import os
import time
import shutil
from typing import Optional, Dict, Any, Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Dependencies:
    """Handles optional dependencies with graceful fallbacks."""
    
    def __init__(self):
        self.torch = self._import_torch()
        self.transformers = self._import_transformers()
    
    def _import_torch(self):
        try:
            import torch
            return torch
        except ImportError:
            logger.warning("torch not installed. Install with: pip install torch")
            return None
    
    def _import_transformers(self):
        try:
            from transformers.trainer import (
                Trainer,
                DataCollatorForLanguageModeling,
                EarlyStoppingCallback
            )
            return {
                "Trainer": Trainer,
                "DataCollatorForLanguageModeling": DataCollatorForLanguageModeling,
                "EarlyStoppingCallback": EarlyStoppingCallback
            }
        except ImportError:
            logger.warning("transformers not installed. Install with: pip install transformers")
            return None


class TrainingMonitor:
    """Monitor training progress and metrics."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.metrics: List[Dict[str, Any]] = []

    def start_training(self) -> None:
        """Start training timer."""
        self.start_time = time.time()
        logger.info("Training started...")

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log training metrics."""
        self.metrics.append({
            'timestamp': time.time(),
            'metrics': metrics
        })

    def get_elapsed_time(self) -> float:
        """Get elapsed training time in minutes."""
        if self.start_time is None:
            return 0.0
        return (time.time() - self.start_time) / 60.0


class TrainerSetup:
    """Handles trainer configuration and setup."""
    
    def __init__(self, deps: Dependencies):
        self.deps = deps
    
    def setup_trainer(self, model, tokenizer, train_dataset, val_dataset, training_args):
        """Setup HuggingFace Trainer with proper configuration."""
        if not self.deps.transformers:
            raise RuntimeError("transformers dependency not available")
        
        try:
            # Data collator
            data_collator = self.deps.transformers["DataCollatorForLanguageModeling"](
                tokenizer=tokenizer,
                mlm=False,  # Causal LM, not masked LM
                return_tensors="pt"
            )

            # Early stopping callback
            early_stopping = self.deps.transformers["EarlyStoppingCallback"](
                early_stopping_patience=2,
                early_stopping_threshold=0.01
            )

            # Create trainer
            trainer = self.deps.transformers["Trainer"](
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                callbacks=[early_stopping]
            )

            return trainer
            
        except Exception as e:
            logger.error(f"Failed to setup trainer: {e}")
            raise


class FileSystemManager:
    """Handles file system operations for training."""
    
    @staticmethod
    def clear_output_directory(output_dir: str) -> bool:
        """Clear output directory before training."""
        try:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
                logger.info(f"Cleared output directory: {output_dir}")

            os.makedirs(output_dir, exist_ok=True)
            return True
            
        except Exception as e:
            logger.error(f"Could not clear output directory: {e}")
            return False

    @staticmethod
    def check_disk_space(required_gb: float = 5.0) -> bool:
        """Check available disk space."""
        try:
            free_bytes = shutil.disk_usage('.').free
            free_gb = free_bytes / (1024**3)

            if free_gb >= required_gb:
                logger.info(f"Disk space OK: {free_gb:.1f} GB available")
                return True
            else:
                logger.warning(f"Low disk space: {free_gb:.1f} GB available, {required_gb} GB required")
                return False
                
        except Exception as e:
            logger.error(f"Could not check disk space: {e}")
            return True  # Assume OK if we can't check


class ModelTrainer:
    """Handles the actual training process."""
    
    def __init__(self, deps: Dependencies):
        self.deps = deps
    
    def start_training(self, trainer, monitor: Optional[TrainingMonitor] = None):
        """Start the training process."""
        if not self.deps.torch:
            raise RuntimeError("torch dependency not available")
        
        if monitor:
            monitor.start_training()

        try:
            # Clear CUDA cache
            if self.deps.torch.cuda.is_available():
                self.deps.torch.cuda.empty_cache()

            # Start training
            logger.info("Starting model training...")
            training_result = trainer.train()

            # Log completion
            if monitor:
                elapsed = monitor.get_elapsed_time()
                logger.info(f"Training completed in {elapsed:.1f} minutes")

            return training_result

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return None


class ModelEvaluator:
    """Handles model evaluation and testing."""
    
    def __init__(self, deps: Dependencies):
        self.deps = deps
    
    def evaluate_model(self, trainer, tokenizer) -> Optional[Dict[str, float]]:
        """Evaluate the trained model."""
        if not self.deps.torch:
            logger.error("torch dependency not available")
            return None
        
        try:
            logger.info("Evaluating model...")
            eval_results = trainer.evaluate()

            logger.info("Evaluation results:")
            for key, value in eval_results.items():
                logger.info(f"  {key}: {value:.4f}")

            # Test generation
            self._test_generation(trainer, tokenizer)

            return eval_results

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return None
    
    def _test_generation(self, trainer, tokenizer) -> None:
        """Test model generation with a sample prompt."""
        try:
            test_prompt = "Hello, how are you?"
            inputs = tokenizer.encode(test_prompt, return_tensors='pt')

            if self.deps.torch.cuda.is_available():
                inputs = inputs.to('cuda')

            with self.deps.torch.no_grad():
                outputs = trainer.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 50,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Test generation: {response}")
            
        except Exception as e:
            logger.warning(f"Test generation failed: {e}")


class ModelSaver:
    """Handles model saving operations."""
    
    @staticmethod
    def save_model(trainer, output_dir: str) -> bool:
        """Save the trained model."""
        try:
            logger.info(f"Saving model to {output_dir}")
            trainer.save_model(output_dir)
            trainer.tokenizer.save_pretrained(output_dir)
            logger.info("Model saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False


class TrainingPipeline:
    """Orchestrates the complete training pipeline."""
    
    def __init__(self):
        self.deps = Dependencies()
        self.trainer_setup = TrainerSetup(self.deps)
        self.fs_manager = FileSystemManager()
        self.model_trainer = ModelTrainer(self.deps)
        self.evaluator = ModelEvaluator(self.deps)
        self.saver = ModelSaver()
    
    def run_complete_training(self) -> bool:
        """Run the complete training pipeline."""
        logger.info("=== Starting Complete Training Pipeline ===")

        try:
            # Load data
            train_data, val_data = self._load_data()
            if train_data is None:
                return False

            # Setup model
            model, tokenizer, config, training_args = self._setup_model()
            if model is None:
                return False

            # Create datasets
            train_dataset, val_dataset = self._create_datasets()
            if train_dataset is None:
                return False

            # Check prerequisites
            if not self._check_prerequisites(training_args.output_dir):
                return False

            # Setup trainer
            trainer = self._setup_trainer(model, tokenizer, train_dataset, val_dataset, training_args)

            # Execute training workflow
            return self._execute_training_workflow(trainer, tokenizer, training_args.output_dir)
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            return False
    
    def _load_data(self) -> Tuple[Optional[Any], Optional[Any]]:
        """Load training and validation data."""
        try:
            logger.info("1. Loading data...")
            from data_loader import main as load_data
            return load_data()
        except ImportError as e:
            logger.error(f"Failed to import data_loader: {e}")
            return None, None
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return None, None
    
    def _setup_model(self) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Any]]:
        """Setup model, tokenizer, config, and training arguments."""
        try:
            logger.info("2. Setting up model...")
            from model_setup import main as setup_model
            return setup_model()
        except ImportError as e:
            logger.error(f"Failed to import model_setup: {e}")
            return None, None, None, None
        except Exception as e:
            logger.error(f"Failed to setup model: {e}")
            return None, None, None, None
    
    def _create_datasets(self) -> Tuple[Optional[Any], Optional[Any]]:
        """Create training and validation datasets."""
        try:
            logger.info("3. Creating datasets...")
            from dataset_creator import main as create_datasets
            return create_datasets()
        except ImportError as e:
            logger.error(f"Failed to import dataset_creator: {e}")
            return None, None
        except Exception as e:
            logger.error(f"Failed to create datasets: {e}")
            return None, None
    
    def _check_prerequisites(self, output_dir: str) -> bool:
        """Check prerequisites for training."""
        logger.info("4. Checking prerequisites...")
        
        if not self.fs_manager.check_disk_space():
            logger.error("Insufficient disk space")
            return False

        if not self.fs_manager.clear_output_directory(output_dir):
            logger.error("Failed to clear output directory")
            return False
        
        return True
    
    def _setup_trainer(self, model, tokenizer, train_dataset, val_dataset, training_args):
        """Setup the trainer."""
        try:
            logger.info("5. Setting up trainer...")
            return self.trainer_setup.setup_trainer(model, tokenizer, train_dataset, val_dataset, training_args)
        except Exception as e:
            logger.error(f"Failed to setup trainer: {e}")
            raise
    
    def _execute_training_workflow(self, trainer, tokenizer, output_dir: str) -> bool:
        """Execute the training workflow."""
        # Start training
        logger.info("6. Starting training...")
        monitor = TrainingMonitor()
        result = self.model_trainer.start_training(trainer, monitor)

        if result is None:
            logger.error("Training failed")
            return False

        # Evaluate model
        logger.info("7. Evaluating model...")
        eval_results = self.evaluator.evaluate_model(trainer, tokenizer)

        # Save model
        logger.info("8. Saving model...")
        save_success = self.saver.save_model(trainer, output_dir)

        if save_success:
            logger.info("=== Training Pipeline Completed Successfully ===")
            logger.info(f"Model saved to: {output_dir}")
            return True
        else:
            logger.error("=== Training Pipeline Failed ===")
            return False


def main() -> bool:
    """Main training function."""
    pipeline = TrainingPipeline()
    return pipeline.run_complete_training()


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)