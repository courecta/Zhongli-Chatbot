"""
Training execution and monitoring for DialoGPT model
"""

import os
import time
from typing import Optional, Dict, Any

try:
    import torch
except ImportError:
    print("torch not installed. Install with: pip install torch")
    torch = None

try:
    from transformers import (
        Trainer,
        DataCollatorForLanguageModeling,
        EarlyStoppingCallback
    )
except ImportError:
    print("transformers not installed. Install with: pip install transformers")
    Trainer = None
    DataCollatorForLanguageModeling = None
    EarlyStoppingCallback = None

class TrainingMonitor:
    """Monitor training progress and metrics"""

    def __init__(self):
        self.start_time = None
        self.metrics = []

    def start_training(self):
        """Start training timer"""
        self.start_time = time.time()
        print("Training started...")

    def log_metrics(self, metrics: Dict[str, float]):
        """Log training metrics"""
        self.metrics.append({
            'timestamp': time.time(),
            'metrics': metrics
        })

    def get_elapsed_time(self) -> float:
        """Get elapsed training time in minutes"""
        if self.start_time is None:
            return 0
        return (time.time() - self.start_time) / 60

def setup_trainer(model, tokenizer, train_dataset, val_dataset, training_args):
    """Setup HuggingFace Trainer"""

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
        return_tensors="pt"
    )

    # Early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=2,
        early_stopping_threshold=0.01
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[early_stopping]
    )

    return trainer

def clear_output_directory(output_dir: str):
    """Clear output directory before training"""
    if os.path.exists(output_dir):
        import shutil
        try:
            shutil.rmtree(output_dir)
            print(f"Cleared output directory: {output_dir}")
        except Exception as e:
            print(f"Warning: Could not clear output directory: {e}")

    os.makedirs(output_dir, exist_ok=True)

def check_disk_space(required_gb: float = 5.0) -> bool:
    """Check available disk space"""
    import shutil

    try:
        free_bytes = shutil.disk_usage('.').free
        free_gb = free_bytes / (1024**3)

        if free_gb >= required_gb:
            print(f"Disk space OK: {free_gb:.1f} GB available")
            return True
        else:
            print(f"Warning: Low disk space: {free_gb:.1f} GB available, {required_gb} GB required")
            return False
    except Exception as e:
        print(f"Could not check disk space: {e}")
        return True

def start_training(trainer, monitor: Optional[TrainingMonitor] = None):
    """Start the training process"""

    if monitor:
        monitor.start_training()

    try:
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Start training
        print("Starting model training...")
        training_result = trainer.train()

        # Log completion
        if monitor:
            elapsed = monitor.get_elapsed_time()
            print(f"Training completed in {elapsed:.1f} minutes")

        return training_result

    except Exception as e:
        print(f"Training failed: {e}")
        return None

def evaluate_model(trainer, tokenizer):
    """Evaluate the trained model"""
    try:
        print("Evaluating model...")
        eval_results = trainer.evaluate()

        print("Evaluation results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value:.4f}")

        # Test generation
        test_prompt = "Hello, how are you?"
        inputs = tokenizer.encode(test_prompt, return_tensors='pt')

        if torch.cuda.is_available():
            inputs = inputs.to('cuda')

        with torch.no_grad():
            outputs = trainer.model.generate(
                inputs,
                max_length=inputs.shape[1] + 50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Test generation: {response}")

        return eval_results

    except Exception as e:
        print(f"Evaluation failed: {e}")
        return None

def save_model(trainer, output_dir: str):
    """Save the trained model"""
    try:
        print(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        trainer.tokenizer.save_pretrained(output_dir)
        print("Model saved successfully")
        return True
    except Exception as e:
        print(f"Failed to save model: {e}")
        return False

def run_complete_training():
    """Run the complete training pipeline"""
    from data_loader import main as load_data
    from model_setup import main as setup_model
    from dataset_creator import main as create_datasets

    print("=== Starting Complete Training Pipeline ===")

    # Load data
    print("\n1. Loading data...")
    train_data, val_data = load_data()
    if train_data is None:
        print("Failed to load data")
        return False

    # Setup model
    print("\n2. Setting up model...")
    model, tokenizer, config, training_args = setup_model()
    if model is None:
        print("Failed to setup model")
        return False

    # Create datasets
    print("\n3. Creating datasets...")
    train_dataset, val_dataset = create_datasets()
    if train_dataset is None:
        print("Failed to create datasets")
        return False

    # Check prerequisites
    print("\n4. Checking prerequisites...")
    if not check_disk_space():
        print("Insufficient disk space")
        return False

    clear_output_directory(training_args.output_dir)

    # Setup trainer
    print("\n5. Setting up trainer...")
    trainer = setup_trainer(model, tokenizer, train_dataset, val_dataset, training_args)

    # Start training
    print("\n6. Starting training...")
    monitor = TrainingMonitor()
    result = start_training(trainer, monitor)

    if result is None:
        print("Training failed")
        return False

    # Evaluate model
    print("\n7. Evaluating model...")
    eval_results = evaluate_model(trainer, tokenizer)

    # Save model
    print("\n8. Saving model...")
    save_success = save_model(trainer, training_args.output_dir)

    if save_success:
        print("\n=== Training Pipeline Completed Successfully ===")
        print(f"Model saved to: {training_args.output_dir}")
        return True
    else:
        print("\n=== Training Pipeline Failed ===")
        return False

def main():
    """Main training function"""
    return run_complete_training()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
