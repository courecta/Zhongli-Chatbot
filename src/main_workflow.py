#!/usr/bin/env python3
"""
Main workflow orchestrator for Zhongli Chatbot training pipeline.

This script coordinates the entire training process by calling the appropriate
modules in sequence. Run with different commands to execute specific stages.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from data_loader import DataLoader, DataConfig
from model_setup import ModelSetup, ModelConfig
from dataset_creator import DatasetCreator, DatasetConfig
from trainer import Trainer, TrainingConfig
from model_uploader import ModelUploader, UploadConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """Orchestrates the complete training workflow."""

    def __init__(self):
        self.data_loader = None
        self.model_setup = None
        self.dataset_creator = None
        self.trainer = None
        self.uploader = None

    def setup_data_loading(self, data_path: str, output_dir: str = "data") -> None:
        """Initialize data loading component."""
        config = DataConfig(
            data_path=data_path,
            output_dir=output_dir,
            chunk_size=1000,
            overlap=200
        )
        self.data_loader = DataLoader(config)
        logger.info("Data loader initialized")

    def setup_model(self, model_name: str = "microsoft/DialoGPT-medium",
                   device: Optional[str] = None) -> None:
        """Initialize model setup component."""
        config = ModelConfig(
            model_name=model_name,
            device=device,
            use_gradient_checkpointing=True,
            use_8bit=True
        )
        self.model_setup = ModelSetup(config)
        logger.info(f"Model setup initialized with {model_name}")

    def setup_dataset(self, data_dir: str, max_length: int = 512) -> None:
        """Initialize dataset creation component."""
        config = DatasetConfig(
            data_dir=data_dir,
            max_length=max_length,
            context_window=3,
            min_response_length=10
        )
        self.dataset_creator = DatasetCreator(config)
        logger.info("Dataset creator initialized")

    def setup_training(self, output_dir: str = "trained_model",
                      epochs: int = 3, batch_size: int = 4) -> None:
        """Initialize training component."""
        config = TrainingConfig(
            output_dir=output_dir,
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=5e-5,
            warmup_steps=100,
            save_steps=500,
            eval_steps=500,
            max_grad_norm=1.0
        )
        self.trainer = Trainer(config)
        logger.info("Trainer initialized")

    def setup_uploader(self, repo_name: str, token: Optional[str] = None) -> None:
        """Initialize model uploader component."""
        config = UploadConfig(
            repo_name=repo_name,
            token=token,
            private=False
        )
        self.uploader = ModelUploader(config)
        logger.info("Model uploader initialized")

    def run_full_pipeline(self, data_path: str, model_name: str = "microsoft/DialoGPT-medium",
                         repo_name: Optional[str] = None) -> None:
        """Run the complete training pipeline."""
        try:
            # Step 1: Load and process data
            logger.info("Step 1: Loading and processing data...")
            self.setup_data_loading(data_path)
            processed_data = self.data_loader.load_and_process()

            # Step 2: Setup model
            logger.info("Step 2: Setting up model...")
            self.setup_model(model_name)
            model, tokenizer = self.model_setup.setup_model_and_tokenizer()

            # Step 3: Create dataset
            logger.info("Step 3: Creating training dataset...")
            self.setup_dataset("data")
            train_dataset, eval_dataset = self.dataset_creator.create_datasets(tokenizer)

            # Step 4: Train model
            logger.info("Step 4: Training model...")
            self.setup_training()
            trained_model = self.trainer.train(model, tokenizer, train_dataset, eval_dataset)

            # Step 5: Upload model (if repo specified)
            if repo_name:
                logger.info("Step 5: Uploading model...")
                self.setup_uploader(repo_name)
                self.uploader.upload_model(trained_model, tokenizer)

            logger.info("Pipeline completed successfully!")

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Zhongli Chatbot Training Pipeline")
    parser.add_argument("command", choices=["full", "data", "train", "upload"],
                       help="Command to execute")
    parser.add_argument("--data-path", required=True, help="Path to training data")
    parser.add_argument("--model-name", default="microsoft/DialoGPT-medium",
                       help="Base model name")
    parser.add_argument("--repo-name", help="HuggingFace repo name for upload")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")

    args = parser.parse_args()

    orchestrator = WorkflowOrchestrator()

    try:
        if args.command == "full":
            orchestrator.run_full_pipeline(args.data_path, args.model_name, args.repo_name)

        elif args.command == "data":
            orchestrator.setup_data_loading(args.data_path)
            orchestrator.data_loader.load_and_process()
            logger.info("Data processing completed")

        elif args.command == "train":
            orchestrator.setup_data_loading(args.data_path)
            orchestrator.data_loader.load_and_process()

            orchestrator.setup_model(args.model_name)
            model, tokenizer = orchestrator.model_setup.setup_model_and_tokenizer()

            orchestrator.setup_dataset("data")
            train_dataset, eval_dataset = orchestrator.dataset_creator.create_datasets(tokenizer)

            orchestrator.setup_training(epochs=args.epochs, batch_size=args.batch_size)
            orchestrator.trainer.train(model, tokenizer, train_dataset, eval_dataset)
            logger.info("Training completed")

        elif args.command == "upload":
            if not args.repo_name:
                logger.error("--repo-name required for upload command")
                sys.exit(1)

            orchestrator.setup_uploader(args.repo_name)
            # Load trained model and upload
            logger.info("Upload functionality - implement model loading from checkpoint")

    except Exception as e:
        logger.error(f"Command failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
