"""
Model configuration and initialization for DialoGPT training
"""

from dataclasses import dataclass
from typing import Optional

try:
    import torch
except ImportError:
    print("torch not installed. Install with: pip install torch")
    torch = None

try:
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments
    )
except ImportError:
    print("transformers not installed. Install with: pip install transformers")
    AutoConfig = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TrainingArguments = None

@dataclass
class ModelConfig:
    """Model configuration settings"""
    model_name: str = 'microsoft/DialoGPT-small'
    model_type: str = 'gpt2'
    cache_dir: str = 'cached'
    output_dir: str = 'output-modern'
    block_size: int = 512

    # Training settings
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Evaluation settings
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"

    # Modern features
    fp16: bool = True
    dataloader_num_workers: int = 2
    remove_unused_columns: bool = False
    report_to: str = "tensorboard"
    seed: int = 42

def load_model_and_tokenizer(config: ModelConfig):
    """Load model and tokenizer with proper configuration"""
    try:
        print(f"Loading model: {config.model_name}")

        # Load configuration
        model_config = AutoConfig.from_pretrained(
            config.model_name,
            cache_dir=config.cache_dir
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            cache_dir=config.cache_dir
        )

        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            config=model_config,
            cache_dir=config.cache_dir,
            torch_dtype=torch.float16 if config.fp16 and torch.cuda.is_available() else torch.float32
        )

        # Resize token embeddings
        model.resize_token_embeddings(len(tokenizer))

        # Move to appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        print(f"Model loaded on {device}")
        print(f"Vocabulary size: {len(tokenizer)}")

        return model, tokenizer, model_config

    except Exception as e:
        print(f"Failed to load model: {e}")
        return None, None, None

def create_training_arguments(config: ModelConfig):
    """Create training arguments"""
    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        evaluation_strategy=config.evaluation_strategy,
        save_strategy=config.save_strategy,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        fp16=config.fp16 and torch.cuda.is_available(),
        dataloader_num_workers=config.dataloader_num_workers,
        remove_unused_columns=config.remove_unused_columns,
        report_to=config.report_to,
        seed=config.seed,
        logging_steps=10,
        logging_dir=f"{config.output_dir}/logs"
    )

def estimate_training_time(num_samples: int, config: ModelConfig) -> float:
    """Estimate training time in minutes"""
    # Rough estimation based on empirical data
    effective_batch_size = config.per_device_train_batch_size * config.gradient_accumulation_steps
    steps_per_epoch = num_samples // effective_batch_size
    total_steps = steps_per_epoch * config.num_train_epochs

    # Assume ~1 second per step on GPU, 3 seconds on CPU
    seconds_per_step = 1.0 if torch.cuda.is_available() else 3.0
    total_minutes = (total_steps * seconds_per_step) / 60

    return total_minutes

def check_system_requirements():
    """Check system requirements for training"""
    requirements = {
        'cuda_available': torch.cuda.is_available(),
        'gpu_memory': 0,
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

    if torch.cuda.is_available():
        requirements['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        requirements['gpu_name'] = torch.cuda.get_device_name(0)

    return requirements

def main():
    """Main model setup function"""
    # Create configuration
    config = ModelConfig()

    # Check system requirements
    sys_req = check_system_requirements()
    print(f"CUDA available: {sys_req['cuda_available']}")
    if sys_req['cuda_available']:
        print(f"GPU: {sys_req.get('gpu_name', 'Unknown')}")
        print(f"GPU Memory: {sys_req['gpu_memory']:.1f} GB")

    # Load model and tokenizer
    model, tokenizer, model_config = load_model_and_tokenizer(config)

    if model is None:
        print("Failed to load model")
        return None, None, None, None

    # Create training arguments
    training_args = create_training_arguments(config)

    print("Model setup complete")
    return model, tokenizer, model_config, training_args

if __name__ == "__main__":
    model, tokenizer, config, training_args = main()
