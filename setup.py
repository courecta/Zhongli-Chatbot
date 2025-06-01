#!/usr/bin/env python3
"""
Setup script for Zhongli Chatbot Python training pipeline.
Run this script to verify installation and setup.
"""

import sys
import subprocess
import importlib
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8 or higher."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'torch',
        'transformers',
        'accelerate',
        'datasets',
        'gradio',
        'diffusers',
        'huggingface_hub',
        'pandas',
        'numpy'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)

    return missing_packages

def install_requirements():
    """Install requirements from requirements.txt."""
    try:
        print("📦 Installing requirements...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available: {device_count} device(s)")
            print(f"   Primary device: {device_name}")
            return True
        else:
            print("⚠️  CUDA not available, training will use CPU")
            return False
    except ImportError:
        print("❌ PyTorch not installed, cannot check CUDA")
        return False

def check_hf_token():
    """Check if HuggingFace token is available."""
    token = os.getenv('HUGGINGFACE_TOKEN')
    if token:
        print("✅ HuggingFace token found in environment")
        return True
    else:
        print("⚠️  HuggingFace token not found in environment")
        print("   Set HUGGINGFACE_TOKEN environment variable for model upload")
        return False

def create_sample_data():
    """Create sample training data for testing."""
    sample_data = [
        {"input": "Hello Zhongli", "output": "Greetings, traveler. How may I assist you today?"},
        {"input": "Tell me about contracts", "output": "A contract, once signed, must be honored. This is the foundation of all agreements."},
        {"input": "What is geo?", "output": "Geo represents the solid, enduring elements of this world. Like stone, it provides stability."},
        {"input": "Who are you?", "output": "I am Zhongli, consultant of the Wangsheng Funeral Parlor and former Geo Archon."},
    ]

    data_dir = Path("sample_data")
    data_dir.mkdir(exist_ok=True)

    import json
    with open(data_dir / "sample_conversations.json", "w") as f:
        json.dump(sample_data, f, indent=2)

    print(f"✅ Sample data created at {data_dir / 'sample_conversations.json'}")
    return str(data_dir / "sample_conversations.json")

def test_pipeline():
    """Test the training pipeline with sample data."""
    try:
        print("\n🧪 Testing pipeline components...")

        # Test data loader
        from src.data_loader import DataLoader, DataConfig
        sample_file = create_sample_data()

        config = DataConfig(data_path=sample_file, output_dir="test_data")
        loader = DataLoader(config)
        print("✅ DataLoader initialized")

        # Test model setup
        from src.model_setup import ModelSetup, ModelConfig
        model_config = ModelConfig(model_name="microsoft/DialoGPT-small")  # Use small for testing
        setup = ModelSetup(model_config)
        print("✅ ModelSetup initialized")

        # Test dataset creator
        from src.dataset_creator import DatasetCreator, DatasetConfig
        dataset_config = DatasetConfig(data_dir="test_data")
        creator = DatasetCreator(dataset_config)
        print("✅ DatasetCreator initialized")

        print("✅ All pipeline components working correctly!")
        return True

    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def main():
    """Main setup and verification."""
    print("🏛️ Zhongli Chatbot Setup and Verification\n")

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    print("\n📦 Checking installed packages...")
    missing = check_requirements()

    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        install_choice = input("Install missing packages? (y/n): ").lower()
        if install_choice == 'y':
            if not install_requirements():
                sys.exit(1)
        else:
            print("Please install missing packages manually")
            sys.exit(1)

    print("\n🖥️  Checking hardware...")
    check_cuda()

    print("\n🔑 Checking authentication...")
    check_hf_token()

    print("\n🧪 Testing pipeline...")
    if test_pipeline():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Set HUGGINGFACE_TOKEN environment variable if not already set")
        print("2. Prepare your training data")
        print("3. Run: python src/main_workflow.py full --data-path /path/to/your/data")
    else:
        print("\n❌ Setup verification failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
