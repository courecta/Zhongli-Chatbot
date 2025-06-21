"""
Model upload and deployment utilities for Zhongli chatbot.

This module provides functionality to upload trained models to HuggingFace Hub
with proper authentication, repository creation, and model card generation.
"""

import os
import subprocess
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import dependencies with better error handling
class Dependencies:
    """Handles optional dependencies with graceful fallbacks."""
    
    def __init__(self):
        self.huggingface_hub = self._import_huggingface_hub()
        self.transformers = self._import_transformers()
        self.git = self._import_git()
    
    def _import_huggingface_hub(self):
        try:
            from huggingface_hub import HfApi, login
            return {"HfApi": HfApi, "login": login}
        except ImportError:
            logger.warning("huggingface_hub not installed. Install with: pip install huggingface_hub")
            return None
    
    def _import_transformers(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            return {"AutoTokenizer": AutoTokenizer, "AutoModelForCausalLM": AutoModelForCausalLM}
        except ImportError:
            logger.warning("transformers not installed. Install with: pip install transformers")
            return None
    
    def _import_git(self):
        try:
            import git
            return git
        except ImportError:
            logger.warning("gitpython not installed. Install with: pip install gitpython")
            return None


class HuggingFaceAuth:
    """Handles HuggingFace authentication."""
    
    def __init__(self, deps: Dependencies):
        self.deps = deps
    
    def setup_authentication(self, token: Optional[str] = None) -> bool:
        """
        Setup HuggingFace authentication.
        
        Args:
            token: Optional HuggingFace token. If None, tries to get from environment.
            
        Returns:
            bool: True if authentication successful, False otherwise.
        """
        if not self.deps.huggingface_hub:
            logger.error("huggingface_hub not available")
            return False
        
        try:
            if token is None:
                token = self._get_token_from_environment()
                if not token:
                    logger.error("No HuggingFace token found")
                    return False
            
            login_func = self.deps.huggingface_hub["login"]
            login_func(token=token, add_to_git_credential=True)
            logger.info("HuggingFace authentication successful")
            return True
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    def _get_token_from_environment(self) -> Optional[str]:
        """Try to get token from various sources."""
        # Try environment variable first
        token = os.getenv('HUGGINGFACE_TOKEN')
        if token:
            return token
        
        # Try Google Colab secrets
        try:
            from google.colab import userdata  # type: ignore
            return userdata.get('HUGGINGFACE_TOKEN')
        except ImportError:
            pass
        
        return None


class ModelRepository:
    """Handles model repository operations."""
    
    def __init__(self, deps: Dependencies):
        self.deps = deps
    
    def create_repository(self, model_name: str, private: bool = False) -> bool:
        """
        Create a new model repository on HuggingFace Hub.
        
        Args:
            model_name: Name of the model repository.
            private: Whether the repository should be private.
            
        Returns:
            bool: True if repository created successfully, False otherwise.
        """
        if not self.deps.huggingface_hub:
            logger.error("huggingface_hub not available")
            return False
        
        try:
            api = self.deps.huggingface_hub["HfApi"]()
            repo_url = api.create_repo(
                repo_id=model_name,
                private=private,
                repo_type="model"
            )
            logger.info(f"Repository created: {repo_url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create repository: {e}")
            return False


class ModelCardGenerator:
    """Generates model cards for uploaded models."""
    
    @staticmethod
    def create_model_card(
        model_dir: str, 
        model_name: str, 
        training_info: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a model card README.
        
        Args:
            model_dir: Directory containing the model.
            model_name: Name of the model.
            training_info: Optional training information.
            
        Returns:
            bool: True if model card created successfully, False otherwise.
        """
        try:
            model_card_content = ModelCardGenerator._generate_model_card_content(
                model_name, training_info
            )
            
            readme_path = os.path.join(model_dir, "README.md")
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(model_card_content)
            
            logger.info("Model card created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create model card: {e}")
            return False
    
    @staticmethod
    def _generate_model_card_content(model_name: str, training_info: Optional[Dict[str, Any]] = None) -> str:
        """Generate the model card content."""
        return f"""---
license: mit
tags:
- conversational
- genshin-impact
- zhongli
- dialogpt
language:
- en
pipeline_tag: text-generation
---

# {model_name}

A fine-tuned DialoGPT model trained on Zhongli dialogue from Genshin Impact.

## Model Description

This model is based on Microsoft's DialoGPT and has been fine-tuned to generate responses in the style and personality of Zhongli from Genshin Impact.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained('{model_name}')
model = AutoModelForCausalLM.from_pretrained('{model_name}')

# Generate response
input_text = "Hello, Zhongli!"
input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')

with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + 50,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
```

## Training

- Base model: microsoft/DialoGPT-small
- Training data: Zhongli dialogue dataset
- Fine-tuning approach: Causal language modeling

## Limitations

This model is designed for entertainment and should not be used for:
- Providing factual information about Genshin Impact lore
- Making financial or legal decisions
- Any critical applications

## License

MIT License
"""


class GitManager:
    """Handles git operations for model preparation."""
    
    def __init__(self, deps: Dependencies):
        self.deps = deps
    
    def prepare_model_for_upload(self, model_dir: str, model_name: str) -> bool:
        """
        Prepare model directory for upload with git and git-lfs.
        
        Args:
            model_dir: Directory containing the model.
            model_name: Name of the model.
            
        Returns:
            bool: True if preparation successful, False otherwise.
        """
        if not self.deps.git:
            logger.error("gitpython not available")
            return False
        
        try:
            self._initialize_git_repo(model_dir)
            self._setup_git_lfs(model_dir)
            self._commit_initial_files(model_dir)
            
            logger.info("Model prepared for upload successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to prepare model: {e}")
            return False
    
    def _initialize_git_repo(self, model_dir: str) -> None:
        """Initialize git repository and configure user."""
        if not self.deps.git:
            raise RuntimeError("Git dependency not available")
            
        repo = self.deps.git.Repo.init(model_dir)
        
        with repo.config_writer() as config:
            config.set_value("user", "name", "Zhongli Bot")
            config.set_value("user", "email", "zhongli@genshin.com")
    
    def _setup_git_lfs(self, model_dir: str) -> None:
        """Setup git-lfs for large files."""
        subprocess.run(["git", "lfs", "install"], cwd=model_dir, check=True)
        
        lfs_patterns = [
            "*.bin",
            "*.safetensors", 
            "*.h5",
            "pytorch_model.bin"
        ]
        
        gitattributes_path = os.path.join(model_dir, ".gitattributes")
        with open(gitattributes_path, "w") as f:
            for pattern in lfs_patterns:
                f.write(f"{pattern} filter=lfs diff=lfs merge=lfs -text\n")
    
    def _commit_initial_files(self, model_dir: str) -> None:
        """Add and commit all files."""
        if not self.deps.git:
            raise RuntimeError("Git dependency not available")
            
        repo = self.deps.git.Repo(model_dir)
        repo.index.add(["."])
        repo.index.commit("Initial commit")


class ModelUploader:
    """Handles model upload to HuggingFace Hub."""
    
    def __init__(self, deps: Dependencies):
        self.deps = deps
    
    def upload_to_hub(
        self, 
        model_dir: str, 
        model_name: str, 
        commit_message: str = "Upload model"
    ) -> bool:
        """
        Upload model to HuggingFace Hub.
        
        Args:
            model_dir: Directory containing the model.
            model_name: Name of the model on the hub.
            commit_message: Commit message for the upload.
            
        Returns:
            bool: True if upload successful, False otherwise.
        """
        if not self.deps.transformers:
            logger.error("transformers not available")
            return False
        
        try:
            model, tokenizer = self._load_model_and_tokenizer(model_dir)
            self._push_to_hub(model, tokenizer, model_name, commit_message)
            
            logger.info(f"Model uploaded successfully to: https://huggingface.co/{model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False
    
    def _load_model_and_tokenizer(self, model_dir: str):
        """Load model and tokenizer from directory."""
        if not self.deps.transformers:
            raise RuntimeError("Transformers dependency not available")
            
        AutoModelForCausalLM = self.deps.transformers["AutoModelForCausalLM"]
        AutoTokenizer = self.deps.transformers["AutoTokenizer"]
        
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        return model, tokenizer
    
    def _push_to_hub(self, model, tokenizer, model_name: str, commit_message: str) -> None:
        """Push model and tokenizer to hub."""
        model.push_to_hub(
            model_name,
            commit_message=commit_message,
            use_auth_token=True
        )
        
        tokenizer.push_to_hub(
            model_name,
            commit_message=commit_message,
            use_auth_token=True
        )


class ModelUploadWorkflow:
    """Orchestrates the complete model upload workflow."""
    
    def __init__(self):
        self.deps = Dependencies()
        self.auth = HuggingFaceAuth(self.deps)
        self.repository = ModelRepository(self.deps)
        self.card_generator = ModelCardGenerator()
        self.git_manager = GitManager(self.deps)
        self.uploader = ModelUploader(self.deps)
    
    def execute_complete_workflow(self, model_dir: str, model_name: str) -> bool:
        """
        Execute the complete model upload workflow.
        
        Args:
            model_dir: Directory containing the trained model.
            model_name: Name for the model on HuggingFace Hub.
            
        Returns:
            bool: True if workflow completed successfully, False otherwise.
        """
        logger.info(f"Starting upload workflow for {model_name}")
        
        # Validate inputs
        if not self._validate_inputs(model_dir):
            return False
        
        # Execute workflow steps
        workflow_steps = [
            ("Authentication", lambda: self.auth.setup_authentication()),
            ("Repository creation", lambda: self.repository.create_repository(model_name, private=False)),
            ("Model card creation", lambda: self.card_generator.create_model_card(model_dir, model_name)),
            ("Model preparation", lambda: self.git_manager.prepare_model_for_upload(model_dir, model_name)),
            ("Model upload", lambda: self.uploader.upload_to_hub(model_dir, model_name))
        ]
        
        for step_name, step_func in workflow_steps:
            logger.info(f"Executing: {step_name}")
            if not step_func():
                logger.error(f"{step_name} failed")
                return False
        
        logger.info("Upload workflow completed successfully")
        return True
    
    def _validate_inputs(self, model_dir: str) -> bool:
        """Validate input parameters."""
        if not os.path.exists(model_dir):
            logger.error(f"Model directory not found: {model_dir}")
            return False
        
        required_files = ["config.json", "pytorch_model.bin"]
        for file_name in required_files:
            file_path = os.path.join(model_dir, file_name)
            if not os.path.exists(file_path):
                logger.warning(f"Expected file not found: {file_path}")
        
        return True


def main() -> bool:
    """
    Main upload function.
    
    Returns:
        bool: True if upload successful, False otherwise.
    """
    # Configuration
    model_dir = "output-modern"
    model_name = "DialoGPT-small-Zhongli"
    
    # Validate model exists
    if not os.path.exists(model_dir):
        logger.error(f"No trained model found at {model_dir}")
        logger.info("Please run training first")
        return False
    
    # Execute upload workflow
    workflow = ModelUploadWorkflow()
    return workflow.execute_complete_workflow(model_dir, model_name)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)