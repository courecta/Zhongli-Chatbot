"""
Model upload and deployment utilities
"""

import os
import subprocess
from typing import Optional

try:
    from huggingface_hub import HfApi, login
except ImportError:
    print("huggingface_hub not installed. Install with: pip install huggingface_hub")
    HfApi = None
    login = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("transformers not installed. Install with: pip install transformers")
    AutoTokenizer = None
    AutoModelForCausalLM = None

try:
    import git
except ImportError:
    print("gitpython not installed. Install with: pip install gitpython")
    git = None

def setup_huggingface_auth(token: Optional[str] = None):
    """Setup HuggingFace authentication"""
    try:
        if token is None:
            # Try to get from environment or Colab secrets
            try:
                from google.colab import userdata  # type: ignore
                token = userdata.get('HUGGINGFACE_TOKEN')
            except ImportError:
                print("No HuggingFace token found")
                return False

        login(token=token, add_to_git_credential=True)
        print("HuggingFace authentication successful")
        return True

    except Exception as e:
        print(f"Authentication failed: {e}")
        return False

def create_model_repo(model_name: str, private: bool = False):
    """Create a new model repository on HuggingFace Hub"""
    try:
        api = HfApi()
        repo_url = api.create_repo(
            repo_id=model_name,
            private=private,
            repo_type="model"
        )
        print(f"Repository created: {repo_url}")
        return True
    except Exception as e:
        print(f"Failed to create repository: {e}")
        return False

def prepare_model_for_upload(model_dir: str, model_name: str):
    """Prepare model directory for upload"""
    try:
        # Initialize git repository
        repo = git.Repo.init(model_dir)

        # Configure git
        with repo.config_writer() as config:
            config.set_value("user", "name", "Zhongli Bot")
            config.set_value("user", "email", "zhongli@genshin.com")

        # Install git-lfs
        subprocess.run(["git", "lfs", "install"], cwd=model_dir, check=True)

        # Track large files
        lfs_files = [
            "*.bin",
            "*.safetensors",
            "*.h5",
            "pytorch_model.bin"
        ]

        gitattributes_path = os.path.join(model_dir, ".gitattributes")
        with open(gitattributes_path, "w") as f:
            for pattern in lfs_files:
                f.write(f"{pattern} filter=lfs diff=lfs merge=lfs -text\n")

        # Add files
        repo.index.add(["."])
        repo.index.commit("Initial commit")

        print("Model prepared for upload")
        return True

    except Exception as e:
        print(f"Failed to prepare model: {e}")
        return False

def upload_model_to_hub(model_dir: str, model_name: str, commit_message: str = "Upload model"):
    """Upload model to HuggingFace Hub"""
    try:
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Push to hub
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

        print(f"Model uploaded successfully to: https://huggingface.co/{model_name}")
        return True

    except Exception as e:
        print(f"Upload failed: {e}")
        return False

def create_model_card(model_dir: str, model_name: str, training_info: dict = None):
    """Create a model card README"""

    model_card = f"""---
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

    readme_path = os.path.join(model_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(model_card)

    print("Model card created")

def complete_upload_workflow(model_dir: str, model_name: str):
    """Complete model upload workflow"""
    print(f"Starting upload workflow for {model_name}")

    # Check if model directory exists
    if not os.path.exists(model_dir):
        print(f"Model directory not found: {model_dir}")
        return False

    # Setup authentication
    if not setup_huggingface_auth():
        print("Authentication failed")
        return False

    # Create repository
    if not create_model_repo(model_name, private=False):
        print("Repository creation failed")
        return False

    # Create model card
    create_model_card(model_dir, model_name)

    # Prepare for upload
    if not prepare_model_for_upload(model_dir, model_name):
        print("Model preparation failed")
        return False

    # Upload to hub
    if not upload_model_to_hub(model_dir, model_name):
        print("Upload failed")
        return False

    print("Upload workflow completed successfully")
    return True

def main():
    """Main upload function"""
    model_dir = "output-modern"
    model_name = "DialoGPT-small-Zhongli"

    if not os.path.exists(model_dir):
        print(f"No trained model found at {model_dir}")
        print("Please run training first")
        return False

    return complete_upload_workflow(model_dir, model_name)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)