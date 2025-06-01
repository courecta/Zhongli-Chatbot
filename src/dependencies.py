"""
Dependency management and optional imports for the Zhongli Chatbot project
"""

# Import necessary libraries directly
import torch
import pandas as pd
import numpy as np
from PIL import Image
import requests
import gradio as gr
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, login
import chardet
import git

# The following imports are kept as try-except blocks as they were in the original code
# and might be optional or have fallbacks.
try:
    from torch import autocast
except ImportError:
    autocast = None

# It's generally better to let the program fail if a critical dependency is missing,
# or to handle it at the point of use if it's truly optional.
# The previous complex checking logic has been removed for simplification.

# Example of how to use these imports in other files:
# from .dependencies import pd, torch, AutoTokenizer, etc.
