"""
Stable Diffusion with Gradio interface
Modern implementation for text-to-image generation
"""

import subprocess
from PIL import Image # Import Image directly
import sys
import gc
import warnings
from io import BytesIO
from typing import Optional, List, Union, TYPE_CHECKING

warnings.filterwarnings("ignore")

import gradio as gr
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
import requests

class ModernStableDiffusion:
    """Modern Stable Diffusion wrapper with proper error handling and memory management"""

    def __init__(self):
        self.txt2img_pipe = None
        self.img2img_pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def clear_memory(self):
        """Comprehensive memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def load_models(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        """Load Stable Diffusion models with modern best practices"""
        try:
            scheduler = DPMSolverMultistepScheduler.from_pretrained(
                model_id,
                subfolder="scheduler"
            )

            self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                scheduler=scheduler,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True
            )

            self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id,
                scheduler=scheduler,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True
            )

            if self.device == "cuda":
                self.txt2img_pipe = self.txt2img_pipe.to("cuda")
                self.img2img_pipe = self.img2img_pipe.to("cuda")

                self.txt2img_pipe.enable_attention_slicing()
                self.img2img_pipe.enable_attention_slicing()

            return True

        except Exception as e:
            print(f"Failed to load models: {e}")
            return False

    def generate_image(self,
                      prompt: str,
                      init_image: Optional[Image.Image] = None,
                      num_images: int = 1,
                      num_inference_steps: int = 20,
                      guidance_scale: float = 7.5,
                      strength: float = 0.75,
                      height: int = 512,
                      width: int = 512) -> Optional[List[Image.Image]]:
        """Generate images with proper error handling"""

        try:
            if not prompt or not prompt.strip():
                raise ValueError("Prompt cannot be empty")

            num_images = max(1, min(num_images, 4))
            num_inference_steps = max(1, min(num_inference_steps, 50))

            self.clear_memory()

            if init_image is not None:
                if self.img2img_pipe is None:
                    raise RuntimeError("Image-to-image pipeline not loaded")

                init_image = init_image.resize((width, height))

                with torch.inference_mode():
                    result = self.img2img_pipe(
                        prompt=prompt,
                        image=init_image,
                        num_images_per_prompt=num_images,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        strength=strength,
                        height=height,
                        width=width
                    )
            else:
                if self.txt2img_pipe is None:
                    raise RuntimeError("Text-to-image pipeline not loaded")

                with torch.inference_mode():
                    result = self.txt2img_pipe(
                        prompt=prompt,
                        num_images_per_prompt=num_images,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        height=height,
                        width=width
                    )

            self.clear_memory()
            return result.images

        except Exception as e:
            print(f"Generation failed: {e}")
            self.clear_memory()
            return None

def create_gradio_interface(sd_generator):
    """Create Gradio interface for Stable Diffusion"""

    def infer(prompt, init_image, samples_num, steps_num, scale, strength):
        if not prompt:
            return []

        images = sd_generator.generate_image(
            prompt=prompt,
            init_image=init_image,
            num_images=samples_num,
            num_inference_steps=steps_num,
            guidance_scale=scale,
            strength=strength
        )

        return images if images else []

    with gr.Blocks(css=".container { max-width: 800px; margin: auto; }") as demo:
        gr.Markdown("<h1><center>Stable Diffusion</center></h1>")

        with gr.Group():
            with gr.Box():
                with gr.Row():
                    text = gr.Textbox(
                        label="Enter your prompt",
                        show_label=False,
                        max_lines=1
                    )
                    btn = gr.Button("Run")

            with gr.Row():
                samples_num = gr.Slider(label="Images", minimum=1, maximum=2, value=2, step=1)
                steps_num = gr.Slider(label="Generation Steps", minimum=1, maximum=200, value=50, step=1)
                scale = gr.Slider(label="CFG Scale", minimum=0, maximum=50, value=7.5, step=0.1)
                strength = gr.Slider(label="Strength", minimum=0, maximum=1, value=0.75, step=0.01)

            image = gr.Image(label="Initial Image", type="pil")
            gallery = gr.Gallery(label="Generated images", show_label=False)

        text.submit(infer, inputs=[text, image, samples_num, steps_num, scale, strength], outputs=gallery)
        btn.click(infer, inputs=[text, image, samples_num, steps_num, scale, strength], outputs=gallery)

    return demo

def main():
    """Main function to run the Stable Diffusion interface"""
    try:
        sd_generator = ModernStableDiffusion()
        model_loaded = sd_generator.load_models()

        if model_loaded:
            demo = create_gradio_interface(sd_generator)
            demo.launch(debug=True)
        else:
            print("Failed to initialize Stable Diffusion")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
