# text-to-image-generator.py
from diffusers import StableDiffusionPipeline
import torch

model_id = "nitrosocke/mo-di-diffusion"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to("cpu")

prompt = "a sun rise over a field of flowers, vibrant colors, high detail, digital art"

print("Generating image...")
image = pipe(prompt).images[0]
image.save("flowers.png")
print("Image saved!")
