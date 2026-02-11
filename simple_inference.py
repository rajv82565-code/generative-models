import torch 
from diffusers import DiffusionPipeline 

# Changed to float16 to fix the dtype mismatch error
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True
).to("cuda")

prompt = "A futuristic cyberpunk city with neon lights, raining, highly detailed, cinematic lighting" 
image = pipe(prompt).images[0]

# Added so you can see the result
image.save("output.png")
print("Image saved as output.png")
