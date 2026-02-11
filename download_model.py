import os
from huggingface_hub import hf_hub_download

def download_sdxl():
    repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
    filename = "sd_xl_base_1.0.safetensors"
    local_dir = "checkpoints"
    
    print(f"Starting download of {filename} from {repo_id}...")
    print("This is a 13GB file, so it may take a while depending on your internet speed.")
    
    try:
        os.makedirs(local_dir, exist_ok=True)
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"\nSuccess! Model downloaded to: {path}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("\nTry downloading manually from: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors")

if __name__ == "__main__":
    download_sdxl()
