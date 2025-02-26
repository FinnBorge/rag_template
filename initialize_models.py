#!/usr/bin/env python3
"""
Initialize RAG Bench by downloading required models.
"""
import os
import argparse
import shutil
import sys
from pathlib import Path
import logging
import requests
from tqdm import tqdm
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model download URLs and file paths
DEFAULT_MODELS = {
    "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf": {
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "size": 700_000_000,  # Approximate size in bytes
        "description": "TinyLlama 1.1B Chat (4-bit quantized version, very small)"
    },
    "phi-2.Q4_K_M.gguf": {
        "url": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
        "size": 1_300_000_000,  # Approximate size in bytes
        "description": "Phi-2 (4-bit quantized version, small Microsoft model)"
    },
    "orca-2-7b.Q4_K_M.gguf": {
        "url": "https://huggingface.co/TheBloke/Orca-2-7B-GGUF/resolve/main/orca-2-7b.Q4_K_M.gguf",
        "size": 3_800_000_000,  # Approximate size in bytes
        "description": "Orca 2 7B (4-bit quantized version, Microsoft model)"
    },
    "llama-3-8b-instruct.Q4_K_M.gguf": {
        "url": "https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF/resolve/main/llama-3-8b-instruct.Q4_K_M.gguf",
        "size": 4_600_000_000,  # Approximate size in bytes
        "description": "Llama 3 8B Instruct (requires special access approval)"
    }
}

def download_file(url, destination, file_size=None):
    """Download a file with progress bar."""
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        
        # Get size if not provided
        if file_size is None:
            file_size = int(response.headers.get('content-length', 0))
        
        # Show progress bar during download
        with open(destination, 'wb') as f:
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Downloading {os.path.basename(destination)}") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

def update_settings_yaml(model_path):
    """Update the settings.yaml file with the downloaded model path."""
    settings_path = Path('settings.yaml')
    
    if not settings_path.exists():
        logger.error(f"Settings file not found at {settings_path}")
        return False
    
    try:
        # Load settings
        with open(settings_path, 'r') as f:
            settings = yaml.safe_load(f)
        
        # Update model path
        if 'local_llm' in settings and 'model_path' in settings['local_llm']:
            settings['local_llm']['model_path'] = str(model_path)
            
            # Write updated settings
            with open(settings_path, 'w') as f:
                yaml.dump(settings, f, sort_keys=False)
            
            logger.info(f"Updated settings.yaml with model path: {model_path}")
            return True
    except Exception as e:
        logger.error(f"Error updating settings.yaml: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Initialize RAG Bench by downloading required models")
    parser.add_argument("--model", choices=list(DEFAULT_MODELS.keys()), 
                       default="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                       help="Which model to download")
    parser.add_argument("--models-dir", default="models", 
                       help="Directory to store models")
    parser.add_argument("--list-models", action="store_true",
                       help="List available models with descriptions")
    args = parser.parse_args()
    
    # List available models if requested
    if args.list_models:
        print("Available models:")
        for model_name, model_info in DEFAULT_MODELS.items():
            size_gb = model_info["size"] / 1_000_000_000
            print(f"  - {model_name:<30} ({size_gb:.1f} GB): {model_info['description']}")
        return
    
    models_dir = Path(args.models_dir)
    
    # Create models directory if it doesn't exist
    if not models_dir.exists():
        logger.info(f"Creating models directory: {models_dir}")
        models_dir.mkdir(parents=True)
    
    # Get model info
    model_name = args.model
    model_info = DEFAULT_MODELS[model_name]
    model_path = models_dir / model_name
    
    # Check if model already exists
    if model_path.exists():
        logger.info(f"Model already exists at {model_path}")
        symlink_path = models_dir / "default-model.gguf"
        
        # Create symlink to the model for easier reference
        if symlink_path.exists():
            symlink_path.unlink()
        # Create a relative symlink to avoid path issues
        os.chdir(models_dir)
        os.symlink(model_name, "default-model.gguf")
        os.chdir('..')
        logger.info(f"Created symlink: default-model.gguf -> {model_name}")
    else:
        # Download the model
        logger.info(f"Downloading {model_name} ({model_info['size'] / 1_000_000_000:.1f} GB)...")
        try:
            download_file(model_info["url"], model_path, model_info["size"])
            logger.info(f"Successfully downloaded {model_name} to {model_path}")
            
            # Create symlink to the model for easier reference
            symlink_path = models_dir / "default-model.gguf"
            if symlink_path.exists():
                symlink_path.unlink()
            # Create a relative symlink to avoid path issues
            os.chdir(models_dir)
            os.symlink(model_name, "default-model.gguf")
            os.chdir('..')
            logger.info(f"Created symlink: default-model.gguf -> {model_name}")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logger.error(f"Authentication error: {e}")
                logger.error("\nTo download this model, you need to authenticate with Hugging Face:")
                logger.error("1. Create an account at https://huggingface.co/join")
                logger.error("2. Create an access token at https://huggingface.co/settings/tokens")
                logger.error("3. Run: pip install huggingface_hub")
                logger.error("4. Run: huggingface-cli login")
                logger.error("For more details, see the README.md file.")
            else:
                logger.error(f"Error downloading model: {e}")
            
            if model_path.exists():
                model_path.unlink()
            return
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            if model_path.exists():
                model_path.unlink()
            return
    
    # Update settings.yaml
    update_settings_yaml(str(symlink_path))
    
    # Set model path in environment
    os.environ["LOCAL_MODEL_PATH"] = str(symlink_path)
    
    logger.info(f"RAG Bench initialization complete.")
    logger.info(f"You can now run: poetry run python -m rag_bench.main")

if __name__ == "__main__":
    main()