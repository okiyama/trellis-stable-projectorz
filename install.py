import subprocess
import sys
import os
import time
from typing import Optional, Tuple
from pathlib import Path
import urllib.request
import urllib.error
import socket

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


class InstallationError(Exception):
    """Custom exception for installation failures"""
    pass

def check_connectivity(url: str = "https://pytorch.org", timeout: int = 5) -> Tuple[bool, Optional[str]]:
    """
    Check internet connectivity and return more detailed error information
    """
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True, None
    except urllib.error.URLError as e:
        if isinstance(e.reason, socket.gaierror):
            return False, f"DNS resolution failed: {e.reason}"
        elif isinstance(e.reason, socket.timeout):
            return False, "Connection timed out"
        else:
            return False, f"Connection failed: {e.reason}"
    except Exception as e:
        return False, f"Unknown error: {str(e)}"

def run_command_with_retry(cmd: str, desc: Optional[str] = None, max_retries: int = MAX_RETRIES) -> subprocess.CompletedProcess:
    """Run a command with retry logic"""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"\nRetry attempt {attempt + 1}/{max_retries} for: {desc or cmd}")
                # Check connectivity before retry
                connected, error_msg = check_connectivity()
                if not connected:
                    print(f"Connection check failed: {error_msg}")
                    print(f"Waiting {RETRY_DELAY} seconds before retry...")
                    time.sleep(RETRY_DELAY)
            
            # Prepare command
            if cmd.startswith('pip install'):
                args = cmd[11:]
                cmd = f'"{sys.executable}" -m pip install --no-cache-dir --isolated {args}'
            
            # Show progress for pip installations
            if "pip install" in cmd:
                if "--progress-bar" not in cmd:
                    cmd += " --progress-bar=on"
                # Use capture_output but still show progress
                result = subprocess.run(cmd, shell=True, text=True, stdout=sys.stdout, stderr=subprocess.PIPE)
            else:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                return result
            
            last_error = result
            print(f"\nCommand failed (attempt {attempt + 1}/{max_retries}):")
            if hasattr(result, 'stderr') and result.stderr:
                print(f"Error output:\n{result.stderr}")
            
        except Exception as e:
            last_error = e
            print(f"\nException during {desc or cmd} (attempt {attempt + 1}/{max_retries}):")
            print(str(e))
        
        if attempt < max_retries - 1:
            print(f"Waiting {RETRY_DELAY} seconds before retry...")
            time.sleep(RETRY_DELAY)
    
    # If we get here, all retries failed
    raise InstallationError(f"Command failed after {max_retries} attempts: {last_error}")


def check_internet():
    try:
        import urllib.request
        urllib.request.urlopen('https://pytorch.org', timeout=5)
        return True
    except:
        return False

def install_dependencies():
    """Install all required dependencies with improved error handling"""
    try:
        # Check initial connectivity
        connected, error_msg = check_connectivity()
        if not connected:
            print(f"Error: Internet connectivity check failed: {error_msg}")
            print("Please check your connection and try again.")
            sys.exit(1)
        
        # Install packages with retry logic
        packages = [
            ("pip install -r requirements.txt", "Installing basic dependencies"),
            (f"pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118", "Installing PyTorch 2.1.2 with CUDA 11.8"),
            ("pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118", "Installing xformers"),
            ("pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8", "Installing utils3d"),
            ("pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.2_cu118.html", "Installing Kaolin"),
            ("pip install spconv-cu118==2.3.6", "Installing spconv"),
        ]

        # Install from local wheel files
        wheel_files = {
            "nvdiffrast": "whl/nvdiffrast-0.3.3-cp311-cp311-win_amd64.whl",
            "diffoctreerast": "whl/diffoctreerast-0.0.0-cp311-cp311-win_amd64.whl",
            "diff_gaussian": "whl/diff_gaussian_rasterization-0.0.0-cp311-cp311-win_amd64.whl"
        }
        # Install packages with retry
        for cmd, desc in packages:
            run_command_with_retry(cmd, desc)

        # Install wheel files
        for name, wheel_path in wheel_files.items():
            wheel = Path(wheel_path)
            if not wheel.exists():
                raise InstallationError(f"Required wheel file not found: {wheel}")
            run_command_with_retry(f"pip install {wheel}", f"Installing {name} from local wheel")

        # Install Gradio last
        run_command_with_retry(
            "pip install gradio==4.44.1 gradio_litmodel3d==0.0.1",
            "Installing gradio for web app"
        )
        print("\nInstallation completed successfully!")

    except InstallationError as e:
        print(f"\nInstallation failed: {str(e)}")
        print("\nSuggestions:")
        print("1. Check your internet connection")
        print("2. Verify your firewall/antivirus isn't blocking connections")
        print("3. Try using a different network if possible")
        print("4. Check if you need to configure git proxy settings")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during installation: {str(e)}")
        sys.exit(1)


def verify_installation():
    """Verify that critical packages were installed correctly"""
    try:
        import torch
        import gradio
        import kaolin
        print(f"PyTorch version: {torch.__version__}")
        print(f"Gradio version: {gradio.__version__}")
        print(f"Kaolin version: {kaolin.__version__}")
        return True
    except ImportError as e:
        print(f"Verification failed: {str(e)}")
        return False


if __name__ == "__main__":
    install_dependencies()
    if verify_installation():
        print("\nInstallation completed and verified successfully!")
    else:
        print("\nInstallation completed but verification failed.")
        sys.exit(1)