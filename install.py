# This script is called by ComfyUI-Manager & Comfy-CLI after requirements.txt is installed: 
# https://github.com/ltdrdata/ComfyUI-Manager/tree/386af67a4c34db3525aa89af47a6f78c819926f2?tab=readme-ov-file#custom-node-support-guide

import sys
import os
from os.path import dirname
import glob
import subprocess
import traceback
import platform

if sys.argv[0] == 'install.py':
    sys.path.append('.')   # for portable version

COMFY3D_ROOT_ABS_PATH = dirname(__file__)
BUILD_SCRIPT_ROOT_ABS_PATH = os.path.join(COMFY3D_ROOT_ABS_PATH, "_Pre_Builds/_Build_Scripts")
sys.path.append(BUILD_SCRIPT_ROOT_ABS_PATH)

try:
    from build_utils import (
        get_platform_config_name,
        git_folder_parallel,
        install_remote_packages,
        install_platform_packages,
        install_isolated_packages,
        wheels_dir_exists_and_not_empty,
        build_config,
        PYTHON_PATH,
        WHEELS_ROOT_ABS_PATH,
        PYTHON_VERSION
    )
    from shared_utils.log_utils import cstr
    
    # --- BLACKWELL & CUDA 13+ COMPATIBILITY LAYER ---
    def apply_bleeding_edge_patches():
        """Injects environment variables to fix 'std' ambiguity and header conflicts on new hardware"""
        import torch
        major_v, minor_v = torch.cuda.get_device_capability()
        
        # Check for Blackwell (sm_120) or newer
        if major_v >= 12:
            cstr(f"Blackwell Architecture detected (sm_{major_v}{minor_v}). Applying build patches...").msg.print()
            # This flag prevents the 'ambiguous symbol std' error in modern CUDA headers
            os.environ["CFLAGS"] = os.environ.get("CFLAGS", "") + " -DbtInverseDynamicsQuat_h"
            os.environ["CXXFLAGS"] = os.environ.get("CXXFLAGS", "") + " -DbtInverseDynamicsQuat_h"
            # Force max optimizations for the 5080
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major_v}.{minor_v}"
        
        # Patch for Python 3.12+ shared libraries
        if sys.version_info >= (3, 12):
            os.environ["DISTUTILS_USE_SDK"] = "1"

    # --- END PATCH LAYER ---

    try:
        import github
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "PyGithub"])

    def install_local_wheels(builds_dir):
        """Install all wheels from local directory"""
        wheel_files = glob.glob(os.path.join(builds_dir, "**/*.whl"), recursive=True)
        if not wheel_files:
            cstr("No wheel files found in directory").warning.print()
            return False
            
        success_count = 0
        for wheel_path in wheel_files:
            # Note: Using --no-build-isolation is safer for local envs with custom Torch/CUDA
            result = subprocess.run([PYTHON_PATH, "-m", "pip", "install", "--no-deps", "--force-reinstall", wheel_path], 
                                text=True, capture_output=True)
            if result.returncode == 0:
                cstr(f"Successfully installed wheel: {os.path.basename(wheel_path)}").msg.print()
                success_count += 1
            else:
                cstr(f"Failed to install wheel {os.path.basename(wheel_path)}: {result.stderr}").error.print()
        
        return success_count == len(wheel_files)

    def try_wheels_first_approach():
        platform_config_name = get_platform_config_name()
        builds_dir = os.path.join(WHEELS_ROOT_ABS_PATH, platform_config_name)
        
        # On Blackwell, we almost always expect wheels to fail unless they are locally built
        # because the remote repo usually doesn't have sm_120 wheels yet.
        cstr("Checking for hardware-compatible wheels...").msg.print()
        
        if wheels_dir_exists_and_not_empty(builds_dir):
            if install_local_wheels(builds_dir):
                return True
        
        remote_builds_dir_name = f"{build_config.wheels_dir_name}/{platform_config_name}"
        if git_folder_parallel(build_config.repo_id, remote_builds_dir_name, recursive=True, root_outdir=builds_dir):
            return install_local_wheels(builds_dir)
        
        return False

    def try_auto_build_all(builds_dir):
        cstr(f"Starting hardware-optimized build for {platform.processor()}...").msg.print()
        apply_bleeding_edge_patches()
        
        # Explicitly build mandatory helpers first for 3D processing
        mandatory_helpers = ["fvcore", "iopath", "ninja"]
        subprocess.run([PYTHON_PATH, "-m", "pip", "install"] + mandatory_helpers)

        result = subprocess.run(
            [PYTHON_PATH, "auto_build_all.py", "--output_root_dir", builds_dir], 
            cwd=BUILD_SCRIPT_ROOT_ABS_PATH, text=True, capture_output=True
        )
        
        cstr(f"[Comfy3D BUILD LOG]\n{result.stdout}").msg.print()
        if result.returncode != 0:
            cstr(f"[Comfy3D BUILD ERROR LOG]\n{result.stderr}").error.print()
            
        return result.returncode == 0

    # 1. Initial tool check
    build_tools = ["setuptools", "wheel", "ninja", "cmake", "pytest-runner"]
    subprocess.run([PYTHON_PATH, "-m", "pip", "install", "--upgrade"] + build_tools)

    # 2. Install base packages
    install_remote_packages(build_config.build_base_packages)
    install_platform_packages()
    
    # 3. Handle Blackwell-sensitive packages (pytorch3d, nvdiffrast, etc.)
    if hasattr(build_config, 'isolated_packages'):
        # For these, we often need to build from source to ensure sm_120 support
        apply_bleeding_edge_patches()
        install_isolated_packages(build_config.isolated_packages)

    # 4. Main installation sequence
    platform_config_name = get_platform_config_name()
    builds_dir = os.path.join(WHEELS_ROOT_ABS_PATH, platform_config_name)
    
    if not try_wheels_first_approach():
        cstr("Compatible wheels not found. Initiating local source build...").warning.print()
        if try_auto_build_all(builds_dir):
            install_local_wheels(builds_dir)
            cstr("Successfully compiled Comfy3D for your hardware!").msg.print()
        else:
            raise RuntimeError("Automated build failed. Please check the logs above.")

    # 5. Finalize CPP headers
    remote_pycpp_dir_name = f"_Python_Source_cpp/{PYTHON_VERSION}"
    python_root_dir = dirname(PYTHON_PATH)
    git_folder_parallel(build_config.repo_id, remote_pycpp_dir_name, recursive=True, root_outdir=python_root_dir)
    
    cstr("Successfully installed Comfy3D! Enjoy your RTX 5080 speed.").msg.print()
    
except Exception as e:
    traceback.print_exc()
    cstr("Installation failed. Since you are on Blackwell, manual compilation is often required: https://github.com/MrForExample/ComfyUI-3D-Pack/").error.print()