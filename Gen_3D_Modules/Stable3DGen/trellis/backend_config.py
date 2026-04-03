# trellis/backend_config.py
from typing import *
import os
import logging
import importlib

# Setup Logger
logger = logging.getLogger(__name__)

# Global variables
BACKEND = 'spconv'  # Default sparse backend
DEBUG = False        # Debug mode flag
ATTN = 'sdpa'       # Default set to sdpa for compatibility (RTX 5080/PyTorch 2.0+)
SPCONV_ALGO = 'implicit_gemm'  # Default algorithm

def get_spconv_algo() -> str:
    """Get current spconv algorithm."""
    global SPCONV_ALGO
    return SPCONV_ALGO

def set_spconv_algo(algo: Literal['implicit_gemm', 'native', 'auto']) -> bool:
    """Set spconv algorithm with validation."""
    global SPCONV_ALGO
    
    if algo not in ['implicit_gemm', 'native', 'auto']:
        logger.warning(f"Invalid spconv algorithm: {algo}. Must be 'implicit_gemm', 'native', or 'auto'")
        return False
        
    SPCONV_ALGO = algo
    os.environ['SPCONV_ALGO'] = algo
    logger.info(f"Set spconv algorithm to: {algo}")
    return True

def _try_import_xformers() -> bool:
    try:
        import xformers.ops
        return True
    except ImportError:
        return False

def _try_import_flash_attn() -> bool:
    try:
        import flash_attn
        return True
    except ImportError:
        return False

def _try_import_sageattention() -> bool:
    try:
        import torch.nn.functional as F
        from sageattention import sageattn
        F.scaled_dot_product_attention = sageattn
        return True
    except ImportError:
        return False

def _try_import_spconv() -> bool:
    try:
        import spconv
        return True
    except ImportError:
        return False

def _try_import_torchsparse() -> bool:
    try:
        import torchsparse
        return True
    except ImportError:
        return False

def get_available_backends() -> Dict[str, bool]:
    """Return dict of available attention backends and their status"""
    return {
        'xformers': _try_import_xformers(),
        'flash_attn': _try_import_flash_attn(),
        'sage': _try_import_sageattention(),
        'naive': True,
        'sdpa': True  # Always available with PyTorch >= 2.0
    }

def get_available_sparse_backends() -> Dict[str, bool]:
    """Return dict of available sparse backends and their status"""
    return {
        'spconv': _try_import_spconv(),
        'torchsparse': _try_import_torchsparse()
    }

def get_attention_backend() -> str:
    """Get current attention backend"""
    global ATTN
    return ATTN

def get_sparse_backend() -> str:
    """Get current sparse backend"""
    global BACKEND
    return BACKEND

def get_debug_mode() -> bool:
    """Get current debug mode status"""
    global DEBUG
    return DEBUG

def __from_env():
    """Initialize settings from environment variables"""
    global BACKEND
    global DEBUG
    global ATTN
    
    env_sparse_backend = os.environ.get('SPARSE_BACKEND')
    env_sparse_debug = os.environ.get('SPARSE_DEBUG')
    env_sparse_attn = os.environ.get('SPARSE_ATTN_BACKEND')
    
    if env_sparse_backend is not None and env_sparse_backend in ['spconv', 'torchsparse']:
        BACKEND = env_sparse_backend
    if env_sparse_debug is not None:
        DEBUG = env_sparse_debug == '1'
    
    # If environment variable exists and is valid, use it; otherwise, default is now 'sdpa'
    if env_sparse_attn is not None and env_sparse_attn in ['xformers', 'flash_attn', 'sage', 'sdpa', 'naive']:
        ATTN = env_sparse_attn
    else:
        # Final fallback check: if ATTN is still 'xformers' but not installed, force 'sdpa'
        if ATTN == 'xformers' and not _try_import_xformers():
            ATTN = 'sdpa'

    os.environ['SPARSE_ATTN_BACKEND'] = ATTN
    os.environ['ATTN_BACKEND'] = ATTN
    
    logger.info(f"[SPARSE] Backend: {BACKEND}, Attention: {ATTN}")

def set_backend(backend: Literal['spconv', 'torchsparse']) -> bool:
    """Set sparse backend with validation"""
    global BACKEND
    
    backend = backend.lower().strip()
    logger.info(f"Setting sparse backend to: {backend}")

    if backend == 'spconv':
        if _try_import_spconv():
            BACKEND = 'spconv'
            os.environ['SPARSE_BACKEND'] = 'spconv'
            return True
        else:
            logger.warning("spconv not available")
            return False
            
    elif backend == 'torchsparse':
        if _try_import_torchsparse():
            BACKEND = 'torchsparse'
            os.environ['SPARSE_BACKEND'] = 'torchsparse'
            return True
        else:
            logger.warning("torchsparse not available")
            return False
    
    return False

def set_sparse_backend(backend: Literal['spconv', 'torchsparse'], algo: str = None) -> bool:
    result = set_backend(backend)
    if algo is not None and result:
        set_spconv_algo(algo)
    return result

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug
    os.environ['SPARSE_DEBUG'] = '1' if debug else '0'

def set_attn(attn: Literal['xformers', 'flash_attn', 'sage', 'sdpa', 'naive']) -> bool:
    """Set attention backend with validation"""
    global ATTN
    attn = attn.lower().strip()
    logger.info(f"Setting attention backend to: {attn}")

    if attn == 'xformers' and _try_import_xformers():
        ATTN = 'xformers'
    elif attn == 'flash_attn' and _try_import_flash_attn():
        ATTN = 'flash_attn'
    elif attn == 'sage' and _try_import_sageattention():
        ATTN = 'sage'
    elif attn == 'sdpa':
        ATTN = 'sdpa'
    elif attn == 'naive':
        ATTN = 'naive'
    else:
        logger.warning(f"Attention backend {attn} not available or failed to import")
        return False

    os.environ['SPARSE_ATTN_BACKEND'] = ATTN
    os.environ['ATTN_BACKEND'] = ATTN
    return True

def set_attention_backend(backend: Literal['xformers', 'flash_attn', 'sage', 'sdpa']) -> bool:
    return set_attn(backend)

# Initialize from environment variables on module import
__from_env()