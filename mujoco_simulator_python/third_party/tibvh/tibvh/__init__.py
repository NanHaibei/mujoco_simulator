 
"""TIBVH - Taichi-based Linear BVH Implementation"""

from . import geometry
from . import lbvh

# Import key classes for convenient access
from .lbvh.aabb import AABB
from .lbvh.lbvh import LBVH

# Import geometry utilities
from .geometry import (
    aabb_generator,
    geom_intersection, 
    utils
)

# Version information
__version__ = "0.1.1"
__author__ = "TIBVH Contributors"
__email__ = "jyf23@mails.tsinghua.edu.cn"

# Package metadata
__title__ = "tibvh"
__description__ = "A high-performance Taichi-based Linear BVH implementation for 3D geometry processing"
__url__ = "https://github.com/TATP-233/tibvh"
__license__ = "MIT"

# Expose main classes at package level
__all__ = [
    # Core classes
    "AABB",
    "LBVH",
    
    # Submodules
    "geometry",
    "lbvh",
    
    # Geometry submodules
    "aabb_generator",
    "geom_intersection",
    "utils",
    
    # Package info
    "__version__",
    "__author__",
    "__email__",
]

# Initialize Taichi backend on import
try:
    import taichi as ti
    # Initialize with CPU by default, users can change if needed
    if not ti.is_logging_effective():
        ti.init(arch=ti.cpu, debug=False)
except ImportError:
    import warnings
    warnings.warn(
        "Taichi not found. Please install taichi: pip install taichi>=1.6.0",
        ImportWarning
    )
except Exception as e:
    import warnings
    warnings.warn(
        f"Failed to initialize Taichi: {e}. "
        "You may need to manually initialize with ti.init()",
        RuntimeWarning
    )