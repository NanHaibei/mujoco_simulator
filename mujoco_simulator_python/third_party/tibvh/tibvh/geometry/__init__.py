from . import aabb_generator
from . import geom_intersection
from . import utils

# Import key functions for convenient access
from .aabb_generator import (
    compute_triangle_aabb,
    compute_plane_aabb,
    compute_sphere_aabb,
    compute_capsule_aabb,
    compute_ellipsoid_aabb,
    compute_cylinder_aabb,
    compute_box_aabb,
    aabb_local2wolrd
)

from .geom_intersection import (
    ray_triangle_distance,
    ray_plane_distance,
    ray_sphere_distance,
    ray_capsule_distance,
    ray_cylinder_distance,
    ray_ellipsoid_distance,
    ray_box_distance,
)

from .utils import (
    _transform_ray_to_local,
    _transform_point_to_world
)

__all__ = [
    # Submodules
    "aabb_generator",
    "geom_intersection", 
    "utils",
    
    # AABB generation functions
    "compute_triangle_aabb",
    "compute_plane_aabb",
    "compute_sphere_aabb",
    "compute_capsule_aabb",
    "compute_ellipsoid_aabb",
    "compute_cylinder_aabb",
    "compute_box_aabb",
    "aabb_local2wolrd"

    # Ray-geometry intersection functions
    "ray_geom_intersection",
    "ray_plane_distance", 
    "ray_sphere_distance",
    "ray_box_distance",
    "ray_cylinder_distance",
    "ray_ellipsoid_distance",
    "ray_capsule_distance",
    
    # Mesh intersection functions
    "ray_triangle_distance",
    
    # Utility functions
    "_transform_ray_to_local",
    "_transform_point_to_world",
]
