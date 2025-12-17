# TIBVH - Taichi-based Linear BVH

A high-performance Python package for 3D spatial data structures and geometry processing using Taichi for GPU acceleration.

## Installation

### Requirements

- Python 3.8+
- Taichi 1.6.0+
- NumPy 1.19.0+
- PyTorch 1.10.0+

### Install from Source

```bash
git clone https://github.com/TATP-233/tibvh.git
cd tibvh
pip install -e .
```

### Supported Geometry Types

- **PLANE (0)**: Infinite planes with optional bounds
- **SPHERE (2)**: Spheres defined by center and radius
- **CAPSULE (3)**: Capsules (cylinders with hemispherical caps)
- **ELLIPSOID (4)**: Ellipsoids with three different radii
- **CYLINDER (5)**: Finite cylinders with caps
- **BOX (6)**: Oriented bounding boxes
- **TRIANGLES**: Individual triangles for mesh processing

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments

- Based on the Linear BVH construction algorithm by Tero Karras (NVIDIA Research)
- Powered by the [Taichi](https://github.com/taichi-dev/taichi) programming language
- Inspired by modern GPU-accelerated spatial data structures

## References

- Karras, T. (2012). "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees". NVIDIA Research.
- Lauterbach, C. et al. (2009). "Fast BVH Construction on GPUs". Computer Graphics Forum.