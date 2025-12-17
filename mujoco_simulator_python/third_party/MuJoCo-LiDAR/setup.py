from setuptools import setup

setup(
    name="mujoco_lidar",
    version="0.2.1",
    author="Yufei Jia",
    author_email="jyf23@mails.tsinghua.edu.cn",
    description="A LiDAR sensor designed for MuJoCo",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/TATP-233/MuJoCo-LiDAR.git",
    packages=["mujoco_lidar"],
    package_data={"mujoco_lidar": ["scan_mode/*.npy"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy >= 1.20.0",
        "mujoco >= 3.2.0",
        "scipy",
        "pynput",
        "matplotlib",
        "zhplot",
        "taichi >= 1.6.0",
        "tibvh @ git+https://github.com/TATP-233/tibvh.git",
    ],
    extras_require={
        "jax": [
            "jax",
            "jaxlib",
        ],
    },
)