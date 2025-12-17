from setuptools import setup

setup(
    name="tibvh",
    version="0.1.0",
    author="Unknown",
    author_email="",
    description="A library for BVH geometry processing",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="",
    packages=["tibvh", "tibvh.geometry", "tibvh.lbvh"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[],
)