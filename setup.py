"""
Setup script for HyperNova Optimizer
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hypernova-optimizer",
    version="0.1.0",
    author="HyperNova Team",
    author_email="hypernova@example.com",
    description="Production-grade optimizer library for all architectures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Zhuanz/HyperNova-Optimizer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "pydantic>=1.9.0",
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
        ],
        "wandb": ["wandb>=0.12.0"],
        "tensorboard": ["tensorboard>=2.0.0"],
    },
)
