#!/usr/bin/env python3
"""
Setup script for ACTS v6.0
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="acts-v6",
    version="6.0.0",
    author="ACTS Development Team",
    author_email="dev@acts-trading.com",
    description="Adaptive Causal Trading System v6.0 - Complete Integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/acts-v6",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "full": [
            "transformers>=4.30.0",
            "faiss-cpu>=1.7.4",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "production": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "prometheus-client>=0.17.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "acts-v6=acts_v6_complete:main",
        ],
    },
)
