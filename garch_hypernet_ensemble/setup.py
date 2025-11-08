"""Setup script for GARCH-HyperNetwork Ensemble."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="garch-hypernet-ensemble",
    version="2.0.0",
    author="AlgoTrading Team",
    description="Production-ready GARCH-HyperNetwork Adaptive Stacking Ensemble",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/garch-hypernet-ensemble",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "mypy>=1.7.1",
            "flake8>=6.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "garch-train=scripts.train:main",
            "garch-predict=scripts.predict:main",
        ],
    },
)
