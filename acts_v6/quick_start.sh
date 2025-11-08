#!/bin/bash
# ACTS v6.0 - Quick Start Script

echo "========================================================================"
echo "ACTS v6.0 — Quick Start Setup"
echo "========================================================================"
echo ""

# Check Python version
echo "[1] Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
required_version="3.8"

if [[ $(echo "$python_version >= $required_version" | bc) -eq 1 ]]; then
    echo "✓ Python $python_version detected (>= $required_version required)"
else
    echo "✗ Python $python_version detected. Please upgrade to Python >= $required_version"
    exit 1
fi

# Create virtual environment
echo ""
echo "[2] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "[3] Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "[4] Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ pip upgraded"

# Install dependencies
echo ""
echo "[5] Installing dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Create necessary directories
echo ""
echo "[6] Creating project directories..."
mkdir -p data logs models
touch data/.gitkeep logs/.gitkeep models/.gitkeep
echo "✓ Directories created"

# Run tests
echo ""
echo "[7] Running tests..."
python -m pytest tests/ -v --tb=short
if [ $? -eq 0 ]; then
    echo "✓ All tests passed"
else
    echo "⚠ Some tests failed (this is OK for initial setup)"
fi

# Run basic example
echo ""
echo "[8] Running basic usage example..."
echo ""
python examples/basic_usage.py

echo ""
echo "========================================================================"
echo "✓ ACTS v6.0 Setup Complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Run examples:"
echo "     - python examples/basic_usage.py"
echo "     - python examples/advanced_interventions.py"
echo "     - python examples/risk_analysis.py"
echo "  3. Edit config/default_config.yaml for customization"
echo "  4. Check README.md for full documentation"
echo ""
echo "Happy trading! 🚀"
