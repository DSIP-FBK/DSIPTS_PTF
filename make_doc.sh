#!/bin/bash
# Documentation build script for DSIPTS
# This script generates API documentation and builds HTML docs using Sphinx

set -e  # Exit on error

echo "========================================"
echo "DSIPTS Documentation Build Script"
echo "========================================"

# Clean up Python cache files
echo "\n[1/5] Cleaning Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Generate API documentation
echo "\n[2/5] Generating API documentation with sphinx-apidoc..."
sphinx-apidoc -f -o docs/ dsipts/ \
    --separate \
    --module-first \
    --no-toc \
    -M \
    --implicit-namespaces

# Optional: Generate bash_examples documentation if needed
# sphinx-apidoc -f -o docs/ bash_examples/

# Clean previous build
echo "\n[3/5] Cleaning previous build..."
cd docs
make clean

# Build HTML documentation
echo "\n[4/5] Building HTML documentation..."
make html

cd ..

# Optional: Generate PDF documentation if pandoc is available
echo "\n[5/5] Generating PDF documentation (optional)..."
if command -v pandoc &> /dev/null; then
    if [ -f "docs/README.md" ]; then
        pandoc docs/README.md -o docs/_build/html/dsipts.pdf -V geometry:landscape 2>/dev/null || echo "Warning: PDF generation failed for main README"
    fi
    if [ -f "bash_examples/README.md" ]; then
        cd bash_examples
        pandoc README.md -o ../docs/_build/html/bash_examples.pdf -V geometry:landscape 2>/dev/null || echo "Warning: PDF generation failed for bash_examples README"
        cd ..
    fi
else
    echo "Pandoc not found. Skipping PDF generation."
fi

echo "\n========================================"
echo "Documentation build complete!"
echo "HTML docs: docs/_build/html/index.html"
echo "========================================"
