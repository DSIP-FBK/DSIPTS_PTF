# Building Documentation

## Quick Build
```bash
bash make_doc.sh
```

## Manual Build
```bash
cd docs
make clean html
```

## View
```bash
xdg-open _build/html/index.html
```

## Dependencies
```bash
pip install -e ".[docs]"
```

Requires: sphinx, sphinx_pdj_theme, myst-parser

## Directory Structure
- `conf.py` - Sphinx configuration
- `index.rst` - Main page
- `modules.rst` - API reference
- `getting-started/` - Getting started guides
- `user-guide/` - User documentation
- `development/` - Developer documentation
- `_build/` - Generated docs (gitignored)

## Adding Documentation
1. Create `.rst` or `.md` file
2. Add to `toctree` in parent file
3. Build and verify

## Regenerate API Docs
```bash
sphinx-apidoc -f -o . dsipts/ --separate --module-first --no-toc
```

## CI/CD
Builds and deploys to GitHub Pages on push to main/master.
See `.github/workflows/docs.yml`
