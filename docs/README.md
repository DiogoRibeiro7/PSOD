# PSOD Documentation

This directory contains the source files for PSOD documentation built with Sphinx.

## Building the Documentation

### Prerequisites

Install the project with the documentation extra from the repository root:

```bash
python -m pip install -e ".[docs]"
```

The `docs` extra is the authoritative documentation dependency set. It includes the Sphinx theme and extensions used by `docs/conf.py`, plus a bundled Pandoc binary for notebook conversion.

### Build HTML Documentation

**Linux/macOS:**
```bash
make html
```

**Windows (if make doesn't work):**
```bash
python -m sphinx -M html . _build
```

The built documentation will be in `_build/html/`. Open `_build/html/index.html` in your browser.

### Other Build Options

```bash
make latexpdf    # Build PDF documentation
make epub        # Build EPUB documentation
make linkcheck   # Check for broken links
make coverage    # Check documentation coverage
make clean       # Remove build artifacts
```

## Documentation Structure

```
docs/
├── conf.py              # Sphinx configuration
├── index.rst            # Main documentation index
├── api/                 # API reference documentation
├── user_guide/          # User guides and tutorials
├── examples/            # Example gallery
├── _static/             # Static assets (CSS, images)
└── _templates/          # Custom HTML templates
```

## Contributing to Documentation

1. Edit RST files in the appropriate directory.
2. Build locally with `make html`.
3. Check links with `make linkcheck`.
4. Submit a pull request.

## Automated Builds

Documentation is built by GitHub Actions and can also be built by Read the Docs. During the repository refactor, documentation publication should be treated as provisional until the documentation workflow itself is fully audited.

## Troubleshooting

If a local build fails, reinstall the docs extra and rebuild from a clean directory:

```bash
python -m pip install -e ".[docs]" --upgrade
make clean
make html
```

If notebook conversion reports `PandocMissing`, verify that the environment was installed from the current `docs` extra. It uses `pypandoc-binary`, which supplies a bundled Pandoc binary. A system Pandoc installation is therefore optional rather than required for the standard development setup.

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [PyData Sphinx Theme](https://pydata-sphinx-theme.readthedocs.io/)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html)
