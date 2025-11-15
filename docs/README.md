# PSOD Documentation

This directory contains the source files for PSOD documentation built with Sphinx.

## Building the Documentation

### Prerequisites

```bash
pip install -r requirements.txt
```

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

1. Edit RST files in the appropriate directory
2. Build locally to preview changes: `make html`
3. Check for broken links: `make linkcheck`
4. Submit a pull request

## Writing Documentation

### RST Syntax

- Use `**bold**` for **bold text**
- Use `` `code` `` for `inline code`
- Use `.. code-block:: python` for code blocks
- Use `.. note::` for note admonitions

### Adding Examples

1. Create a Python script in `../examples/` with `plot_` prefix
2. The script will automatically appear in the gallery
3. Use docstrings to describe the example

### API Documentation

API documentation is automatically generated from docstrings. Use NumPy or Google style:

```python
def my_function(param1, param2):
    """
    Short description.

    Longer description if needed.

    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type
        Description of param2

    Returns
    -------
    type
        Description of return value

    Examples
    --------
    >>> my_function(1, 2)
    3
    """
    return param1 + param2
```

## Automated Builds

Documentation is automatically built and deployed:

- **GitHub Actions**: Builds on every push
- **Read the Docs**: Available at https://psod.readthedocs.io (when configured)
- **GitHub Pages**: Available at https://your-username.github.io/PSOD/ (when configured)

## Troubleshooting

### Build Errors

If you encounter build errors, try:

```bash
make clean
make html
```

### Missing Dependencies

```bash
pip install -r requirements.txt --upgrade
```

### Pandoc Missing Error

If you see `PandocMissing` error, Pandoc is required for Jupyter notebook conversion. The `pypandoc` package (included in `requirements.txt`) bundles Pandoc automatically. If issues persist, install Pandoc manually:

- **Windows**: `winget install --source winget --exact --id JohnMacFarlane.Pandoc`
- **macOS**: `brew install pandoc`
- **Linux**: `sudo apt-get install pandoc` or `sudo yum install pandoc`

### Broken Links

Check for broken links:

```bash
make linkcheck
```

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [PyData Sphinx Theme](https://pydata-sphinx-theme.readthedocs.io/)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html)
