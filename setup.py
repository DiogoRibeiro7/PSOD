from setuptools import find_packages, setup

# TODO: Update version management - consider using setuptools-scm
VERSION = "0.1.0"

# TODO: Update author email with actual email
AUTHOR_EMAIL = "dfr@esmad.ipp.pt"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# TODO: Add more classifiers once package is more mature
setup(
    name="outlier-pseudo-supervised",
    version=VERSION,
    author="Diogo Ribeiro",
    author_email=AUTHOR_EMAIL,
    description="Outlier detection using pseudo-supervised learning approach",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/diogoribeiro7/outlier_pseudo_supervised",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        # TODO: Add Python 3.12 support and testing
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "category-encoders>=2.3.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
            "sphinx>=4.0",
            # TODO: Add pre-commit to dev dependencies
        ],
        "viz": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
        # TODO: Add 'docs' extra with documentation dependencies
    },
    # TODO: Add entry points for command-line interface if needed
    project_urls={
        "Bug Reports": "https://github.com/diogoribeiro7/outlier_pseudo_supervised/issues",
        "Source": "https://github.com/diogoribeiro7/outlier_pseudo_supervised",
        # TODO: Add documentation URL once docs are hosted
    },
)

