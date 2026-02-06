from setuptools import setup, find_packages

setup(
    name="falcon",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "wandb>=0.15.0",
        "torch>=2.0.0",
        "numpy",
        "ray",
        "sbi",
        "omegaconf",
        "coolname",
        "rich>=13.0.0",
        "blessed>=1.20.0",
    ],
    extras_require={
        "monitor": ["textual>=0.40.0"],
        "docs": [
            "mkdocs>=1.5",
            "mkdocs-material>=9.0",
            "mkdocstrings[python]>=0.24",
        ],
    },
    entry_points={
        "console_scripts": [
            "falcon=falcon.cli:main",
        ],
    },
)
