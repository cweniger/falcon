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
    ],
    extras_require={
        "monitor": ["textual>=0.40.0"],
    },
    entry_points={
        "console_scripts": [
            "falcon=falcon.cli:main",
        ],
    },
)
