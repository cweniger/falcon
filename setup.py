from setuptools import setup, find_packages

setup(
    name="falcon",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "wandb>=0.15.0",
        "torch>=2.0.0",
        "zarr",
        "numpy",
        "ray",
        "sbi",
        "hydra-core",
        "omegaconf",
    ],
    entry_points={
        'console_scripts': [
            'falcon=falcon.cli:main',
        ],
    },
)