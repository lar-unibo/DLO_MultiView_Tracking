from setuptools import setup, find_packages

setup(
    name="dlo_python",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "depthai",
        "numpy==1.24.4",
        "matplotlib",
        "wandb",
        "tqdm",
        "timm",
        "shutils",
        "scipy==1.14.1",

    ],
)
