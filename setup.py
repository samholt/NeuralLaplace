import os
import re
import setuptools


def read(fname: str) -> str:
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def find_version() -> str:
    version_file = read("torchlaplace/__init__.py")
    version_raw = version_file.split('__version__')[1].split('"')[1]

    version = version_raw.group("version")
    return version


with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

required = [
    "torch>=1.3.0",
    "scipy>=1.4.0",
]

extras = {
    "experiments": [
        "ddeint",
        "keyboard",
        "matplotlib",
        "pandas",
        "sklearn",
        "stribor==0.1.0",
        "torchdiffeq>=0.2.1",
        "torchvision",
        "tqdm",
        "Sphinx",
        "sphinx-rtd-theme",
        "sphinx-panels",
        "nbsphinx"
    ]
}

extras['all'] = list(set([item for group in extras.values() for item in group]))



setuptools.setup(
    name="torchlaplace",
    version=find_version(),
    author="Sam Holt",
    author_email="samuel.holt.direct@gmail.com",
    description="Differentiable Laplace Reconstructions in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/samholt/NeuralLaplace",
    license="MIT",
    packages=setuptools.find_packages(),
    python_requires="~=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=required,
    extras_require=extras,
)
