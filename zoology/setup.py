from setuptools import setup


_REQUIRED = [
    "numpy",
    "einops",
    "tqdm",
    "click",
    "pydantic",
    "wandb",
    "flash-linear-attention",
    "rotary-embedding-torch",
    "causal_conv1d",
    "einx",
    "transformers",
]

_OPTIONAL = {
    "analysis": [
        "pandas",
        "seaborn",
        "matplotlib",
    ],
    "extra":[
        "rich", 
        "ray",
        "PyYAML",
    ]
}

# ensure that torch is installed, and send to torch website if not
try:
    import torch
except ModuleNotFoundError:
    raise ValueError("Please install torch first: https://pytorch.org/get-started/locally/")

setup(
    name="zoology", 
    version="0.0.1",
    description="",
    packages=["zoology"],  
    install_requires=_REQUIRED,
    extras_require=_OPTIONAL,
    entry_points={
        'console_scripts': ['zg=zoology.cli:cli'],
    },
)
