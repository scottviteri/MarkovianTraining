from setuptools import setup, find_packages

setup(
    name="markovian_training",
    version="0.1.0",
    description="Markovian Training with Vector Quantization",
    author="",
    author_email="",
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.10.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
) 