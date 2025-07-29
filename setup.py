from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))

setup(
    name="markovian_training",
    version="0.1.0",
    description="Markovian Training with Vector Quantization",
    author="Scott Viteri",
    author_email="scottviteri@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    py_modules=[
        "utils", "train", "evaluate_gsm8k", "constants", 
        "analyze_base_logprobs", "evaluate_cross_model",
        "log_file_quick_analysis", "perturbation_analysis",
        "plot_cot_answer_accuracy", "plot_training_metrics",
        "test_tokenizers"
    ],
    python_requires='>=3.8',
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.10.0",
        "peft>=0.4.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.2.0",
    ],
    entry_points={
        'console_scripts': [
            'markovian-train=train:main',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
) 