from setuptools import setup, find_packages

setup(
    name="memsched",
    version="0.1.0",
    description="KV Cache Aware Scheduling for LLM Serving",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.62.0",
    ],
)
