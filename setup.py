from setuptools import setup, find_packages

setup(
    name="mas_ids",
    version="1.1.0",
    description="Multi-Agent Intrusion Detection System for DoS/DDoS and Jamming in UAV/UGV networks",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24", "pandas>=2.0", "scipy>=1.10",
        "scikit-learn>=1.3", "tensorflow>=2.15", "torch>=2.0",
    ],
)
