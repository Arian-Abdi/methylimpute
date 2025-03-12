from setuptools import setup, find_packages

setup(
    name="methylimpute",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.19.0",
        "scikit-learn>=0.23.0",
    ],
    entry_points={
        'console_scripts': [
            'methylimpute=methylimpute.cli:main',
        ],
    },
    author="Arian Abdi",
    author_email="arian.abdipour9@gmail.com",
    description="A package for mapping CpG sites to chromosomes and imputing missing methylation beta values",
    keywords="methylation, EPIC array, imputation, MICE",
    python_requires=">=3.7",
)