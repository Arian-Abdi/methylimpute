"""
MethylImpute: A package for mapping CpG sites to chromosomes and imputing missing methylation beta values.

This package provides tools to:
1. Map CpG sites to chromosomes using Illumina manifest files
2. Impute missing beta values based on MICE (Multivariate Imputation by Chained Equations)
"""

__version__ = "0.1.0"

from .mapper import CpGMapper
from .imputer import MethylImputer

__all__ = ['CpGMapper', 'MethylImputer']