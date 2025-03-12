"""Utility functions for the MethylImpute package."""

import pandas as pd
import numpy as np
import gc
import logging
import re

# Set up logger
logger = logging.getLogger(__name__)

def setup_logger(level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('methylimpute')

def optimize_dtypes(df):
    """Optimize data types to reduce memory usage."""
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

def extract_base_cpg_id(cpg_id):
    """Extract the base CpG ID from a potentially complex ID."""
    return str(cpg_id).split('_')[0]

def normalize_chromosome(chr_):
    """Normalize chromosome format (e.g., '1' -> 'chr1', 'X' -> 'chrX')."""
    if pd.isna(chr_) or str(chr_).upper() in ['NA', '0']:
        return "unknown"
    chr_ = str(chr_).upper()
    if chr_.startswith('CHR'):
        return chr_
    return f"chr{chr_}"

def analyze_cpg_patterns(df, id_column='ProbeID'):
    """Analyze CpG ID patterns in the dataframe."""
    illumina_pattern = re.compile(r'(cg|ch|rs)\d+')
    matches = df[id_column].str.match(illumina_pattern).sum()
    logger.info(f"CpGs matching expected pattern: {matches} ({matches/len(df)*100:.2f}%)")
    
    # Sample of non-matching IDs
    non_matches = df[~df[id_column].str.match(illumina_pattern)][id_column].head(5)
    if len(non_matches) > 0:
        logger.info(f"Sample of non-matching IDs: {', '.join(non_matches)}")
    
    return matches, non_matches

def force_gc():
    """Force garbage collection."""
    gc.collect()