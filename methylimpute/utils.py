"""Utility functions for the MethylImpute package."""

import os
import re
import logging
import pandas as pd
import requests
import zipfile
import io

logger = logging.getLogger(__name__)

def setup_logger(level=logging.INFO):
    """
    Set up the logger for the package.
    
    Parameters:
    -----------
    level : int, default=logging.INFO
        Logging level
        
    Returns:
    --------
    logging.Logger
        Configured logger
    """
    logger = logging.getLogger("methylimpute")
    logger.setLevel(level)
    
    # Create handler if not already set up
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def extract_base_cpg_id(cpg_id):
    """
    Extract the base CpG ID without additional suffixes.
    
    Parameters:
    -----------
    cpg_id : str
        CpG ID, possibly with suffixes
        
    Returns:
    --------
    str
        Base CpG ID
    """
    return str(cpg_id).split('_')[0]

def normalize_chromosome(chr_):
    """
    Normalize chromosome format.
    
    Parameters:
    -----------
    chr_ : str
        Chromosome identifier
        
    Returns:
    --------
    str
        Normalized chromosome identifier
    """
    if pd.isna(chr_) or str(chr_).upper() in ['NA', '0']:
        return "unknown"
    chr_ = str(chr_).upper()
    if chr_.startswith('CHR'):
        return chr_
    return f"chr{chr_}"

def optimize_dtypes(df):
    """
    Optimize data types in DataFrame to reduce memory usage.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to optimize
        
    Returns:
    --------
    pandas.DataFrame
        Optimized DataFrame
    """
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

def analyze_cpg_patterns(df, probe_id_col='ProbeID'):
    """
    Analyze CpG ID patterns in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with CpG IDs
    probe_id_col : str, default='ProbeID'
        Column name for CpG IDs
        
    Returns:
    --------
    None
    """
    illumina_pattern = re.compile(r'(cg|ch|rs)\d+')
    custom_pattern = re.compile(r'([A-Za-z]+)(\d+)')
    
    illumina_matches = df[probe_id_col].str.match(illumina_pattern).sum()
    
    logger.info(f"CpGs matching expected Illumina pattern: {illumina_matches} ({illumina_matches/len(df)*100:.2f}%)")
    
    # Sample of non-matching IDs
    non_matching = df[~df[probe_id_col].str.match(illumina_pattern)]
    if len(non_matching) > 0:
        logger.info("Sample of non-matching CpG IDs and their patterns:")
        for cpg in non_matching[probe_id_col].head():
            match = custom_pattern.match(cpg)
            if match:
                logger.info(f"{cpg}: Prefix={match.group(1)}, Number={match.group(2)}")
            else:
                logger.info(f"{cpg}: No pattern match")

def get_project_dir(input_file=None):
    """
    Get the project directory based on the input file.
    
    Parameters:
    -----------
    input_file : str, optional
        Path to input file
        
    Returns:
    --------
    str
        Project directory
    """
    if input_file:
        # Use the directory of the input file as the project directory
        return os.path.dirname(os.path.abspath(input_file))
    else:
        # If no input file is provided, use the current working directory
        return os.getcwd()

def download_and_extract_manifest(output_dir=None, input_file=None, force_download=False):
    """
    Download the Illumina EPIC v1.0 B5 manifest file and extract it.
    
    Parameters:
    -----------
    output_dir : str, optional
        Directory to save the extracted manifest
        If None, uses the project directory + /manifests
    input_file : str, optional
        Path to input file, used to determine project directory
    force_download : bool, default=False
        Whether to force download even if the file already exists
        
    Returns:
    --------
    str or None
        Path to the extracted manifest file if successful, None otherwise
    """
    # Set default download directory to project/manifests if not provided
    if output_dir is None:
        project_dir = get_project_dir(input_file)
        output_dir = os.path.join(project_dir, 'manifests')
        logger.info(f"Using project-related manifest directory: {output_dir}")
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # URL for EPIC v1.0 B5 manifest
    url = "https://webdata.illumina.com/downloads/productfiles/methylationEPIC/infinium-methylationepic-v-1-0-b5-manifest-file-csv.zip"
    
    # Expected filename after extraction
    expected_csv = "infinium-methylationepic-v-1-0-b5-manifest-file.csv"
    output_path = os.path.join(output_dir, expected_csv)
    
    # Check if file already exists
    if os.path.exists(output_path) and not force_download:
        logger.info(f"Manifest file already exists at {output_path}")
        return output_path
    
    try:
        # Start download
        logger.info(f"Downloading EPIC v1.0 B5 manifest from Illumina website to {output_dir}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Download zip file to memory
        zip_data = io.BytesIO()
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                zip_data.write(chunk)
        
        # Reset pointer to beginning of file
        zip_data.seek(0)
        
        # Extract manifest file
        logger.info("Extracting manifest file from zip archive...")
        with zipfile.ZipFile(zip_data) as zip_ref:
            file_list = zip_ref.namelist()
            logger.info(f"Zip file contains {len(file_list)} files")
            
            # Find the manifest CSV file
            csv_file = None
            for file_name in file_list:
                if file_name.lower().endswith('.csv') and "manifest" in file_name.lower():
                    csv_file = file_name
                    break
            
            if csv_file:
                # Extract the file
                logger.info(f"Extracting {csv_file} to {output_path}")
                with zip_ref.open(csv_file) as source, open(output_path, 'wb') as target:
                    target.write(source.read())
                
                logger.info(f"Manifest file extracted to {output_path}")
                return output_path
            else:
                logger.error("No manifest CSV file found in the zip archive")
                return None
        
    except Exception as e:
        logger.error(f"Error downloading or extracting manifest file: {e}")
        return None