"""Module for mapping CpG sites to chromosomes using Illumina manifest files."""

import pandas as pd
import logging
import os
from .utils import (
    extract_base_cpg_id, 
    normalize_chromosome, 
    optimize_dtypes,
    analyze_cpg_patterns,
    setup_logger,
    download_and_extract_manifest
)

logger = logging.getLogger(__name__)

class CpGMapper:
    """Maps CpG sites to chromosomes using Illumina EPIC v1.0 manifest files."""
    
    def __init__(self, manifest_file=None, probe_col='IlmnID', download_manifest=True, chr_col='CHR', 
                 skiprows=7, manifest_dir=None, input_file=None, verbose=True):
        """
        Initialize the CpG mapper.
        
        Parameters:
        -----------
        manifest_file : str
            Path to Illumina EPIC v1.0 manifest file
        probe_col : str, default='IlmnID'
            Column name for probe IDs in manifest file
        download_manifest : bool, default=True
            Whether to download the manifest file if manifest_file is None
        chr_col : str, default='CHR'
            Column name for chromosome in manifest file
        skiprows : int, default=7
            Number of header rows to skip in manifest file
        manifest_dir : str, optional
            Directory to save manifest files (overrides project-related directory)
        input_file : str, optional
            Path to input file, used to determine project directory for manifests
        verbose : bool, default=True
            Whether to log detailed information
        """
        self.manifest_file = manifest_file
        self.probe_col = probe_col
        self.download_manifest = download_manifest
        self.chr_col = chr_col
        self.skiprows = skiprows
        self.manifest_dir = manifest_dir
        self.input_file = input_file
        self.cpg_to_chr = {}
        self.illumina_data = None
        
        if verbose:
            setup_logger()
    
    def load_manifest(self, file_path=None):
        """
        Load Illumina EPIC v1.0 manifest file.
        
        Parameters:
        -----------
        file_path : str, optional
            Path to Illumina manifest file
            If None, uses the file specified during initialization
            
        Returns:
        --------
        self
        """
        if file_path:
            self.manifest_file = file_path

        # If no manifest file is specified, download it
        if not self.manifest_file and self.download_manifest:
            self.manifest_file = download_and_extract_manifest(
                output_dir=self.manifest_dir,
                input_file=self.input_file
            )
            if not self.manifest_file:
                logger.error("Failed to download manifest file")
                return self

        if not self.manifest_file:
            logger.warning("No manifest file provided and download_manifest=False.")
            return self
        
        if not os.path.exists(self.manifest_file):
            logger.error(f"Manifest file not found: {self.manifest_file}")
            return self
        
        logger.info(f"Loading manifest file: {self.manifest_file}")
        
        try:
            self.illumina_data = pd.read_csv(self.manifest_file, skiprows=self.skiprows, low_memory=False)
            logger.info(f"Loaded manifest with {len(self.illumina_data)} entries")
            
            # Remove duplicates if any
            if self.illumina_data.duplicated(subset=self.probe_col).any():
                dup_count = self.illumina_data.duplicated(subset=self.probe_col).sum()
                logger.info(f"Removing {dup_count} duplicate probe entries")
                self.illumina_data.drop_duplicates(subset=self.probe_col, keep='first', inplace=True)
            
            # Create mapping dictionary
            self._create_mapping()
        except Exception as e:
            logger.error(f"Error loading manifest file: {e}")
        
        return self
    
    def _create_mapping(self):
        """Create mapping from CpG IDs to chromosomes."""
        if self.illumina_data is None:
            logger.warning("No manifest data loaded. Cannot create mapping.")
            return
        
        logger.info("Creating CpG to chromosome mapping...")
        self.cpg_to_chr = {
            extract_base_cpg_id(cpg_id): normalize_chromosome(chr_)
            for cpg_id, chr_ in zip(
                self.illumina_data[self.probe_col], 
                self.illumina_data[self.chr_col]
            )
        }
        logger.info(f"Created mapping for {len(self.cpg_to_chr)} unique CpG sites")
    
    def map_methylome(self, methylome_df, probe_id_col='ProbeID', output_col='chr'):
        """
        Map chromosomes to methylome data.
        
        Parameters:
        -----------
        methylome_df : pandas.DataFrame
            Methylome data with CpG IDs
        probe_id_col : str, default='ProbeID'
            Column name for probe IDs in methylome data
        output_col : str, default='chr'
            Column name for output chromosome mapping
            
        Returns:
        --------
        pandas.DataFrame
            Methylome data with chromosome mapping
        """
        if not self.cpg_to_chr:
            logger.warning("No CpG to chromosome mapping exists. Run load_manifest() first.")
            return methylome_df
        
        logger.info(f"Mapping chromosomes to methylome data with {len(methylome_df)} rows...")
        
        # Check and standardize column names
        if probe_id_col not in methylome_df.columns:
            if 'probe_id' in methylome_df.columns and probe_id_col == 'ProbeID':
                logger.info("Renaming 'probe_id' column to 'ProbeID'")
                methylome_df = methylome_df.rename(columns={'probe_id': 'ProbeID'})
                probe_id_col = 'ProbeID'
            else:
                raise ValueError(f"Column '{probe_id_col}' not found in methylome data")
        
        # Apply mapping
        methylome_df[output_col] = methylome_df[probe_id_col].map(self.cpg_to_chr)
        
        # Analyze mapping success
        unmapped = methylome_df[methylome_df[output_col].isna()]
        logger.info(f"Total CpGs in methylome data: {len(methylome_df)}")
        logger.info(f"Unmapped CpGs in methylome: {len(unmapped)} ({len(unmapped)/len(methylome_df)*100:.2f}%)")
        
        # Analyze CpG ID patterns
        analyze_cpg_patterns(methylome_df, probe_id_col)
        
        # Fill NA values with "unknown"
        methylome_df[output_col] = methylome_df[output_col].fillna('unknown')
        
        # Analyze chromosome distribution
        chr_distribution = methylome_df[output_col].value_counts()
        logger.info("Chromosome distribution:")
        logger.info(chr_distribution)
        
        return methylome_df
    
    def save_mapped_data(self, methylome_df, output_file):
        """
        Save methylome data with chromosome mapping to a CSV file.
        
        Parameters:
        -----------
        methylome_df : pandas.DataFrame
            Methylome data with chromosome mapping
        output_file : str
            Path to output CSV file
            
        Returns:
        --------
        None
        """
        logger.info(f"Saving mapped data to {output_file}...")
        methylome_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(methylome_df)} rows to {output_file}")