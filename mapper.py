"""Module for mapping CpG sites to chromosomes using Illumina manifest files."""

import pandas as pd
import logging
from .utils import (
    extract_base_cpg_id, 
    normalize_chromosome, 
    optimize_dtypes,
    analyze_cpg_patterns,
    setup_logger
)

logger = logging.getLogger(__name__)

class CpGMapper:
    """Maps CpG sites to chromosomes using Illumina manifest files."""
    
    def __init__(self, manifest_files=None, probe_col='IlmnID', chr_col='CHR', 
                 skiprows=7, verbose=True):
        """
        Initialize the CpG mapper.
        
        Parameters:
        -----------
        manifest_files : str or list of str
            Path(s) to Illumina manifest file(s)
        probe_col : str, default='IlmnID'
            Column name for probe IDs in manifest file
        chr_col : str, default='CHR'
            Column name for chromosome in manifest file
        skiprows : int, default=7
            Number of header rows to skip in manifest file
        verbose : bool, default=True
            Whether to log detailed information
        """
        self.manifest_files = manifest_files if isinstance(manifest_files, list) else [manifest_files]
        self.probe_col = probe_col
        self.chr_col = chr_col
        self.skiprows = skiprows
        self.cpg_to_chr = {}
        self.illumina_data = None
        
        if verbose:
            setup_logger()
    
    def load_manifest(self, file_path=None):
        """
        Load Illumina manifest file(s).
        
        Parameters:
        -----------
        file_path : str or list of str, optional
            Path(s) to Illumina manifest file(s)
            If None, uses the files specified during initialization
            
        Returns:
        --------
        self
        """
        if file_path:
            self.manifest_files = file_path if isinstance(file_path, list) else [file_path]
        
        if not self.manifest_files or all(f is None for f in self.manifest_files):
            logger.warning("No manifest files provided.")
            return self
        
        logger.info("Loading Illumina manifest files...")
        dataframes = []
        
        for file in self.manifest_files:
            if file:
                try:
                    df = pd.read_csv(file, skiprows=self.skiprows, low_memory=False)
                    logger.info(f"Loaded manifest file: {file}, found {len(df)} entries")
                    dataframes.append(df)
                except Exception as e:
                    logger.error(f"Error loading manifest file {file}: {e}")
        
        if dataframes:
            self.illumina_data = pd.concat(dataframes, ignore_index=True)
            # Remove duplicates
            self.illumina_data.drop_duplicates(subset=self.probe_col, keep='first', inplace=True)
            logger.info(f"Total CpGs in combined manifest data: {len(self.illumina_data)}")
            
            # Create mapping dictionary
            self._create_mapping()
        else:
            logger.warning("No manifest data loaded.")
        
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