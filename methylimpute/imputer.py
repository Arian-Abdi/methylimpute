"""Module for imputing missing methylation values using MICE."""

import os
import pandas as pd
import numpy as np
import time
import gc
import logging
from multiprocessing import Pool, cpu_count
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from .utils import optimize_dtypes, setup_logger

logger = logging.getLogger(__name__)

class MethylImputer:
    """Imputes missing methylation values using MICE algorithm."""
    
    def __init__(self, max_iter=10, n_nearest_features=10, missing_threshold=0.05, 
                 n_jobs=1, random_state=0, verbose=True):
        """
        Initialize the methylation imputer.
        
        Parameters:
        -----------
        max_iter : int, default=10
            Maximum number of imputation iterations
        n_nearest_features : int, default=10
            Number of nearest features to use for imputation
        missing_threshold : float, default=0.05
            Maximum fraction of missing values for a row to be kept
            (1 - missing_threshold = min fraction of non-missing values required)
        n_jobs : int, default=1
            Number of parallel jobs for processing
        random_state : int, default=0
            Random seed for reproducibility
        verbose : bool, default=True
            Whether to log detailed information
        """
        self.max_iter = max_iter
        self.n_nearest_features = n_nearest_features
        self.missing_threshold = missing_threshold
        self.n_jobs = min(n_jobs, cpu_count()) if n_jobs > 0 else 1
        self.random_state = random_state
        
        if verbose:
            setup_logger()
    
    def prepare_data_for_imputation(self, df):
        """
        Prepare data for imputation by excluding non-numeric columns.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with only numeric columns
        list
            List of excluded column names
        """
        # Identify non-numeric columns
        non_numeric_cols = []
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric_cols.append(col)
        
        # Store non-numeric columns separately
        metadata = {col: df[col].copy() for col in non_numeric_cols}
        
        # Remove non-numeric columns
        numeric_df = df.drop(columns=non_numeric_cols)
        
        logger.info(f"Excluded {len(non_numeric_cols)} non-numeric columns: {non_numeric_cols}")
        
        return numeric_df, metadata
    
    def restore_metadata(self, imputed_df, metadata):
        """
        Restore metadata columns to imputed DataFrame.
        
        Parameters:
        -----------
        imputed_df : pandas.DataFrame
            Imputed DataFrame with only numeric columns
        metadata : dict
            Dictionary with non-numeric columns
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with both numeric and non-numeric columns
        """
        # Make a copy to avoid modifying the original
        result_df = imputed_df.copy()
        
        # Add back non-numeric columns
        for col, data in metadata.items():
            result_df[col] = data
        
        return result_df
    
    def impute_batch(self, batch):
        """
        Impute missing values for a batch of methylome data using MICE.
        
        Parameters:
        -----------
        batch : pandas.DataFrame
            Batch of methylome data
            
        Returns:
        --------
        pandas.DataFrame
            Imputed batch
        """
        start_time = time.time()
        
        # Check if there are any missing values in the batch
        if not batch.isnull().values.any():
            logger.info("No missing values in this batch. Skipping imputation.")
            return batch
        
        # Prepare data for imputation
        numeric_batch, metadata = self.prepare_data_for_imputation(batch)
        
        if numeric_batch.empty:
            logger.warning("No numeric columns to impute.")
            return batch
        
        # Create a MICE imputer
        mice_imputer = IterativeImputer(
            estimator=BayesianRidge(), 
            max_iter=self.max_iter, 
            random_state=self.random_state,
            n_nearest_features=self.n_nearest_features
        )

        # Impute missing values
        imputed_data = mice_imputer.fit_transform(numeric_batch)
        
        # Convert back to DataFrame with original column names
        batch_imputed = pd.DataFrame(
            imputed_data, 
            columns=numeric_batch.columns,
            index=numeric_batch.index
        )
        
        # Restore metadata
        batch_imputed = self.restore_metadata(batch_imputed, metadata)

        end_time = time.time()
        logger.info(f"Imputation of batch took {end_time - start_time:.2f} seconds")

        return batch_imputed
    
    def process_chromosome(self, chr_batch, chr_value, output_file=None, chunk_dir=None):
        """
        Process a chromosome batch: filter and impute.
        
        Parameters:
        -----------
        chr_batch : pandas.DataFrame
            Methylome data for a specific chromosome
        chr_value : str
            Chromosome identifier
        output_file : str, optional
            Path to output CSV file
        chunk_dir : str, optional
            Directory to save chromosome chunks
            
        Returns:
        --------
        pandas.DataFrame
            Imputed chromosome batch
        """
        logger.info(f"Processing chromosome {chr_value}")
        
        # Save the chromosome column before dropping it
        chr_col_name = None
        for col in chr_batch.columns:
            if chr_batch[col].eq(chr_value).all():
                chr_col_name = col
                break
        
        # Drop rows with too many missing values
        # Only consider numeric columns when calculating missing values
        numeric_cols = []
        for col in chr_batch.columns:
            if pd.api.types.is_numeric_dtype(chr_batch[col]) and col != chr_col_name:
                numeric_cols.append(col)
        
        if numeric_cols:
            required_non_na = len(numeric_cols) * (1 - self.missing_threshold)
            rows_before = len(chr_batch)
            chr_batch_filtered = chr_batch.dropna(subset=numeric_cols, thresh=required_non_na)
            rows_after = len(chr_batch_filtered)
            
            logger.info(f"Chromosome {chr_value}: Dropped {rows_before - rows_after} rows out of {rows_before}")
        else:
            logger.warning(f"No numeric columns found for chromosome {chr_value}")
            chr_batch_filtered = chr_batch
        
        # Impute missing values if there are any
        if not chr_batch_filtered.empty:
            chr_batch_imputed = self.impute_batch(chr_batch_filtered)
            
            # Save the imputed chunk if requested
            if chunk_dir and output_file:
                os.makedirs(chunk_dir, exist_ok=True)
                chunk_file = os.path.join(chunk_dir, f"chr_{chr_value}.csv")
                chr_batch_imputed.to_csv(chunk_file, index=True)
                logger.info(f"Saved imputed data for chromosome {chr_value} to {chunk_file}")
            
            return chr_batch_imputed
        else:
            logger.warning(f"No rows left for chromosome {chr_value} after filtering")
            return pd.DataFrame()
    
    def _worker_process_chromosome(self, args):
        """Worker function for parallel processing."""
        chr_batch, chr_value, output_file, chunk_dir = args
        return self.process_chromosome(chr_batch, chr_value, output_file, chunk_dir)
    
    def impute(self, input_file=None, input_df=None, chr_col='chr', output_file=None, 
               save_chunks=False, chunk_dir=None):
        """
        Impute missing values in methylome data by chromosome.
        
        Parameters:
        -----------
        input_file : str, optional
            Path to input CSV file with chromosome mapping
        input_df : pandas.DataFrame, optional
            DataFrame with methylome data and chromosome mapping
            (either input_file or input_df must be provided)
        chr_col : str, default='chr'
            Column name for chromosome
        output_file : str, optional
            Path to output CSV file with imputed values
        save_chunks : bool, default=False
            Whether to save imputed data for each chromosome separately
        chunk_dir : str, optional
            Directory to save chromosome chunks (required if save_chunks=True)
            
        Returns:
        --------
        pandas.DataFrame or None
            Imputed methylome data if output_file is None, otherwise None
        """
        start_time = time.time()
        
        # Validate input parameters
        if input_file is None and input_df is None:
            raise ValueError("Either input_file or input_df must be provided")
        
        if save_chunks and not chunk_dir:
            raise ValueError("chunk_dir must be provided if save_chunks=True")
        
        # Read input data if needed
        if input_df is None:
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file not found: {input_file}")
            
            logger.info(f"Reading methylome data from {input_file}")
            df = pd.read_csv(input_file)
            df = optimize_dtypes(df)
        else:
            df = input_df.copy()
            df = optimize_dtypes(df)
        
        # Check that chromosome column exists
        if chr_col not in df.columns:
            raise ValueError(f"Chromosome column '{chr_col}' not found in input data")
        
        # Get unique chromosome values
        chromosomes = df[chr_col].unique()
        logger.info(f"Found {len(chromosomes)} unique chromosomes in input data")
        
        # Process each chromosome
        all_imputed = []
        
        if self.n_jobs > 1 and len(chromosomes) > 1:
            # Parallel processing for multiple chromosomes
            logger.info(f"Using {self.n_jobs} parallel jobs for processing")
            
            # Prepare arguments for parallel processing
            args_list = []
            for chr_value in chromosomes:
                # Filter rows for the current chromosome
                chr_batch = df[df[chr_col] == chr_value]
                args_list.append((chr_batch, chr_value, output_file if save_chunks else None, 
                                 chunk_dir if save_chunks else None))
            
            # Process chromosomes in parallel
            with Pool(self.n_jobs) as pool:
                results = pool.map(self._worker_process_chromosome, args_list)
            
            # Collect results
            for result in results:
                if not result.empty:
                    all_imputed.append(result)
        else:
            # Sequential processing
            for chr_value in chromosomes:
                # Filter rows for the current chromosome
                chr_batch = df[df[chr_col] == chr_value]
                
                # Process chromosome batch
                imputed_batch = self.process_chromosome(
                    chr_batch, chr_value, 
                    output_file if save_chunks else None,
                    chunk_dir if save_chunks else None
                )
                
                if not imputed_batch.empty:
                    all_imputed.append(imputed_batch)
                
                # Force garbage collection
                gc.collect()
        
        # Combine all imputed batches
        if all_imputed:
            imputed_data = pd.concat(all_imputed, axis=0)
            logger.info(f"Combined imputed data: {imputed_data.shape[0]} rows, {imputed_data.shape[1]} columns")
            
            # Write combined results if output file is specified
            if output_file:
                logger.info(f"Saving imputed data to {output_file}")
                imputed_data.to_csv(output_file, index=False)
                logger.info(f"Saved {len(imputed_data)} rows to {output_file}")
                
                # Return None if output file is specified
                return None
            
            # Return imputed data if no output file is specified
            return imputed_data
        else:
            logger.warning("No imputed data to save")
            return pd.DataFrame()