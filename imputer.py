"""Module for imputing missing values in methylation data using MICE."""

import os
import pandas as pd
import numpy as np
import time
import logging
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from concurrent.futures import ProcessPoolExecutor
from .utils import optimize_dtypes, force_gc, setup_logger

logger = logging.getLogger(__name__)

class MethylImputer:
    """Imputes missing values in methylation data using MICE algorithm."""
    
    def __init__(self, max_iter=10, random_state=0, n_nearest_features=10, 
                 n_jobs=1, missing_threshold=0.05, use_bayesian=True, verbose=True):
        """
        Initialize the MethylImputer.
        
        Parameters:
        -----------
        max_iter : int, default=10
            Maximum number of imputation iterations
        random_state : int, default=0
            Random seed for reproducibility
        n_nearest_features : int, default=10
            Number of nearest features to use for imputation
        n_jobs : int, default=1
            Number of parallel jobs for processing
        missing_threshold : float, default=0.05
            Maximum fraction of missing values for a row to be kept (0.05 = 95% non-missing required)
        use_bayesian : bool, default=True
            Whether to use BayesianRidge for imputation (otherwise LinearRegression)
        verbose : bool, default=True
            Whether to log detailed information
        """
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_nearest_features = n_nearest_features
        self.n_jobs = n_jobs
        self.missing_threshold = missing_threshold
        self.use_bayesian = use_bayesian
        
        if verbose:
            setup_logger()
    
    def _create_imputer(self):
        """Create an IterativeImputer instance."""
        if self.use_bayesian:
            estimator = BayesianRidge()
        else:
            from sklearn.linear_model import LinearRegression
            estimator = LinearRegression()
        
        return IterativeImputer(
            estimator=estimator,
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_nearest_features=self.n_nearest_features,
            verbose=0
        )
    
    def impute_batch(self, batch):
        """
        Impute missing values for a batch of the methylation dataset.
        
        Parameters:
        -----------
        batch : pandas.DataFrame
            Batch of methylation data to impute
            
        Returns:
        --------
        pandas.DataFrame
            Imputed data
        """
        start_time = time.time()
        
        # Check if there are any missing values in the batch
        if not batch.isnull().values.any():
            logger.info("No missing values in this batch. Skipping imputation.")
            return batch
        
        # Track the original index and column order
        original_index = batch.index
        original_columns = batch.columns
        
        # Create a MICE imputer
        mice_imputer = self._create_imputer()
        
        # Impute missing values
        batch_imputed = pd.DataFrame(
            mice_imputer.fit_transform(batch),
            columns=original_columns,
            index=original_index
        )
        
        end_time = time.time()
        logger.info(f"Imputation took {end_time - start_time:.2f} seconds")
        
        return batch_imputed
    
    def process_chromosome(self, df, chr_value, chr_col='chr'):
        """
        Process a single chromosome's data.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Methylation data with chromosome column
        chr_value : str
            Chromosome value to process
        chr_col : str, default='chr'
            Column name for chromosome
            
        Returns:
        --------
        pandas.DataFrame
            Imputed data for the chromosome
        """
        logger.info(f"Processing chromosome {chr_value}")
        
        # Filter rows for the current chromosome
        chr_batch = df[df[chr_col] == chr_value].drop(chr_col, axis=1)
        
        # Drop rows with too many missing values
        rows_before = len(chr_batch)
        missing_threshold = len(chr_batch.columns) * (1 - self.missing_threshold)
        chr_batch_dropped = chr_batch.dropna(thresh=missing_threshold)
        rows_after = len(chr_batch_dropped)
        
        logger.info(f"Chromosome {chr_value}: Dropped {rows_before - rows_after} rows out of {rows_before}")
        
        if chr_batch_dropped.empty:
            logger.warning(f"No data left for chromosome {chr_value} after filtering")
            return pd.DataFrame()
        
        # Impute missing values
        chr_batch_imputed = self.impute_batch(chr_batch_dropped)
        
        # Add back the chromosome column
        chr_batch_imputed[chr_col] = chr_value
        
        # Force garbage collection
        force_gc()
        
        return chr_batch_imputed
    
    def impute(self, input_df=None, input_file=None, chr_col='chr',
               output_file=None, save_chunks=False, chunk_dir=None):
        """
        Impute missing values in methylation data.
        
        Parameters:
        -----------
        input_df : pandas.DataFrame, optional
            Methylation data with chromosome column
        input_file : str, optional
            Path to CSV file with methylation data
        chr_col : str, default='chr'
            Column name for chromosome
        output_file : str, optional
            Path to output CSV file
        save_chunks : bool, default=False
            Whether to save imputed data for each chromosome separately
        chunk_dir : str, optional
            Directory to save chromosome chunks (required if save_chunks=True)
            
        Returns:
        --------
        pandas.DataFrame or None
            Imputed data (if output_file is None) or None (if output_file is provided)
        """
        start_time = time.time()
        
        # Load data if provided as a file
        if input_df is None and input_file is not None:
            logger.info(f"Reading input file: {input_file}")
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file not found: {input_file}")
            input_df = pd.read_csv(input_file)
            input_df = optimize_dtypes(input_df)
        
        elif input_df is None and input_file is None:
            raise ValueError("Either input_df or input_file must be provided")
        
        # Check for chromosome column
        if chr_col not in input_df.columns:
            raise ValueError(f"Chromosome column '{chr_col}' not found in the input data")
        
        # Get unique chromosome values
        chromosomes = input_df[chr_col].unique()
        logger.info(f"Found {len(chromosomes)} unique chromosome values")
        
        # Process each chromosome
        imputed_dfs = []
        total_rows_processed = 0
        total_rows_kept = 0
        
        # Create directory for chunks if needed
        if save_chunks and chunk_dir:
            os.makedirs(chunk_dir, exist_ok=True)
        
        # Process chromosomes in parallel if n_jobs > 1
        if self.n_jobs > 1:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {
                    executor.submit(self.process_chromosome, input_df, chr_val, chr_col): chr_val
                    for chr_val in chromosomes
                }
                
                for future in futures:
                    chr_val = futures[future]
                    try:
                        chr_result = future.result()
                        if not chr_result.empty:
                            if save_chunks and chunk_dir:
                                chunk_file = os.path.join(chunk_dir, f"chr_{chr_val}.csv")
                                chr_result.to_csv(chunk_file, index=False)
                                logger.info(f"Saved chromosome {chr_val} data to {chunk_file}")
                            
                            imputed_dfs.append(chr_result)
                            total_rows_processed += len(input_df[input_df[chr_col] == chr_val])
                            total_rows_kept += len(chr_result)
                    except Exception as e:
                        logger.error(f"Error processing chromosome {chr_val}: {e}")
        else:
            # Process chromosomes sequentially
            for chr_val in chromosomes:
                try:
                    chr_result = self.process_chromosome(input_df, chr_val, chr_col)
                    if not chr_result.empty:
                        if save_chunks and chunk_dir:
                            chunk_file = os.path.join(chunk_dir, f"chr_{chr_val}.csv")
                            chr_result.to_csv(chunk_file, index=False)
                            logger.info(f"Saved chromosome {chr_val} data to {chunk_file}")
                        
                        imputed_dfs.append(chr_result)
                        total_rows_processed += len(input_df[input_df[chr_col] == chr_val])
                        total_rows_kept += len(chr_result)
                except Exception as e:
                    logger.error(f"Error processing chromosome {chr_val}: {e}")
        
        # Combine results
        if imputed_dfs:
            logger.info("Combining imputed data from all chromosomes")
            full_imputed = pd.concat(imputed_dfs, ignore_index=True)
            logger.info(f"Combined data has {len(full_imputed)} rows")
            
            # Save combined results if output file is provided
            if output_file:
                logger.info(f"Saving imputed data to {output_file}")
                full_imputed.to_csv(output_file, index=False)
        else:
            logger.warning("No data was successfully imputed")
            full_imputed = pd.DataFrame()
        
        end_time = time.time()
        logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")
        logger.info(f"Total rows processed: {total_rows_processed}")
        logger.info(f"Total rows kept: {total_rows_kept}")
        
        if output_file:
            logger.info(f"Imputation complete. Imputed data saved to '{output_file}'")
            return None
        else:
            return full_imputed