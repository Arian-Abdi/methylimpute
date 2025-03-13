"""Integration tests for MethylImpute package."""

import os
import pytest
import pandas as pd
import numpy as np
from methylimpute.mapper import CpGMapper
from methylimpute.imputer import MethylImputer


def test_map_impute_integration(small_methylome_file, mock_manifest_file, temp_output_dir):
    """Test integration of mapping and imputation steps."""
    # Define paths
    mapped_output = os.path.join(temp_output_dir, "mapped.csv")
    imputed_output = os.path.join(temp_output_dir, "imputed.csv")
    
    # Step 1: Mapping
    mapper = CpGMapper(manifest_file=mock_manifest_file)
    mapper.load_manifest()
    
    # Read methylome data
    methylome = pd.read_csv(small_methylome_file)
    
    # Map chromosomes
    mapped_data = mapper.map_methylome(methylome)
    
    # Save mapped data
    mapper.save_mapped_data(mapped_data, mapped_output)
    
    # Verify mapping results
    assert os.path.exists(mapped_output)
    assert 'chr' in mapped_data.columns
    
    # Check that all rows have a chromosome assignment (even if 'unknown')
    assert mapped_data['chr'].isna().sum() == 0
    
    # Step 2: Imputation
    imputer = MethylImputer(max_iter=5, n_jobs=1)
    
    # Run imputation
    imputer.impute(
        input_file=mapped_output,
        output_file=imputed_output
    )
    
    # Verify imputation results
    assert os.path.exists(imputed_output)
    
    # Read imputed data
    imputed_data = pd.read_csv(imputed_output)
    
    # Check that imputed data has same dimensions
    assert imputed_data.shape[0] <= mapped_data.shape[0]  # May be less due to filtering
    
    # Check that numeric columns were imputed (no missing values)
    numeric_cols = imputed_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        assert imputed_data[col].isna().sum() == 0


def test_full_pipeline_integration(small_methylome_file, mock_manifest_file, temp_output_dir):
    """Test integration of the full pipeline."""
    # Define paths
    mapped_output = os.path.join(temp_output_dir, "mapped_pipeline.csv")
    imputed_output = os.path.join(temp_output_dir, "imputed_pipeline.csv")
    chunk_dir = os.path.join(temp_output_dir, "chunks")
    
    # Step 1: Mapping
    mapper = CpGMapper(manifest_file=mock_manifest_file)
    mapper.load_manifest()
    
    # Read methylome data
    methylome = pd.read_csv(small_methylome_file)
    
    # Map chromosomes
    mapped_data = mapper.map_methylome(methylome)
    
    # Save mapped data
    mapper.save_mapped_data(mapped_data, mapped_output)
    
    # Step 2: Imputation with save_chunks=True
    imputer = MethylImputer(max_iter=5, n_jobs=1)
    
    # Run imputation
    imputer.impute(
        input_df=mapped_data,
        chr_col='chr',
        output_file=imputed_output,
        save_chunks=True,
        chunk_dir=chunk_dir
    )
    
    # Verify output files
    assert os.path.exists(mapped_output)
    assert os.path.exists(imputed_output)
    assert os.path.exists(chunk_dir)
    
    # Check that chunk files were created for each chromosome
    chromosomes = mapped_data['chr'].unique()
    for chrom in chromosomes:
        chunk_file = os.path.join(chunk_dir, f"chr_{chrom}.csv")
        if len(mapped_data[mapped_data['chr'] == chrom]) > 0:  # Only if there's data for this chromosome
            assert os.path.exists(chunk_file)
    
    # Read imputed data
    imputed_data = pd.read_csv(imputed_output)
    
    # Check that imputed data has same columns
    assert 'ProbeID' in imputed_data.columns
    assert 'chr' in imputed_data.columns
    
    # Check that numeric columns were imputed (no missing values)
    numeric_cols = imputed_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        assert imputed_data[col].isna().sum() == 0