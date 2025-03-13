"""Pytest configuration file for MethylImpute tests."""

import pytest
import os
import pandas as pd
import numpy as np
import tempfile
import shutil


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data."""
    test_dir = tmp_path_factory.mktemp("test_data")
    return test_dir


@pytest.fixture(scope="session")
def small_methylome_file(test_data_dir):
    """Create a small methylome CSV file for testing."""
    # Create a small dataset
    data = pd.DataFrame({
        'ProbeID': [f"cg{i:08d}" for i in range(100)],
        'Sample1': np.random.rand(100),
        'Sample2': np.random.rand(100),
        'Sample3': np.random.rand(100)
    })
    
    # Add some missing values
    mask = np.random.rand(100, 3) < 0.1
    for i, col in enumerate(['Sample1', 'Sample2', 'Sample3']):
        data.loc[mask[:, i], col] = np.nan
    
    # Save to CSV
    file_path = os.path.join(test_data_dir, "small_methylome.csv")
    data.to_csv(file_path, index=False)
    
    return file_path


@pytest.fixture(scope="session")
def mock_manifest_file(test_data_dir):
    """Create a mock Illumina manifest file for testing."""
    # Create header lines (skipped during reading)
    header_lines = ["Header line 1", "Header line 2", "Header line 3", 
                   "Header line 4", "Header line 5", "Header line 6", "Header line 7"]
    
    # Create data
    data = pd.DataFrame({
        'IlmnID': [f"cg{i:08d}" for i in range(100)],
        'CHR': np.random.choice(['1', '2', '3', '4', '5', '6', '7', '8', 'X', 'Y'], size=100),
        'MAPINFO': np.random.randint(1000, 10000, size=100)
    })
    
    # Save to CSV
    file_path = os.path.join(test_data_dir, "mock_manifest.csv")
    
    with open(file_path, 'w') as f:
        # Write header lines
        for line in header_lines:
            f.write(line + "\n")
        
        # Write data
        data.to_csv(f, index=False)
    
    return file_path


@pytest.fixture(scope="session")
def mapped_methylome_file(test_data_dir, small_methylome_file, mock_manifest_file):
    """Create a mapped methylome CSV file for testing."""
    # Read the methylome data
    methylome = pd.read_csv(small_methylome_file)
    
    # Read the manifest data (skipping header rows)
    manifest = pd.read_csv(mock_manifest_file, skiprows=7)
    
    # Create mapping dictionary
    cpg_to_chr = {
        cpg_id: f"chr{chr_}" for cpg_id, chr_ in 
        zip(manifest['IlmnID'], manifest['CHR'])
    }
    
    # Map chromosomes
    methylome['chr'] = methylome['ProbeID'].map(cpg_to_chr)
    
    # Fill missing with 'unknown'
    methylome['chr'] = methylome['chr'].fillna('unknown')
    
    # Save to CSV
    file_path = os.path.join(test_data_dir, "mapped_methylome.csv")
    methylome.to_csv(file_path, index=False)
    
    return file_path


@pytest.fixture(scope="function")
def temp_output_dir():
    """Create a temporary directory for output files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after the test
    shutil.rmtree(temp_dir)