"""Tests for the CpG mapper module."""

import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from methylimpute.mapper import CpGMapper


@pytest.fixture
def sample_illumina_data():
    """Create a sample Illumina manifest dataframe for testing."""
    return pd.DataFrame({
        'IlmnID': ['cg00000029', 'cg00000108', 'cg00000165', 'cg00000236'],
        'CHR': ['1', '3', '1', '7'],
        'MAPINFO': [1000, 2000, 3000, 4000]
    })


@pytest.fixture
def sample_methylome_data():
    """Create a sample methylome dataframe for testing."""
    return pd.DataFrame({
        'ProbeID': ['cg00000029', 'cg00000108', 'cg00000165', 'cg00000236'],
        'Sample1': [0.8, 0.7, 0.6, 0.5],
        'Sample2': [0.7, 0.6, 0.5, 0.4]
    })


@pytest.fixture
def mock_manifest_file(tmp_path, sample_illumina_data):
    """Create a mock manifest file."""
    # Create header lines (skipped during reading)
    header_lines = ["Header line 1", "Header line 2", "Header line 3", 
                   "Header line 4", "Header line 5", "Header line 6", "Header line 7"]
    
    # Create the manifest file
    manifest_file = tmp_path / "mock_manifest.csv"
    
    # Write header lines
    with open(manifest_file, 'w') as f:
        for line in header_lines:
            f.write(line + "\n")
    
    # Append the data
    sample_illumina_data.to_csv(manifest_file, mode='a', index=False)
    
    return manifest_file


def test_mapper_initialization():
    """Test the initialization of CpGMapper."""
    mapper = CpGMapper(manifest_file="test.csv")
    assert mapper.manifest_file == "test.csv"
    assert mapper.probe_col == 'IlmnID'
    assert mapper.chr_col == 'CHR'
    assert mapper.skiprows == 7
    assert mapper.cpg_to_chr == {}
    assert mapper.illumina_data is None


def test_load_manifest(mock_manifest_file):
    """Test loading a manifest file."""
    mapper = CpGMapper(manifest_file=str(mock_manifest_file))
    mapper.load_manifest()
    
    assert mapper.illumina_data is not None
    assert len(mapper.illumina_data) == 4
    assert 'IlmnID' in mapper.illumina_data.columns
    assert 'CHR' in mapper.illumina_data.columns
    assert len(mapper.cpg_to_chr) == 4


def test_map_methylome(mock_manifest_file, sample_methylome_data):
    """Test mapping chromosomes to methylome data."""
    mapper = CpGMapper(manifest_file=str(mock_manifest_file))
    mapper.load_manifest()
    
    mapped_data = mapper.map_methylome(sample_methylome_data)
    
    assert 'chr' in mapped_data.columns
    assert mapped_data.loc[0, 'chr'] == 'chr1'
    assert mapped_data.loc[1, 'chr'] == 'chr3'
    assert mapped_data.loc[2, 'chr'] == 'chr1'
    assert mapped_data.loc[3, 'chr'] == 'chr7'


def test_mapping_with_missing_probes(mock_manifest_file):
    """Test mapping with probes that aren't in the manifest."""
    mapper = CpGMapper(manifest_file=str(mock_manifest_file))
    mapper.load_manifest()
    
    # Create methylome data with probes not in the manifest
    methylome_data = pd.DataFrame({
        'ProbeID': ['cg00000029', 'cgNOTINMANIFEST', 'cg00000165', 'cgALSONOTINMANIFEST'],
        'Sample1': [0.8, 0.7, 0.6, 0.5],
        'Sample2': [0.7, 0.6, 0.5, 0.4]
    })
    
    mapped_data = mapper.map_methylome(methylome_data)
    
    assert 'chr' in mapped_data.columns
    assert mapped_data.loc[0, 'chr'] == 'chr1'
    assert mapped_data.loc[1, 'chr'] == 'unknown'  # Missing probe
    assert mapped_data.loc[2, 'chr'] == 'chr1'
    assert mapped_data.loc[3, 'chr'] == 'unknown'  # Missing probe


def test_save_mapped_data(mock_manifest_file, sample_methylome_data, tmp_path):
    """Test saving mapped data to a CSV file."""
    mapper = CpGMapper(manifest_file=str(mock_manifest_file))
    mapper.load_manifest()
    
    mapped_data = mapper.map_methylome(sample_methylome_data)
    
    output_file = tmp_path / "mapped_data.csv"
    mapper.save_mapped_data(mapped_data, str(output_file))
    
    assert os.path.exists(output_file)
    
    # Read back the saved data and verify
    saved_data = pd.read_csv(output_file)
    assert 'chr' in saved_data.columns
    assert len(saved_data) == len(sample_methylome_data)


@patch('methylimpute.mapper.download_and_extract_manifest')
def test_auto_download_manifest(mock_download, sample_methylome_data):
    """Test automatic download of manifest file."""
    # Set up the mock
    mock_manifest_path = "/tmp/mock_manifest.csv"
    mock_download.return_value = mock_manifest_path
    
    # Create a mapper with mock data
    with patch('pandas.read_csv') as mock_read_csv:
        mock_read_csv.return_value = pd.DataFrame({
            'IlmnID': ['cg00000029', 'cg00000108', 'cg00000165', 'cg00000236'],
            'CHR': ['1', '3', '1', '7']
        })
        
        # Create mapper without specifying manifest file
        mapper = CpGMapper(manifest_file=None, download_manifest=True)
        mapper.load_manifest()
        
        # Verify download was attempted
        mock_download.assert_called_once()
        assert mapper.manifest_file == mock_manifest_path


def test_no_manifest():
    """Test behavior when no manifest is provided and download is disabled."""
    mapper = CpGMapper(manifest_file=None, download_manifest=False)
    mapper.load_manifest()
    
    assert mapper.illumina_data is None
    assert not mapper.cpg_to_chr