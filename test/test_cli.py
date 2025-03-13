"""Tests for the command-line interface."""

import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
from io import StringIO

# Import the CLI module and functions
from methylimpute.cli import (
    parse_args,
    map_command,
    impute_command,
    pipeline_command,
    main
)


@patch('argparse.ArgumentParser.parse_args')
def test_parse_args_map(mock_parse_args):
    """Test parsing arguments for the map command."""
    # Set up the mock to return map command args
    mock_args = MagicMock()
    mock_args.command = 'map'
    mock_args.manifest = 'manifest.csv'
    mock_args.input = 'input.csv'
    mock_args.output = 'output.csv'
    mock_args.probe_col = 'ProbeID'
    mock_args.chr_col = 'chr'
    mock_args.skiprows = 7
    mock_args.no_download = False
    mock_args.debug = False
    
    mock_parse_args.return_value = mock_args
    
    # Call parse_args function
    args = parse_args()
    
    # Check the returned args
    assert args.command == 'map'
    assert args.manifest == 'manifest.csv'
    assert args.input == 'input.csv'
    assert args.output == 'output.csv'
    assert args.probe_col == 'ProbeID'
    assert args.chr_col == 'chr'
    assert args.skiprows == 7
    assert args.no_download is False
    assert args.debug is False


@patch('argparse.ArgumentParser.parse_args')
def test_parse_args_impute(mock_parse_args):
    """Test parsing arguments for the impute command."""
    # Set up the mock to return impute command args
    mock_args = MagicMock()
    mock_args.command = 'impute'
    mock_args.input = 'input.csv'
    mock_args.output = 'output.csv'
    mock_args.chr_col = 'chr'
    mock_args.max_iter = 10
    mock_args.n_nearest = 10
    mock_args.threshold = 0.05
    mock_args.save_chunks = False
    mock_args.chunk_dir = None
    mock_args.jobs = 1
    mock_args.debug = False
    
    mock_parse_args.return_value = mock_args
    
    # Call parse_args function
    args = parse_args()
    
    # Check the returned args
    assert args.command == 'impute'
    assert args.input == 'input.csv'
    assert args.output == 'output.csv'
    assert args.chr_col == 'chr'
    assert args.max_iter == 10
    assert args.n_nearest == 10
    assert args.threshold == 0.05
    assert args.save_chunks is False
    assert args.chunk_dir is None
    assert args.jobs == 1
    assert args.debug is False


@patch('argparse.ArgumentParser.parse_args')
def test_parse_args_pipeline(mock_parse_args):
    """Test parsing arguments for the pipeline command."""
    # Set up the mock to return pipeline command args
    mock_args = MagicMock()
    mock_args.command = 'pipeline'
    mock_args.manifest = 'manifest.csv'
    mock_args.input = 'input.csv'
    mock_args.output = 'output.csv'
    mock_args.mapped_output = 'mapped.csv'
    mock_args.probe_col = 'ProbeID'
    mock_args.chr_col = 'chr'
    mock_args.skiprows = 7
    mock_args.max_iter = 10
    mock_args.n_nearest = 10
    mock_args.threshold = 0.05
    mock_args.save_chunks = False
    mock_args.chunk_dir = None
    mock_args.jobs = 1
    mock_args.no_download = False
    mock_args.debug = False
    
    mock_parse_args.return_value = mock_args
    
    # Call parse_args function
    args = parse_args()
    
    # Check the returned args
    assert args.command == 'pipeline'
    assert args.manifest == 'manifest.csv'
    assert args.input == 'input.csv'
    assert args.output == 'output.csv'
    assert args.mapped_output == 'mapped.csv'
    assert args.probe_col == 'ProbeID'
    assert args.chr_col == 'chr'
    assert args.skiprows == 7
    assert args.max_iter == 10
    assert args.n_nearest == 10
    assert args.threshold == 0.05
    assert args.save_chunks is False
    assert args.chunk_dir is None
    assert args.jobs == 1
    assert args.no_download is False
    assert args.debug is False


@patch('methylimpute.cli.setup_logger')
@patch('methylimpute.cli.CpGMapper')
@patch('pandas.read_csv')
def test_map_command(mock_read_csv, mock_mapper_class, mock_setup_logger):
    """Test the map command."""
    # Set up mocks
    mock_args = MagicMock()
    mock_args.manifest = 'manifest.csv'
    mock_args.input = 'input.csv'
    mock_args.output = 'output.csv'
    mock_args.probe_col = 'ProbeID'
    mock_args.chr_col = 'chr'
    mock_args.skiprows = 7
    mock_args.no_download = False
    mock_args.debug = False
    mock_args.manifest_dir = None
    
    # Mock mapper instance
    mock_mapper = MagicMock()
    mock_mapper_class.return_value = mock_mapper
    
    # Mock DataFrame
    mock_df = pd.DataFrame({
        'ProbeID': ['cg00000029', 'cg00000108'],
        'Sample1': [0.5, 0.6]
    })
    mock_read_csv.return_value = mock_df
    
    # Mock mapper functions
    mock_mapper.map_methylome.return_value = mock_df
    
    # Call the function
    result = map_command(mock_args)
    
    # Check that the mapper was initialized correctly
    mock_mapper_class.assert_called_once_with(
        manifest_file='manifest.csv',
        download_manifest=True,
        manifest_dir=None,
        input_file='input.csv',
        probe_col='IlmnID',
        chr_col='CHR',
        skiprows=7
    )
    
    # Check that the manifest was loaded
    mock_mapper.load_manifest.assert_called_once()
    
    # Check that read_csv was called with the correct file
    mock_read_csv.assert_called_once_with('input.csv')
    
    # Check that map_methylome was called with the correct arguments
    mock_mapper.map_methylome.assert_called_once_with(
        methylome_df=mock_df,
        probe_id_col='ProbeID',
        output_col='chr'
    )
    
    # Check that save_mapped_data was called with the correct arguments
    mock_mapper.save_mapped_data.assert_called_once_with(mock_df, 'output.csv')
    
    # Check that the function returned the mapped data
    assert result is mock_df


@patch('methylimpute.cli.setup_logger')
@patch('methylimpute.cli.MethylImputer')
def test_impute_command(mock_imputer_class, mock_setup_logger):
    """Test the impute command."""
    # Set up mocks
    mock_args = MagicMock()
    mock_args.input = 'input.csv'
    mock_args.output = 'output.csv'
    mock_args.chr_col = 'chr'
    mock_args.max_iter = 10
    mock_args.n_nearest = 10
    mock_args.threshold = 0.05
    mock_args.save_chunks = False
    mock_args.chunk_dir = None
    mock_args.jobs = 1
    mock_args.debug = False
    
    # Mock imputer instance
    mock_imputer = MagicMock()
    mock_imputer_class.return_value = mock_imputer
    
    # Call the function
    impute_command(mock_args)
    
    # Check that the imputer was initialized correctly
    mock_imputer_class.assert_called_once_with(
        max_iter=10,
        n_nearest_features=10,
        missing_threshold=0.05,
        n_jobs=1
    )
    
    # Check that impute was called with the correct arguments
    mock_imputer.impute.assert_called_once_with(
        input_file='input.csv',
        chr_col='chr',
        output_file='output.csv',
        save_chunks=False,
        chunk_dir=None
    )


@patch('methylimpute.cli.setup_logger')
@patch('methylimpute.cli.CpGMapper')
@patch('methylimpute.cli.MethylImputer')
@patch('pandas.read_csv')
def test_pipeline_command(mock_read_csv, mock_imputer_class, mock_mapper_class, mock_setup_logger):
    """Test the pipeline command."""
    # Set up mocks
    mock_args = MagicMock()
    mock_args.manifest = 'manifest.csv'
    mock_args.input = 'input.csv'
    mock_args.output = 'output.csv'
    mock_args.mapped_output = 'mapped.csv'
    mock_args.probe_col = 'ProbeID'
    mock_args.chr_col = 'chr'
    mock_args.skiprows = 7
    mock_args.max_iter = 10
    mock_args.n_nearest = 10
    mock_args.threshold = 0.05
    mock_args.save_chunks = False
    mock_args.chunk_dir = None
    mock_args.jobs = 1
    mock_args.no_download = False
    mock_args.debug = False
    mock_args.manifest_dir = None
    
    # Mock mapper instance
    mock_mapper = MagicMock()
    mock_mapper_class.return_value = mock_mapper
    
    # Mock imputer instance
    mock_imputer = MagicMock()
    mock_imputer_class.return_value = mock_imputer
    
    # Mock DataFrame
    mock_df = pd.DataFrame({
        'ProbeID': ['cg00000029', 'cg00000108'],
        'Sample1': [0.5, 0.6]
    })
    mock_read_csv.return_value = mock_df
    
    # Mock mapper functions
    mock_mapper.map_methylome.return_value = mock_df
    
    # Call the function
    pipeline_command(mock_args)
    
    # Check that the mapper was initialized correctly
    mock_mapper_class.assert_called_once_with(
        manifest_file='manifest.csv',
        download_manifest=True,
        manifest_dir=None,
        input_file='input.csv',
        probe_col='IlmnID',
        chr_col='CHR',
        skiprows=7
    )
    
    # Check that the manifest was loaded
    mock_mapper.load_manifest.assert_called_once()
    
    # Check that read_csv was called with the correct file
    mock_read_csv.assert_called_once_with('input.csv')
    
    # Check that map_methylome was called with the correct arguments
    mock_mapper.map_methylome.assert_called_once_with(
        methylome_df=mock_df,
        probe_id_col='ProbeID',
        output_col='chr'
    )
    
    # Check that save_mapped_data was called if mapped_output was provided
    mock_mapper.save_mapped_data.assert_called_once_with(mock_df, 'mapped.csv')
    
    # Check that the imputer was initialized correctly
    mock_imputer_class.assert_called_once_with(
        max_iter=10,
        n_nearest_features=10,
        missing_threshold=0.05,
        n_jobs=1
    )
    
    # Check that impute was called with the correct arguments
    mock_imputer.impute.assert_called_once_with(
        input_df=mock_df,
        chr_col='chr',
        output_file='output.csv',
        save_chunks=False,
        chunk_dir=None
    )


@patch('methylimpute.cli.parse_args')
@patch('methylimpute.cli.map_command')
@patch('methylimpute.cli.impute_command')
@patch('methylimpute.cli.pipeline_command')
def test_main_map(mock_pipeline, mock_impute, mock_map, mock_parse_args):
    """Test the main function with map command."""
    # Set up mocks
    mock_args = MagicMock()
    mock_args.command = 'map'
    mock_parse_args.return_value = mock_args
    
    # Redirect stdout to capture print output
    captured_output = StringIO()
    sys.stdout = captured_output
    
    # Call the function
    main()
    
    # Restore stdout
    sys.stdout = sys.__stdout__
    
    # Check that the correct command function was called
    mock_map.assert_called_once_with(mock_args)
    mock_impute.assert_not_called()
    mock_pipeline.assert_not_called()


@patch('methylimpute.cli.parse_args')
@patch('methylimpute.cli.map_command')
@patch('methylimpute.cli.impute_command')
@patch('methylimpute.cli.pipeline_command')
def test_main_impute(mock_pipeline, mock_impute, mock_map, mock_parse_args):
    """Test the main function with impute command."""
    # Set up mocks
    mock_args = MagicMock()
    mock_args.command = 'impute'
    mock_parse_args.return_value = mock_args
    
    # Redirect stdout to capture print output
    captured_output = StringIO()
    sys.stdout = captured_output
    
    # Call the function
    main()
    
    # Restore stdout
    sys.stdout = sys.__stdout__
    
    # Check that the correct command function was called
    mock_map.assert_not_called()
    mock_impute.assert_called_once_with(mock_args)
    mock_pipeline.assert_not_called()


@patch('methylimpute.cli.parse_args')
@patch('methylimpute.cli.map_command')
@patch('methylimpute.cli.impute_command')
@patch('methylimpute.cli.pipeline_command')
def test_main_pipeline(mock_pipeline, mock_impute, mock_map, mock_parse_args):
    """Test the main function with pipeline command."""
    # Set up mocks
    mock_args = MagicMock()
    mock_args.command = 'pipeline'
    mock_parse_args.return_value = mock_args
    
    # Redirect stdout to capture print output
    captured_output = StringIO()
    sys.stdout = captured_output
    
    # Call the function
    main()
    
    # Restore stdout
    sys.stdout = sys.__stdout__
    
    # Check that the correct command function was called
    mock_map.assert_not_called()
    mock_impute.assert_not_called()
    mock_pipeline.assert_called_once_with(mock_args)


@patch('methylimpute.cli.parse_args')
@patch('methylimpute.cli.map_command')
@patch('methylimpute.cli.impute_command')
@patch('methylimpute.cli.pipeline_command')
def test_main_no_command(mock_pipeline, mock_impute, mock_map, mock_parse_args):
    """Test the main function with no command."""
    # Set up mocks
    mock_args = MagicMock()
    mock_args.command = None
    mock_parse_args.return_value = mock_args
    
    # Redirect stdout to capture print output
    captured_output = StringIO()
    sys.stdout = captured_output
    
    # Call the function
    main()
    
    # Restore stdout
    sys.stdout = sys.__stdout__
    
    # Check that no command function was called
    mock_map.assert_not_called()
    mock_impute.assert_not_called()
    mock_pipeline.assert_not_called()
    
    # Check that help message was printed
    output = captured_output.getvalue()
    assert "Please specify a command" in output
    assert "Use --help for more information" in output