"""Command-line interface for the MethylImpute package."""

import argparse
import os
import logging
import pandas as pd
from .mapper import CpGMapper
from .imputer import MethylImputer
from .utils import setup_logger

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MethylImpute: Map CpG sites to chromosomes and impute missing methylation values"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Map command
    map_parser = subparsers.add_parser("map", help="Map CpG sites to chromosomes")
    map_parser.add_argument(
        "--manifest", "-m", required=False,
        help="Path to Illumina EPIC v1.0 manifest file (optional, will download if not provided)"
    )
    map_parser.add_argument(
        "--manifest-dir",
        help="Directory to download and store manifest files (default: [input-dir]/manifests)"
    )
    map_parser.add_argument(
        "--input", "-i", required=True,
        help="Path to methylome CSV file"
    )
    map_parser.add_argument(
        "--output", "-o", required=True,
        help="Path to output CSV file with chromosome mapping"
    )
    map_parser.add_argument(
        "--probe-col", default="ProbeID",
        help="Column name for probe IDs in methylome data (default: ProbeID)"
    )
    map_parser.add_argument(
        "--chr-col", default="chr",
        help="Column name for output chromosome mapping (default: chr)"
    )
    map_parser.add_argument(
        "--skiprows", type=int, default=7,
        help="Number of header rows to skip in manifest file (default: 7)"
    )
    map_parser.add_argument(
        "--no-download", action="store_true",
        help="Disable automatic manifest download if manifest file is not provided"
    )
    
    # Impute command
    impute_parser = subparsers.add_parser("impute", help="Impute missing methylation values")
    impute_parser.add_argument(
        "--input", "-i", required=True,
        help="Path to methylome CSV file with chromosome mapping"
    )
    impute_parser.add_argument(
        "--output", "-o", required=True,
        help="Path to output CSV file with imputed values"
    )
    impute_parser.add_argument(
        "--chr-col", default="chr",
        help="Column name for chromosome (default: chr)"
    )
    impute_parser.add_argument(
        "--max-iter", type=int, default=10,
        help="Maximum number of imputation iterations (default: 10)"
    )
    impute_parser.add_argument(
        "--n-nearest", type=int, default=10,
        help="Number of nearest features to use for imputation (default: 10)"
    )
    impute_parser.add_argument(
        "--threshold", type=float, default=0.05,
        help="Maximum fraction of missing values for a row to be kept (default: 0.05)"
    )
    impute_parser.add_argument(
        "--save-chunks", action="store_true",
        help="Save imputed data for each chromosome separately"
    )
    impute_parser.add_argument(
        "--chunk-dir",
        help="Directory to save chromosome chunks (required if --save-chunks is used)"
    )
    impute_parser.add_argument(
        "--jobs", "-j", type=int, default=1,
        help="Number of parallel jobs for processing (default: 1)"
    )
    
    # Pipeline command (map and impute in one go)
    pipeline_parser = subparsers.add_parser("pipeline", help="Run full pipeline (map and impute)")
    pipeline_parser.add_argument(
        "--manifest", "-m", required=False,
        help="Path to Illumina EPIC v1.0 manifest file (optional, will download if not provided)"
    )
    pipeline_parser.add_argument(
        "--manifest-dir",
        help="Directory to download and store manifest files (default: [input-dir]/manifests)"
    )
    pipeline_parser.add_argument(
        "--input", "-i", required=True,
        help="Path to methylome CSV file"
    )
    pipeline_parser.add_argument(
        "--output", "-o", required=True,
        help="Path to output CSV file with imputed values"
    )
    pipeline_parser.add_argument(
        "--mapped-output",
        help="Path to save intermediate mapped CSV file (optional)"
    )
    pipeline_parser.add_argument(
        "--probe-col", default="ProbeID",
        help="Column name for probe IDs in methylome data (default: ProbeID)"
    )
    pipeline_parser.add_argument(
        "--chr-col", default="chr",
        help="Column name for chromosome (default: chr)"
    )
    pipeline_parser.add_argument(
        "--skiprows", type=int, default=7,
        help="Number of header rows to skip in manifest file (default: 7)"
    )
    pipeline_parser.add_argument(
        "--max-iter", type=int, default=10,
        help="Maximum number of imputation iterations (default: 10)"
    )
    pipeline_parser.add_argument(
        "--n-nearest", type=int, default=10,
        help="Number of nearest features to use for imputation (default: 10)"
    )
    pipeline_parser.add_argument(
        "--threshold", type=float, default=0.05,
        help="Maximum fraction of missing values for a row to be kept (default: 0.05)"
    )
    pipeline_parser.add_argument(
        "--save-chunks", action="store_true",
        help="Save imputed data for each chromosome separately"
    )
    pipeline_parser.add_argument(
        "--chunk-dir",
        help="Directory to save chromosome chunks (required if --save-chunks is used)"
    )
    pipeline_parser.add_argument(
        "--jobs", "-j", type=int, default=1,
        help="Number of parallel jobs for processing (default: 1)"
    )
    pipeline_parser.add_argument(
        "--no-download", action="store_true",
        help="Disable automatic manifest download if manifest file is not provided"
    )
    
    # Debug mode
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()

def map_command(args):
    """Run the mapping command."""
    logger = setup_logger(level=logging.DEBUG if args.debug else logging.INFO)
    
    logger.info("Starting CpG to chromosome mapping")
    
    # Initialize mapper
    mapper = CpGMapper(
        manifest_file=args.manifest,
        download_manifest=not args.no_download,
        manifest_dir=args.manifest_dir,
        input_file=args.input,  # Pass input file for project directory
        probe_col='IlmnID',
        chr_col='CHR',
        skiprows=args.skiprows
    )
    
    # Load manifest
    mapper.load_manifest()
    
    # Load methylome data
    logger.info(f"Loading methylome data from {args.input}")
    methylome = pd.read_csv(args.input)
    
    # Map chromosomes
    mapped_data = mapper.map_methylome(
        methylome_df=methylome,
        probe_id_col=args.probe_col,
        output_col=args.chr_col
    )
    
    # Save mapped data
    mapper.save_mapped_data(mapped_data, args.output)
    logger.info("Mapping complete")
    
    return mapped_data

def impute_command(args):
    """Run the imputation command."""
    logger = setup_logger(level=logging.DEBUG if args.debug else logging.INFO)
    
    logger.info("Starting MICE imputation")
    
    # Initialize imputer
    imputer = MethylImputer(
        max_iter=args.max_iter,
        n_nearest_features=args.n_nearest,
        missing_threshold=args.threshold,
        n_jobs=args.jobs
    )
    
    # Run imputation
    imputer.impute(
        input_file=args.input,
        chr_col=args.chr_col,
        output_file=args.output,
        save_chunks=args.save_chunks,
        chunk_dir=args.chunk_dir
    )
    
    logger.info("Imputation complete")

def pipeline_command(args):
    """Run the full pipeline (mapping and imputation)."""
    logger = setup_logger(level=logging.DEBUG if args.debug else logging.INFO)
    
    logger.info("Starting full MethylImpute pipeline")
    
    # Step 1: Mapping
    logger.info("Step 1: CpG to chromosome mapping")
    
    # Initialize mapper
    mapper = CpGMapper(
        manifest_file=args.manifest,
        download_manifest=not args.no_download,
        manifest_dir=args.manifest_dir,
        input_file=args.input,  # Pass input file for project directory
        probe_col='IlmnID',
        chr_col='CHR',
        skiprows=args.skiprows
    )
    
    # Load manifest
    mapper.load_manifest()
    
    # Load methylome data
    logger.info(f"Loading methylome data from {args.input}")
    methylome = pd.read_csv(args.input)
    
    # Map chromosomes
    mapped_data = mapper.map_methylome(
        methylome_df=methylome,
        probe_id_col=args.probe_col,
        output_col=args.chr_col
    )
    
    # Save mapped data if requested
    if args.mapped_output:
        mapper.save_mapped_data(mapped_data, args.mapped_output)
        logger.info(f"Saved mapped data to {args.mapped_output}")
    
    # Step 2: Imputation
    logger.info("Step 2: MICE imputation")
    
    # Initialize imputer
    imputer = MethylImputer(
        max_iter=args.max_iter,
        n_nearest_features=args.n_nearest,
        missing_threshold=args.threshold,
        n_jobs=args.jobs
    )
    
    # Run imputation
    imputer.impute(
        input_df=mapped_data,
        chr_col=args.chr_col,
        output_file=args.output,
        save_chunks=args.save_chunks,
        chunk_dir=args.chunk_dir
    )
    
    logger.info("Pipeline complete")

def main():
    """Main entry point for the command-line interface."""
    args = parse_args()
    
    if args.command == "map":
        map_command(args)
    elif args.command == "impute":
        impute_command(args)
    elif args.command == "pipeline":
        pipeline_command(args)
    else:
        print("Please specify a command: map, impute, or pipeline")
        print("Use --help for more information")

if __name__ == "__main__":
    main()