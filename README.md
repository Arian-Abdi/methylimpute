# MethylImpute

A Python package for mapping CpG sites to chromosomes and imputing missing methylation beta values in Illumina EPIC arrays.

## Features

- **CpG Mapping**: Map CpG sites to chromosomes using Illumina manifest files
- **MICE Imputation**: Impute missing methylation values using Multivariate Imputation by Chained Equations
- **Chromosome-Specific Processing**: Process methylation data by chromosome for better imputation accuracy
- **Automatic Manifest Handling**: Automatically download and manage Illumina manifest files
- **Command-Line Interface**: Easy-to-use CLI with comprehensive options
- **Python API**: Programmatic interface for integration into analysis pipelines

## Installation

```bash
# Install from GitHub
pip install git+https://github.com/yourusername/methylimpute.git

# For development installation
git clone https://github.com/yourusername/methylimpute.git
cd methylimpute
pip install -e .
```

## Quick Start

### Command Line Usage

```bash
# Map CpG sites to chromosomes
methylimpute map --input data.csv --output mapped_data.csv

# Impute missing methylation values
methylimpute impute --input mapped_data.csv --output imputed_data.csv

# Run full pipeline (mapping + imputation)
methylimpute pipeline --input data.csv --output imputed_data.csv
```

### Python API Usage

```python
import pandas as pd
from methylimpute import CpGMapper, MethylImputer

# Load data
methylome = pd.read_csv('data.csv')

# Map chromosomes
mapper = CpGMapper()
mapper.load_manifest()
mapped_data = mapper.map_methylome(methylome)

# Impute missing values
imputer = MethylImputer(max_iter=10, n_jobs=4)
imputed_data = imputer.impute(input_df=mapped_data)
```

## Documentation

### Command Line Options

#### Global Options

- `--debug`: Enable debug logging

#### Map Command

```bash
methylimpute map --input INPUT_FILE --output OUTPUT_FILE [OPTIONS]
```

Options:
- `--manifest PATH`: Path to Illumina EPIC v1.0 manifest file (optional)
- `--manifest-dir DIR`: Directory to store manifest files (default: [input-dir]/manifests)
- `--probe-col COL`: Column name for probe IDs (default: ProbeID)
- `--chr-col COL`: Column name for chromosome mapping (default: chr)
- `--skiprows N`: Number of header rows to skip in manifest file (default: 7)
- `--no-download`: Disable automatic manifest download

#### Impute Command

```bash
methylimpute impute --input INPUT_FILE --output OUTPUT_FILE [OPTIONS]
```

Options:
- `--chr-col COL`: Column name for chromosome (default: chr)
- `--max-iter N`: Maximum number of imputation iterations (default: 10)
- `--n-nearest N`: Number of nearest features for imputation (default: 10)
- `--threshold N`: Maximum fraction of missing values for a row (default: 0.05)
- `--jobs N`: Number of parallel jobs for processing (default: 1)

#### Pipeline Command

```bash
methylimpute pipeline --input INPUT_FILE --output OUTPUT_FILE [OPTIONS]
```

Options: Combines all options from the map and impute commands, plus:
- `--mapped-output PATH`: Path to save intermediate mapped CSV file

## Data Requirements

- Input data should be a CSV file with:
  - A column for CpG probe IDs (default name: 'ProbeID')
  - Rows for beta values (numeric)
  
- The package works with Illumina EPIC v1.0 arrays by default

## Performance Considerations

- For large datasets (>100K probes), increase the `--jobs` parameter to use multiple CPU cores
- Adjust the `--threshold` parameter to control row filtering based on missing data
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
