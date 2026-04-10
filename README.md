# Opera_tools

Python tools for working with [OPERA](https://www.jpl.nasa.gov/go/opera) SAR/InSAR product files.

## Features

* **Spatial subsetting** – Clip OPERA product files to a geographic bounding box.
  * Supports **GeoTIFF** (`.tif`, `.tiff`) products (RTC, DSWx, …)
  * Supports **HDF5** (`.h5`, `.hdf5`, `.nc`) products (CSLC, DISP-S1, …)

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

Or directly from GitHub:

```bash
pip install git+https://github.com/geodesymiami/Opera_tools.git
```

## Quick start

### Command-line

```bash
# Subset one or more GeoTIFF files to a bounding box
opera_subset --bbox -118.5 33.5 -117.5 34.5 product.tif

# Subset multiple files and write results to a directory
opera_subset --bbox -118.5 33.5 -117.5 34.5 -o ./subset/ *.tif *.h5

# Custom output suffix
opera_subset --bbox -118.5 33.5 -117.5 34.5 --suffix _cropped product.h5
```

Positional and optional arguments:

```
usage: opera_subset [-h] --bbox MIN_LON MIN_LAT MAX_LON MAX_LAT
                    [-o DIR] [--suffix SUFFIX]
                    FILE [FILE ...]

positional arguments:
  FILE                  Input OPERA product file(s) (.tif, .tiff, .h5, .hdf5, .nc).

optional arguments:
  --bbox MIN_LON MIN_LAT MAX_LON MAX_LAT
                        Geographic bounding box in WGS 84 degrees.
  -o DIR, --outdir DIR  Output directory (default: current directory).
  --suffix SUFFIX       Suffix appended to each output filename (default: _subset).
```

### Python API

```python
from opera_tools import subset_opera_file, subset_opera_files

# Subset a single file
success = subset_opera_file(
    "OPERA_L2_RTC-S1_T064-135524-IW1_20231101T015038Z_20231101T101117Z_S1A_30_v1.0.tif",
    "output_subset.tif",
    bbox=(-118.5, 33.5, -117.5, 34.5),  # (min_lon, min_lat, max_lon, max_lat)
)

# Subset a batch of files
written = subset_opera_files(
    input_files=["file1.tif", "file2.h5"],
    output_dir="./subset/",
    bbox=(-118.5, 33.5, -117.5, 34.5),
)
```

## Dependencies

* [numpy](https://numpy.org/)
* [rasterio](https://rasterio.readthedocs.io/) – GeoTIFF subsetting
* [h5py](https://www.h5py.org/) – HDF5 subsetting

## Running tests

```bash
pip install pytest
pytest tests/
```
