"""
Tool for subsetting OPERA product files to a geographic bounding box.

Supports GeoTIFF (.tif) and HDF5 (.h5) OPERA product files.

Usage (command-line):
    opera_subset --bbox <min_lon> <min_lat> <max_lon> <max_lat> \\
        [-o <output_dir>] <file1> [<file2> ...]

Example:
    opera_subset --bbox -118.5 33.5 -117.5 34.5 -o ./subset/ *.tif
"""

import argparse
import os
import sys

import numpy as np

try:
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.transform import array_bounds
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False

try:
    import h5py
    _H5PY_AVAILABLE = True
except ImportError:
    _H5PY_AVAILABLE = False


def _check_bbox(bbox):
    """Validate a bounding box tuple (min_lon, min_lat, max_lon, max_lat)."""
    if len(bbox) != 4:
        raise ValueError("bbox must have exactly 4 elements: (min_lon, min_lat, max_lon, max_lat)")
    min_lon, min_lat, max_lon, max_lat = bbox
    if min_lon >= max_lon:
        raise ValueError(f"min_lon ({min_lon}) must be less than max_lon ({max_lon})")
    if min_lat >= max_lat:
        raise ValueError(f"min_lat ({min_lat}) must be less than max_lat ({max_lat})")
    if min_lon < -180 or max_lon > 360:
        raise ValueError(f"Longitude values must be in the range [-180, 360]")
    if min_lat < -90 or max_lat > 90:
        raise ValueError(f"Latitude values must be in the range [-90, 90]")


def subset_geotiff(input_file, output_file, bbox):
    """
    Subset a GeoTIFF file to the given bounding box.

    Parameters
    ----------
    input_file : str
        Path to the input GeoTIFF file.
    output_file : str
        Path to the output GeoTIFF file.
    bbox : tuple of float
        Bounding box as (min_lon, min_lat, max_lon, max_lat) in geographic
        coordinates (EPSG:4326 / WGS 84 degrees).

    Returns
    -------
    bool
        True if the subset was written successfully, False if the bounding box
        does not overlap the file extent.

    Raises
    ------
    ImportError
        If rasterio is not installed.
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If the bounding box is invalid.
    """
    if not _RASTERIO_AVAILABLE:
        raise ImportError(
            "rasterio is required for GeoTIFF subsetting. "
            "Install it with: pip install rasterio"
        )
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    _check_bbox(bbox)

    min_lon, min_lat, max_lon, max_lat = bbox

    with rasterio.open(input_file) as src:
        # Reproject bbox to file CRS if necessary
        from rasterio.crs import CRS
        wgs84 = CRS.from_epsg(4326)
        file_crs = src.crs

        left, bottom, right, top = min_lon, min_lat, max_lon, max_lat

        if file_crs is not None and not file_crs.equals(wgs84):
            try:
                from rasterio.warp import transform_bounds
                left, bottom, right, top = transform_bounds(
                    wgs84, file_crs, min_lon, min_lat, max_lon, max_lat
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Could not reproject bounding box to file CRS: {exc}"
                ) from exc

        # Compute the pixel window that covers the bounding box
        window = from_bounds(left, bottom, right, top, src.transform)
        file_window = rasterio.windows.Window(0, 0, src.width, src.height)

        try:
            window = window.intersection(file_window)
        except rasterio.errors.WindowError:
            # Bounding box does not overlap the file extent
            return False

        if window.width <= 0 or window.height <= 0:
            return False

        # Read the data in the window
        data = src.read(window=window)
        transform = src.window_transform(window)

        # Prepare output metadata
        meta = src.meta.copy()
        meta.update({
            "height": data.shape[1],
            "width": data.shape[2],
            "transform": transform,
        })

        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with rasterio.open(output_file, "w", **meta) as dst:
            dst.write(data)

    return True


def subset_hdf5(input_file, output_file, bbox):
    """
    Subset an HDF5 OPERA file to the given bounding box.

    The function looks for 2-D latitude and longitude datasets (or coordinate
    arrays stored in the file) to determine which rows and columns fall within
    the requested bounding box.  If no geographic coordinate information can be
    found, a ``RuntimeError`` is raised.

    Parameters
    ----------
    input_file : str
        Path to the input HDF5 file.
    output_file : str
        Path to the output HDF5 file.
    bbox : tuple of float
        Bounding box as (min_lon, min_lat, max_lon, max_lat) in geographic
        coordinates (degrees).

    Returns
    -------
    bool
        True if the subset was written successfully, False if the bounding box
        does not overlap the file extent.

    Raises
    ------
    ImportError
        If h5py is not installed.
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If the bounding box is invalid.
    RuntimeError
        If no geographic coordinate information can be found in the file.
    """
    if not _H5PY_AVAILABLE:
        raise ImportError(
            "h5py is required for HDF5 subsetting. "
            "Install it with: pip install h5py"
        )
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    _check_bbox(bbox)

    min_lon, min_lat, max_lon, max_lat = bbox

    with h5py.File(input_file, "r") as src:
        # Locate latitude/longitude datasets
        lat_data, lon_data = _find_latlon_datasets(src)
        if lat_data is None or lon_data is None:
            raise RuntimeError(
                f"Cannot find latitude/longitude information in {input_file}. "
                "Ensure the file contains datasets named 'latitude'/'longitude' "
                "or similar geographic coordinate datasets."
            )

        lat = np.asarray(lat_data)
        lon = np.asarray(lon_data)

        # Support 1-D coordinate vectors
        if lat.ndim == 1 and lon.ndim == 1:
            lat_mask = (lat >= min_lat) & (lat <= max_lat)
            lon_mask = (lon >= min_lon) & (lon <= max_lon)
            row_idx = np.where(lat_mask)[0]
            col_idx = np.where(lon_mask)[0]
        elif lat.ndim == 2 and lon.ndim == 2:
            mask = (
                (lat >= min_lat) & (lat <= max_lat) &
                (lon >= min_lon) & (lon <= max_lon)
            )
            rows, cols = np.where(mask)
            if rows.size == 0:
                return False
            row_idx = np.arange(rows.min(), rows.max() + 1)
            col_idx = np.arange(cols.min(), cols.max() + 1)
        else:
            raise RuntimeError(
                "Latitude/longitude datasets have unsupported dimensions "
                f"(lat.ndim={lat.ndim}, lon.ndim={lon.ndim})."
            )

        if row_idx.size == 0 or col_idx.size == 0:
            return False

        row_start, row_stop = int(row_idx[0]), int(row_idx[-1]) + 1
        col_start, col_stop = int(col_idx[0]), int(col_idx[-1]) + 1

        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with h5py.File(output_file, "w") as dst:
            _copy_hdf5_subset(src, dst, row_start, row_stop, col_start, col_stop)

    return True


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

_LAT_NAMES = {"latitude", "lat", "/latitude", "/lat", "y", "y_coords"}
_LON_NAMES = {"longitude", "lon", "/longitude", "/lon", "x", "x_coords"}


def _find_latlon_datasets(h5file):
    """Return (lat_dataset, lon_dataset) from an open HDF5 file, or (None, None)."""
    lat_ds = None
    lon_ds = None

    def _visitor(name, obj):
        nonlocal lat_ds, lon_ds
        if isinstance(obj, h5py.Dataset):
            lower = name.lower().split("/")[-1]
            if lower in _LAT_NAMES and lat_ds is None:
                lat_ds = obj
            if lower in _LON_NAMES and lon_ds is None:
                lon_ds = obj

    h5file.visititems(_visitor)
    return lat_ds, lon_ds


def _copy_hdf5_subset(src, dst, row_start, row_stop, col_start, col_stop):
    """Recursively copy a spatial subset of datasets from src to dst."""
    # Copy root attributes
    for key, val in src.attrs.items():
        try:
            dst.attrs[key] = val
        except Exception:
            pass

    def _copy_item(name, obj):
        if isinstance(obj, h5py.Group):
            grp = dst.require_group(name)
            for key, val in obj.attrs.items():
                try:
                    grp.attrs[key] = val
                except Exception:
                    pass
        elif isinstance(obj, h5py.Dataset):
            data = obj[()]
            lower = name.lower().split("/")[-1]

            if data.ndim == 2:
                data = data[row_start:row_stop, col_start:col_stop]
            elif data.ndim == 3:
                data = data[:, row_start:row_stop, col_start:col_stop]
            elif data.ndim == 1:
                # 1-D coordinate array: subset rows or columns
                if lower in _LAT_NAMES:
                    data = data[row_start:row_stop]
                elif lower in _LON_NAMES:
                    data = data[col_start:col_stop]
                # Other 1-D datasets are copied as-is

            ds = dst.require_dataset(
                name,
                shape=data.shape,
                dtype=data.dtype,
                data=data,
            )
            for key, val in obj.attrs.items():
                try:
                    ds.attrs[key] = val
                except Exception:
                    pass

    src.visititems(_copy_item)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def subset_opera_file(input_file, output_file, bbox):
    """
    Subset a single OPERA product file to the given bounding box.

    The file format is detected automatically from the file extension:
    * ``.tif`` / ``.tiff`` → GeoTIFF (requires rasterio)
    * ``.h5`` / ``.hdf5`` / ``.nc`` → HDF5 (requires h5py)

    Parameters
    ----------
    input_file : str
        Path to the input OPERA product file.
    output_file : str
        Path to the output (subsetted) file.
    bbox : tuple of float
        Bounding box as ``(min_lon, min_lat, max_lon, max_lat)`` in WGS 84
        geographic coordinates (degrees).

    Returns
    -------
    bool
        ``True`` if the output file was written, ``False`` if the bounding
        box does not overlap the file's spatial extent.

    Raises
    ------
    ValueError
        If the file extension is not recognised or the bounding box is invalid.
    """
    ext = os.path.splitext(input_file)[-1].lower()
    if ext in {".tif", ".tiff"}:
        return subset_geotiff(input_file, output_file, bbox)
    elif ext in {".h5", ".hdf5", ".nc"}:
        return subset_hdf5(input_file, output_file, bbox)
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            "Supported formats: .tif, .tiff, .h5, .hdf5, .nc"
        )


def subset_opera_files(input_files, output_dir, bbox, suffix="_subset"):
    """
    Subset multiple OPERA product files to the given bounding box.

    Parameters
    ----------
    input_files : list of str
        Paths to the input OPERA product files.
    output_dir : str
        Directory where subsetted files will be written.
    bbox : tuple of float
        Bounding box as ``(min_lon, min_lat, max_lon, max_lat)`` in WGS 84
        geographic coordinates (degrees).
    suffix : str, optional
        Suffix appended to the base filename (before the extension) for the
        output files.  Default is ``'_subset'``.

    Returns
    -------
    list of str
        Paths of output files that were successfully written.
    """
    os.makedirs(output_dir, exist_ok=True)
    written = []
    for input_file in input_files:
        base = os.path.basename(input_file)
        name, ext = os.path.splitext(base)
        output_file = os.path.join(output_dir, f"{name}{suffix}{ext}")
        success = subset_opera_file(input_file, output_file, bbox)
        if success:
            written.append(output_file)
        else:
            print(
                f"WARNING: bounding box does not overlap '{input_file}'; skipped.",
                file=sys.stderr,
            )
    return written


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _build_parser():
    parser = argparse.ArgumentParser(
        prog="opera_subset",
        description=(
            "Subset OPERA product files (GeoTIFF or HDF5) to a geographic "
            "bounding box."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Subset a single GeoTIFF file
  opera_subset --bbox -118.5 33.5 -117.5 34.5 product.tif

  # Subset multiple files, writing results to ./subset/
  opera_subset --bbox -118.5 33.5 -117.5 34.5 -o ./subset/ *.tif *.h5

  # Use a custom suffix for output filenames
  opera_subset --bbox -118.5 33.5 -117.5 34.5 --suffix _cropped product.tif
""",
    )
    parser.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help="Input OPERA product file(s) (.tif, .tiff, .h5, .hdf5, .nc).",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        required=True,
        help=(
            "Geographic bounding box in WGS 84 degrees: "
            "MIN_LON MIN_LAT MAX_LON MAX_LAT."
        ),
    )
    parser.add_argument(
        "-o", "--outdir",
        default=".",
        metavar="DIR",
        help="Output directory for subsetted files (default: current directory).",
    )
    parser.add_argument(
        "--suffix",
        default="_subset",
        metavar="SUFFIX",
        help=(
            "Suffix appended to the base filename for each output file "
            "(default: '_subset')."
        ),
    )
    return parser


def main(argv=None):
    """Entry point for the ``opera_subset`` command-line tool."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    bbox = tuple(args.bbox)
    written = subset_opera_files(args.files, args.outdir, bbox, suffix=args.suffix)

    if written:
        print(f"Wrote {len(written)} file(s):")
        for path in written:
            print(f"  {path}")
    else:
        print("No output files were written (bounding box may not overlap any input files).")
        sys.exit(1)


if __name__ == "__main__":
    main()
