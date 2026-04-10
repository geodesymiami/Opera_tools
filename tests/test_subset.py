"""Tests for opera_tools.subset."""

import os
import tempfile

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers for creating small synthetic test files
# ---------------------------------------------------------------------------

def _make_geotiff(path, lon_min=-118.5, lat_min=33.5, lon_max=-117.5, lat_max=34.5,
                  width=100, height=100, nodata=-9999):
    """Create a minimal single-band GeoTIFF covering the given extent."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
    data = np.arange(width * height, dtype=np.float32).reshape(1, height, width)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=CRS.from_epsg(4326),
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data)


def _make_hdf5(path, lat_min=33.5, lat_max=34.5, lon_min=-118.5, lon_max=-117.5,
               nlat=100, nlon=100):
    """Create a minimal HDF5 file with 1-D lat/lon coordinate arrays."""
    h5py = pytest.importorskip("h5py")

    lat = np.linspace(lat_max, lat_min, nlat)  # descending (north-to-south)
    lon = np.linspace(lon_min, lon_max, nlon)
    data = np.random.rand(nlat, nlon).astype(np.float32)

    with h5py.File(path, "w") as f:
        f.create_dataset("latitude", data=lat)
        f.create_dataset("longitude", data=lon)
        f.create_dataset("data", data=data)
        f.attrs["description"] = "synthetic test file"


# ---------------------------------------------------------------------------
# Tests for _check_bbox
# ---------------------------------------------------------------------------

class TestCheckBbox:
    def test_valid_bbox(self):
        from opera_tools.subset import _check_bbox
        _check_bbox((-118.5, 33.5, -117.5, 34.5))  # should not raise

    def test_wrong_length(self):
        from opera_tools.subset import _check_bbox
        with pytest.raises(ValueError, match="exactly 4"):
            _check_bbox((-118.5, 33.5, -117.5))

    def test_min_lon_ge_max_lon(self):
        from opera_tools.subset import _check_bbox
        with pytest.raises(ValueError, match="min_lon"):
            _check_bbox((-117.5, 33.5, -118.5, 34.5))

    def test_min_lat_ge_max_lat(self):
        from opera_tools.subset import _check_bbox
        with pytest.raises(ValueError, match="min_lat"):
            _check_bbox((-118.5, 34.5, -117.5, 33.5))

    def test_invalid_longitude(self):
        from opera_tools.subset import _check_bbox
        with pytest.raises(ValueError, match="Longitude"):
            _check_bbox((-200.0, 33.5, -117.5, 34.5))

    def test_invalid_latitude(self):
        from opera_tools.subset import _check_bbox
        with pytest.raises(ValueError, match="Latitude"):
            _check_bbox((-118.5, -91.0, -117.5, 34.5))


# ---------------------------------------------------------------------------
# Tests for subset_geotiff
# ---------------------------------------------------------------------------

class TestSubsetGeotiff:
    def test_basic_subset(self, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        from opera_tools.subset import subset_geotiff

        src = str(tmp_path / "input.tif")
        dst = str(tmp_path / "output.tif")
        _make_geotiff(src)

        # Subset to a smaller bbox inside the file extent
        result = subset_geotiff(src, dst, (-118.2, 33.7, -117.8, 34.2))
        assert result is True
        assert os.path.isfile(dst)

        with rasterio.open(dst) as f:
            assert f.width > 0
            assert f.height > 0
            # Output must be smaller than input
            with rasterio.open(src) as orig:
                assert f.width <= orig.width
                assert f.height <= orig.height

    def test_no_overlap_returns_false(self, tmp_path):
        pytest.importorskip("rasterio")
        from opera_tools.subset import subset_geotiff

        src = str(tmp_path / "input.tif")
        dst = str(tmp_path / "output.tif")
        _make_geotiff(src)

        # Bbox completely outside the file extent
        result = subset_geotiff(src, dst, (10.0, 10.0, 20.0, 20.0))
        assert result is False
        assert not os.path.isfile(dst)

    def test_file_not_found(self, tmp_path):
        pytest.importorskip("rasterio")
        from opera_tools.subset import subset_geotiff

        with pytest.raises(FileNotFoundError):
            subset_geotiff(str(tmp_path / "missing.tif"), str(tmp_path / "out.tif"),
                           (-118.5, 33.5, -117.5, 34.5))

    def test_invalid_bbox_raises(self, tmp_path):
        pytest.importorskip("rasterio")
        from opera_tools.subset import subset_geotiff

        src = str(tmp_path / "input.tif")
        _make_geotiff(src)
        with pytest.raises(ValueError):
            subset_geotiff(src, str(tmp_path / "out.tif"), (-117.5, 33.5, -118.5, 34.5))

    def test_full_overlap_preserves_data(self, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        from opera_tools.subset import subset_geotiff

        src = str(tmp_path / "input.tif")
        dst = str(tmp_path / "output.tif")
        _make_geotiff(src)

        # Subset with the exact file extent should copy all pixels
        result = subset_geotiff(src, dst, (-118.5, 33.5, -117.5, 34.5))
        assert result is True

        with rasterio.open(src) as s, rasterio.open(dst) as d:
            assert s.width == d.width
            assert s.height == d.height

    def test_output_directory_created(self, tmp_path):
        pytest.importorskip("rasterio")
        from opera_tools.subset import subset_geotiff

        src = str(tmp_path / "input.tif")
        _make_geotiff(src)
        out_dir = tmp_path / "subdir" / "nested"
        dst = str(out_dir / "output.tif")

        result = subset_geotiff(src, dst, (-118.2, 33.7, -117.8, 34.2))
        assert result is True
        assert os.path.isfile(dst)


# ---------------------------------------------------------------------------
# Tests for subset_hdf5
# ---------------------------------------------------------------------------

class TestSubsetHdf5:
    def test_basic_subset(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        from opera_tools.subset import subset_hdf5

        src = str(tmp_path / "input.h5")
        dst = str(tmp_path / "output.h5")
        _make_hdf5(src)

        result = subset_hdf5(src, dst, (-118.2, 33.7, -117.8, 34.2))
        assert result is True
        assert os.path.isfile(dst)

        with h5py.File(src) as s, h5py.File(dst) as d:
            assert d["latitude"].shape[0] <= s["latitude"].shape[0]
            assert d["longitude"].shape[0] <= s["longitude"].shape[0]
            # Lat/lon values in output must be within bbox
            assert d["latitude"][()].min() >= 33.7 - 1e-6
            assert d["latitude"][()].max() <= 34.2 + 1e-6
            assert d["longitude"][()].min() >= -118.2 - 1e-6
            assert d["longitude"][()].max() <= -117.8 + 1e-6

    def test_no_overlap_returns_false(self, tmp_path):
        pytest.importorskip("h5py")
        from opera_tools.subset import subset_hdf5

        src = str(tmp_path / "input.h5")
        dst = str(tmp_path / "output.h5")
        _make_hdf5(src)

        result = subset_hdf5(src, dst, (10.0, 10.0, 20.0, 20.0))
        assert result is False
        assert not os.path.isfile(dst)

    def test_file_not_found(self, tmp_path):
        pytest.importorskip("h5py")
        from opera_tools.subset import subset_hdf5

        with pytest.raises(FileNotFoundError):
            subset_hdf5(str(tmp_path / "missing.h5"), str(tmp_path / "out.h5"),
                        (-118.5, 33.5, -117.5, 34.5))

    def test_attributes_preserved(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        from opera_tools.subset import subset_hdf5

        src = str(tmp_path / "input.h5")
        dst = str(tmp_path / "output.h5")
        _make_hdf5(src)

        subset_hdf5(src, dst, (-118.2, 33.7, -117.8, 34.2))

        with h5py.File(dst) as d:
            assert d.attrs.get("description") == "synthetic test file"


# ---------------------------------------------------------------------------
# Tests for subset_opera_file (dispatcher)
# ---------------------------------------------------------------------------

class TestSubsetOperaFile:
    def test_dispatches_geotiff(self, tmp_path):
        pytest.importorskip("rasterio")
        from opera_tools.subset import subset_opera_file

        src = str(tmp_path / "product.tif")
        dst = str(tmp_path / "product_subset.tif")
        _make_geotiff(src)

        result = subset_opera_file(src, dst, (-118.2, 33.7, -117.8, 34.2))
        assert result is True

    def test_dispatches_hdf5(self, tmp_path):
        pytest.importorskip("h5py")
        from opera_tools.subset import subset_opera_file

        src = str(tmp_path / "product.h5")
        dst = str(tmp_path / "product_subset.h5")
        _make_hdf5(src)

        result = subset_opera_file(src, dst, (-118.2, 33.7, -117.8, 34.2))
        assert result is True

    def test_unsupported_extension(self, tmp_path):
        from opera_tools.subset import subset_opera_file

        with pytest.raises(ValueError, match="Unsupported file extension"):
            subset_opera_file(
                str(tmp_path / "file.xyz"),
                str(tmp_path / "out.xyz"),
                (-118.5, 33.5, -117.5, 34.5),
            )


# ---------------------------------------------------------------------------
# Tests for subset_opera_files (batch)
# ---------------------------------------------------------------------------

class TestSubsetOperaFiles:
    def test_batch_geotiff(self, tmp_path):
        pytest.importorskip("rasterio")
        from opera_tools.subset import subset_opera_files

        src1 = str(tmp_path / "a.tif")
        src2 = str(tmp_path / "b.tif")
        _make_geotiff(src1)
        _make_geotiff(src2)

        out_dir = str(tmp_path / "subset")
        written = subset_opera_files([src1, src2], out_dir,
                                     (-118.2, 33.7, -117.8, 34.2))
        assert len(written) == 2
        for path in written:
            assert os.path.isfile(path)

    def test_skips_non_overlapping(self, tmp_path):
        pytest.importorskip("rasterio")
        from opera_tools.subset import subset_opera_files

        src1 = str(tmp_path / "a.tif")
        src2 = str(tmp_path / "b.tif")
        _make_geotiff(src1)
        _make_geotiff(src2)

        # Bbox far from both files
        out_dir = str(tmp_path / "subset")
        written = subset_opera_files([src1, src2], out_dir, (10.0, 10.0, 20.0, 20.0))
        assert len(written) == 0

    def test_custom_suffix(self, tmp_path):
        pytest.importorskip("rasterio")
        from opera_tools.subset import subset_opera_files

        src = str(tmp_path / "product.tif")
        _make_geotiff(src)

        out_dir = str(tmp_path / "subset")
        written = subset_opera_files([src], out_dir,
                                     (-118.2, 33.7, -117.8, 34.2),
                                     suffix="_cropped")
        assert len(written) == 1
        assert written[0].endswith("_cropped.tif")


# ---------------------------------------------------------------------------
# Tests for CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def test_basic_cli(self, tmp_path):
        pytest.importorskip("rasterio")
        from opera_tools.subset import main

        src = str(tmp_path / "product.tif")
        out_dir = str(tmp_path / "out")
        _make_geotiff(src)

        main(["--bbox", "-118.2", "33.7", "-117.8", "34.2",
              "-o", out_dir, src])

        assert os.path.isfile(os.path.join(out_dir, "product_subset.tif"))

    def test_cli_no_overlap_exits_nonzero(self, tmp_path):
        pytest.importorskip("rasterio")
        from opera_tools.subset import main

        src = str(tmp_path / "product.tif")
        out_dir = str(tmp_path / "out")
        _make_geotiff(src)

        with pytest.raises(SystemExit) as exc_info:
            main(["--bbox", "10.0", "10.0", "20.0", "20.0",
                  "-o", out_dir, src])
        assert exc_info.value.code != 0

    def test_cli_custom_suffix(self, tmp_path):
        pytest.importorskip("rasterio")
        from opera_tools.subset import main

        src = str(tmp_path / "product.tif")
        out_dir = str(tmp_path / "out")
        _make_geotiff(src)

        main(["--bbox", "-118.2", "33.7", "-117.8", "34.2",
              "-o", out_dir, "--suffix", "_cropped", src])

        assert os.path.isfile(os.path.join(out_dir, "product_cropped.tif"))
