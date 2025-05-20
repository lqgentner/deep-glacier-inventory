"""
download.py

This module provides the GEEDownloader class for downloading and processing
geospatial data patches from Google Earth Engine (GEE).
"""

import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Literal, TypeVar
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

import ee
import google.api_core.exceptions
import numpy as np
import pandas as pd
import pyproj
import rasterio
import requests
from google.api_core import retry
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
from rasterio.io import MemoryFile


class GEEDownloader:
    """
    Downloads and processes geospatial data patches from Google Earth Engine.
    The patches must be specified according to the
    [Major TOM grid specification](https://arxiv.org/abs/2402.12095).
    This class handles downloading image patches from GEE, transforming coordinates,
    and saving results in Numpy and GeoTIFF formats.
    """

    def __init__(
        self,
        grid_dist: tuple[int, int] | int,
        padding_ratio: float,
        scale: int | float,
        bands: str | list[str],
        write_dir: str,
        file_format: Literal["NPY", "GEO_TIFF"] = "GEO_TIFF",
        file_name: str = "patch-{id}",
    ):
        """
        Initialize a GEEDownloader instance.

        Parameters
        ----------
        file_format : Literal["NPY", "GEO_TIFF"]
            Output file format. Must be either 'NPY' or 'GEO_TIFF'.
        grid_dist : int
            MajorTOM grid spacing in meters.
            Used to calculate the patch's height and width.
        padding_ratio : float
            Ratio of padding to apply around the patch.
        scale : int or float
            Pixel resolution in meters.
        bands : str or list of str
            Band or list of bands to download.
        write_dir : str
            The directory to write the files into.
        file_name : str
            File name with named formatting options to be auto-filled.
            Must include `{id}`, can include `{scale}` or `{bands}`.
        """
        # Set file reader and writer
        match file_format.lower():
            case "npy":
                self._file_handler = NumpyFileHandler()
            case "geo_tiff":
                self._file_handler = GeoTiffFileHandler()
            case _:
                raise ValueError("`file_format` must be 'NPY' or 'GEO_TIFF'")

        # Verify that the write directory exists
        write_dir: Path = Path(write_dir)
        if not write_dir.exists():
            write_dir.mkdir(parents=True)
            print(f"Created directory '{write_dir}'.")
        self.write_dir = write_dir

        if "{id}" not in file_name:
            raise ValueError(
                "`file_name` must include '{id}' formatting option to make the file name unique."
            )
        self.file_name = file_name

        # Calculate patch size in pixels
        self.grid_dist_px = int(grid_dist / scale)
        self.padding_px = int(self.grid_dist_px * padding_ratio)
        self.patch_size_px = self.grid_dist_px + 2 * self.padding_px

        self.scale = scale
        self.bands = bands

    def load_patch(
        self, image: ee.Image, coords: tuple[float, float], crs: str
    ) -> np.ndarray | rasterio.DatasetReader:
        """
        Load a patch from GEE without writing to disk.

        Parameters
        ----------
        image : ee.Image
            Earth Engine image to download.
        coords : tuple of float
            (longitude, latitude) coordinates in WGS84.
            The coordinates define the lower right corner of the patch,
            according to the Major TOM grid specification.
        crs : str
            Coordinate reference system.

        Returns
        -------
        Any
            The loaded data in the format specified by config.file_format.
        """
        bytes_ = self._get_patch_bytes(image=image, coords=coords, crs=crs)
        return self._file_handler.read(bytes_)

    def write_patch(
        self, image: ee.Image, coords: tuple[float, float], crs: str, id_: str
    ) -> Path | None:
        """
        Download and write a patch to disk.

        Parameters
        ----------
        image : ee.Image
            Earth Engine image to download.
        coords : tuple of float
            (longitude, latitude) coordinates in WGS84.
        crs : str
            Coordinate reference system.
        id : str
            Identifier for the patch, used in filename.

        Returns
        -------
        Path
            Path to the written file.
        """
        bytes_ = self._get_patch_bytes(image=image, coords=coords, crs=crs)
        obj = self._file_handler.read(bytes_)
        # Create kwargs of attributes which can be used for file naming
        attrs = {
            "id": id_,
            "scale": self.scale,
            "bands": "-".join([str(val) for val in self.bands]),
        }
        file_name = self.file_name.format(**attrs)
        file_path = self.write_dir / f"{file_name}{self._file_handler.file_extension}"
        self._file_handler.write(obj=obj, file_path=str(file_path))
        return file_path

    @retry.Retry()
    def _get_patch_bytes(
        self, image: ee.Image, coords: tuple[float, float], crs: str
    ) -> bytes:
        """Get a patch relative to the coordinates of the bottom right corner"""

        proj = ee.Projection(crs).atScale(self.scale).getInfo()

        # Transform coordinates from WGS84 to projected CRS
        transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        coords = transformer.transform(*coords)

        # Get scales out of the transform.
        scale_x = proj["transform"][0]
        scale_y = -proj["transform"][4]

        # Offset to the upper left corner.
        offset_x = -scale_x * self.padding_px
        offset_y = -scale_y * (self.grid_dist_px + self.padding_px)

        request = {
            "expression": image,
            "fileFormat": self._file_handler.format_name,
            "bandIds": self.bands,
            "grid": {
                "dimensions": {
                    "width": self.patch_size_px,
                    "height": self.patch_size_px,
                },
                "affineTransform": {
                    "scaleX": scale_x,
                    "shearX": 0,
                    "translateX": coords[0] + offset_x,
                    "shearY": 0,
                    "scaleY": scale_y,
                    "translateY": coords[1] + offset_y,
                },
                "crsCode": crs,
            },
        }
        return ee.data.computePixels(request)

    def write_patches_from_df(
        self, image: ee.Image, df: pd.DataFrame, max_workers: int
    ) -> list[Path | None]:
        """
        Batch download and write patches based on a DataFrame.

        Parameters
        ----------
        image : ee.Image
            Earth Engine image to download.
        df : pd.DataFrame
            DataFrame with columns ['lon', 'lat', 'utm_zone', 'id']
        max_workers: int
            The maximum number of threads that can be used to
            execute the downloading and writing operations.

        Returns
        -------
        list of Path or None
            List of file paths for written patches.
        """
        # Prepare arguments as lists
        coords_list = list(zip(df.lon, df.lat))
        crs_list = df.utm_zone.tolist()
        id_list = df.id.tolist()

        # Parallel processing in multiple threads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(
                    executor.map(
                        lambda args: self.write_patch(image, *args),
                        zip(coords_list, crs_list, id_list),
                    ),
                    total=len(df),
                    desc="Downloading patches",
                )
            )

        return results

    @retry.Retry()
    def load_thumb(self, cell: pd.Series, image: ee.Image) -> JpegImageFile:
        """Helper to display a patch using notebook widgets."""
        roi_coords = list(cell.geometry.exterior.coords)
        roi_ee = ee.Geometry.Polygon(roi_coords)
        url = image.getThumbURL(
            {
                "region": roi_ee,
                "dimensions": "250",
                "format": "jpg",
                "min": 0,
                "max": 5000,
                "bands": ["B4", "B3", "B2"],
            }
        )

        r = requests.get(url, stream=True, timeout=5)
        if r.status_code != 200:
            raise google.api_core.exceptions.from_http_response(r)

        return Image.open(io.BytesIO(r.content))


# Define type variable for file handler return types
T = TypeVar("T")


class FileHandler(ABC, Generic[T]):
    """Abstract base class for file format handlers."""

    @abstractmethod
    def read(self, bytes_: bytes) -> T:
        """Read data from bytes."""

    @abstractmethod
    def write(self, obj: T, file_path: Path) -> None:
        """Write data to file."""

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return the format name for GEE API."""

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """Return the file extension."""


class NumpyFileHandler(FileHandler[np.ndarray]):
    """Handler for NPY file format."""

    def read(self, bytes_: bytes) -> np.ndarray:
        """Read NumPy array from bytes."""
        return np.load(io.BytesIO(bytes_))

    def write(self, obj: np.ndarray, file_path: Path) -> None:
        """Write NumPy array to file."""
        np.save(file=str(file_path), arr=obj)

    @property
    def format_name(self) -> str:
        return "NPY"

    @property
    def file_extension(self) -> str:
        return ".npy"


class GeoTiffFileHandler(FileHandler[MemoryFile]):
    """Handler for GeoTIFF file format."""

    def read(self, bytes_: bytes) -> MemoryFile:
        """Read GeoTIFF from bytes."""
        return MemoryFile(bytes_)
        # with MemoryFile(bytes_) as memfile:
        #     return memfile.open()

    def write(self, obj: MemoryFile, file_path: Path) -> None:
        """Write GeoTIFF to file."""
        with obj.open() as src:
            profile = src.profile
            array = src.read()
        with rasterio.open(str(file_path), "w", **profile) as dst:
            dst.write(array)

    @property
    def format_name(self) -> str:
        return "GEO_TIFF"

    @property
    def file_extension(self) -> str:
        return ".tif"
