from copy import copy
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from shapely.geometry import Polygon
import pygeohash


class MajorTOMGrid:
    """
    A class to generate a global grid based on the distance between grid points,
    based on the MajorTOM grid specification.
    The grid is defined by latitude and longitude ranges and can be used for geospatial analysis.
    """

    EARTH_RADIUS = 6378137
    WGS84 = "EPSG:4326"
    # Equal area projection for centroid calculation
    EA_PROJ = "+proj=sinu"

    def __init__(
        self,
        dist: float = 10_000,
        geohash_precision: pygeohash.GeohashPrecision = 8,
        utm_definition: Literal["center", "bottomleft"] = "center",
    ) -> None:
        """Initiate a MajorTOM grid."""
        if dist <= 0:
            raise ValueError("Grid spacing must be positive")
        geohash_precision = int(geohash_precision)

        if not 1 <= geohash_precision <= 12:
            raise ValueError("`geohash_precision` must be an integer between 1 and 12.")

        if utm_definition not in ("center", "bottomleft"):
            raise ValueError(
                "`utm_definition` must be either 'center' or 'bottomleft'."
            )

        self.dist = dist
        self.geohash_precision = geohash_precision
        self.utm_definition = utm_definition
        self.n_rows = self._get_n_rows()
        self.lat_spacing = 180.0 / self.n_rows

        self._df = self._construct_table()

        self._points = None
        self._cells = None

    @property
    def df(self) -> pd.DataFrame:
        """Return a DataFrame containing the MajorTOM grid"""
        return self._df

    @df.setter
    def df(self, value: pd.DataFrame) -> None:
        # This method makes sure that cached geometries are invalidated
        # when self.df is updated

        # Verify that the new value is a dataframe
        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"Expected pandas.DataFrame, got {type(value).__name__}")
        # Check columns exist
        missing_cols = set(self.df.columns) - set(value.columns)
        if missing_cols:
            raise ValueError(f"DataFrame missing required columns: {missing_cols}")

        self._df = value

        # Clear cached geometries
        self._points = None
        self._cells = None

    def filter(
        self, geometry: Polygon | None = None, buffer_ratio: float = 0.0
    ) -> "MajorTOMGrid":
        """
        Filter the grid by intersection with a geometry.

        Parameters
        ----------
        geometry : shapely.geometry.Polygon or None
            The geometry to filter the grid by. Only grid cells intersecting this geometry will be kept.
        buffer_ratio : float, optional
            Ratio to buffer the grid cells before filtering (default is 0.0).

        Returns
        -------
        MajorTOMGrid
            A new MajorTOMGrid instance filtered to only include intersecting cells.
        """
        new_instance = copy(self)

        # Rough pre-filtering of points based on the geometry's bounds
        # Extend minium values by one cell spacing to account that the
        # grid point of a cell could lie outside of the geometry,
        # with the cell still intersecting it.

        min_lon, min_lat, max_lon, max_lat = geometry.bounds

        min_lat -= self.lat_spacing * (1 + buffer_ratio)
        max_lat += self.lat_spacing * buffer_ratio

        # Find the largest lon spacing
        lats_in_bbox = self.df[(self.df.lat >= min_lat) & (self.df.lat <= max_lat)]
        largest_lon_spacing = lats_in_bbox.lon_spacing.max()

        min_lon -= largest_lon_spacing * (1 + buffer_ratio)
        max_lon += largest_lon_spacing * buffer_ratio

        # Filter DataFrame:
        # This will automatically clear _points and _cells
        new_instance.df = self.df[
            self.df["lon"].between(min_lon, max_lon)
            & self.df["lat"].between(min_lat, max_lat)
        ]

        # Filter based on cell intersection
        gdf_cells = new_instance.get_cells(buffer_ratio=buffer_ratio)
        intersecting_cells = gdf_cells[gdf_cells.intersects(geometry)]

        # Filter df to match intersecting cells and clear cache again
        new_instance.df = new_instance.df.loc[intersecting_cells.index]

        return new_instance

    def get_points(self) -> gpd.GeoDataFrame:
        """
        Get a GeoDataFrame containing the point geometries.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with point geometries for each grid cell center.
        """
        if self._points is None:
            self._points = gpd.GeoDataFrame(
                self.df,
                geometry=gpd.points_from_xy(self.df.lon, self.df.lat),
                crs=self.WGS84,
            )
        return self._points

    def get_cells(self, buffer_ratio: float = 0.0) -> gpd.GeoDataFrame:
        """
        Get a GeoDataFrame containing the cell geometries.

        Parameters
        ----------
        buffer_ratio : float, optional
            Ratio to buffer the grid cells (default is 0.0).

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with polygon geometries for each grid cell.
        """
        if self._cells is None:
            self._cells = gpd.GeoDataFrame(
                self.df,
                geometry=self._get_cell_geometry(self.df, buffer_ratio=buffer_ratio),
                crs=self.WGS84,
            )
        return self._cells

    def _construct_table(self) -> pd.DataFrame:
        """Construct the main DataFrame representing the grid."""
        rows = np.arange(self.n_rows)
        lats = self._get_row_lat(rows)

        # Iterate over rows
        row_dfs = []
        for row, lat in zip(rows, lats):
            n_cols = self._get_n_cols(lat)
            cols = np.arange(n_cols)
            lon_spacing = 360.0 / n_cols
            lons = self._get_col_lon(cols, lon_spacing)

            row_df = pd.DataFrame(
                {
                    "lon_idx": cols,
                    "lat_idx": row,
                    "lon": lons,
                    "lat": lat,
                    "lon_spacing": lon_spacing,
                }
            )
            row_dfs.append(row_df)
        # Combine DataFrames of individual rows
        df = pd.concat(row_dfs).reset_index(drop=True)
        # Insert ID and UTM zone columns
        df["id"] = self._get_geohash(df.lon, df.lat)
        df["utm_zone"] = self._get_utm_code(df.lon, df.lat, df.lon_spacing)
        return df

    def _get_n_rows(self) -> int:
        """Calculate the number of rows in the grid."""
        return int(np.ceil(np.pi * self.EARTH_RADIUS / self.dist))

    def _get_row_lat(self, row_idx: NDArray[np.int_]) -> NDArray[np.floating]:
        """Get the latitude values for given row indices."""
        return -90.0 + row_idx * self.lat_spacing

    def _get_n_cols(self, lat: NDArray[np.floating]) -> NDArray[np.int_]:
        """Calculate the number of columns for given latitudes."""
        # Clip latitude to prevent singularity
        lat_rad = np.deg2rad(np.clip(lat, -89.0, 89.0))
        circumference = 2 * np.pi * self.EARTH_RADIUS * np.cos(lat_rad)
        return int(np.ceil(circumference / self.dist))

    def _get_col_lon(
        self, col_idx: NDArray[np.int_], lon_spacing: float
    ) -> NDArray[np.floating]:
        """Get the longitude values for given column indices."""
        return -180.0 + col_idx * lon_spacing

    def _get_cell_center(
        self,
        lon: NDArray[np.floating],
        lat: NDArray[np.floating],
        lon_spacing: NDArray[np.floating] | None = None,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Get the grid cell center coordinates for given grid points."""
        lon_center = lon + lon_spacing
        lat_center = lat + self.lat_spacing
        return lon_center, lat_center

    def _get_utm_code(
        self,
        lon: NDArray[np.floating],
        lat: NDArray[np.floating],
        lon_spacing: NDArray[np.floating] | None = None,
    ) -> NDArray[np.str_]:
        """Calculate the UTM zone EPSG code for given grid points."""
        if self.utm_definition == "center":
            lon, lat = self._get_cell_center(lon, lat, lon_spacing)
        elif self.utm_definition == "bottomleft":
            # Keep grid point coordinates
            pass
        else:
            raise ValueError("`utm_definition` must be 'center' or 'bottomleft'.")

        # Calculate zone numbers (1-60)
        zone_number = (np.floor((lon + 180) / 6) % 60 + 1).astype(int)

        # Special zones for Norway
        # Create boolean masks for each special condition
        mask_norway = (lat >= 56.0) & (lat < 64.0) & (lon >= 3.0) & (lon < 12.0)
        zone_number[mask_norway] = 32

        # Special zones for Svalbard
        mask_high_lat = (lat >= 72.0) & (lat < 84.0)
        mask_zone_31 = mask_high_lat & (lon >= 0.0) & (lon < 9.0)
        mask_zone_33 = mask_high_lat & (lon >= 9.0) & (lon < 21.0)
        mask_zone_35 = mask_high_lat & (lon >= 21.0) & (lon < 33.0)
        mask_zone_37 = mask_high_lat & (lon >= 33.0) & (lon < 42.0)

        zone_number[mask_zone_31] = 31
        zone_number[mask_zone_33] = 33
        zone_number[mask_zone_35] = 35
        zone_number[mask_zone_37] = 37

        # Determine hemisphere and construct EPSG codes
        south_mask = lat < 0
        zone_number[south_mask] += 32_700
        zone_number[~south_mask] += 32_600

        return "EPSG:" + zone_number.astype(str)

    def _get_geohash(
        self,
        lon: NDArray[np.floating],
        lat: NDArray[np.floating],
    ) -> NDArray[np.str_]:
        """Calculate the geohashes for given grid points."""

        # Vectorized version of pygeohash encoder
        @np.vectorize(otypes=[str])
        def encode(lon, lat):
            return pygeohash.encode(
                longitude=lon, latitude=lat, precision=self.geohash_precision
            )

        return encode(lon, lat)

    def _get_cell_geometry(
        self, gdf: pd.DataFrame, buffer_ratio: float = 0.0
    ) -> gpd.GeoSeries:
        """Create polygon geometries for a given MajorTOMGrid DataFrame."""

        def make_polygon(row):
            """Create a Polygon from center point and spacing"""
            lon = row["lon"]
            lat = row["lat"]
            lon_spacing = row["lon_spacing"]
            lon_buffer = lon_spacing * buffer_ratio
            lat_buffer = self.lat_spacing * buffer_ratio

            min_lon = lon - lon_buffer
            min_lat = lat - lat_buffer
            max_lon = lon + lon_spacing + lon_buffer
            max_lat = lat + self.lat_spacing + lat_buffer

            return Polygon(
                [
                    [min_lon, min_lat],
                    [max_lon, min_lat],
                    [max_lon, max_lat],
                    [min_lon, max_lat],
                    [min_lon, min_lat],
                ]
            )

        # Apply the function to each row and create a new 'geometry' column
        return gdf.apply(make_polygon, axis=1)
