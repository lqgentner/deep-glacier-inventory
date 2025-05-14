from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from shapely.geometry import Polygon
import pygeohash


class MajorTOMGrid:
    """
    A class to generate a global grid based on the distance between grid points, based on the MajorTOM grid specification.
    The grid is defined by latitude and longitude ranges and can be used for geospatial analysis.
    """

    EARTH_RADIUS = 6378137
    WGS84 = "EPSG:4326"
    # Equal area projection for centroid calculation
    EA_PROJ = "+proj=sinu"

    def __init__(
        self, dist: float = 10_000, geohash_precision: pygeohash.GeohashPrecision = 8
    ) -> None:
        """Initiate a MajorTOM grid."""
        if dist <= 0:
            raise ValueError("Grid spacing must be positive")
        geohash_precision = int(geohash_precision)
        if not 1 <= geohash_precision <= 12:
            raise ValueError("`geohash_precision` must be an integer between 1 and 12.")

        self.dist = dist
        self.n_rows = self._get_n_rows()
        self.lat_spacing = 180.0 / self.n_rows
        self.points = self._get_points()
        self.geohash_precision = geohash_precision

    def get_cells(self, aoi: Polygon, buffer_ratio: float = 0.0) -> gpd.GeoDataFrame:
        """Get a GeoDataFrame containing the cell geometries within an area of interest.
        Buffer is not considered for AOI. Should we change that?"""

        # Do a rough pre-filtering of points based on the geometry's bounds
        # This is faster than calculating the cells of all points
        min_lon, min_lat, max_lon, max_lat = aoi.bounds
        # Buffer min coordinates:
        # Point is in the bottomleft of cell, but cell could still intersect AOI
        min_lat -= self.lat_spacing
        # Find nearest smaller latitude value in points to determine lon_spacing
        smaller_lats = self.points.lat[self.points.lat <= min_lat]
        closest_lat_idx = smaller_lats.idxmax()
        lon_spacing = self.points.lon_spacing.iloc[closest_lat_idx]
        min_lon -= lon_spacing

        # Filter DataFrame based on bounds
        points = self.points[
            self.points["lon"].between(min_lon, max_lon)
            & self.points["lat"].between(min_lat, max_lat)
        ]

        # Construct cells
        cells = points.copy()
        cells.geometry = self._get_cell_geometry(cells, buffer_ratio=buffer_ratio)

        # Only keep cells intersecting the geometry
        cells = cells[cells.intersects(aoi)]

        # Add Encoding
        cells.insert(5, "id", self._get_geohash(cells))
        # Add CRS col
        cells.insert(6, "utm_zone", self._get_utm_code(cells))
        return cells

    def _get_points(self) -> gpd.GeoDataFrame:
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

        # Add geometry of points
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=self.WGS84
        )
        return gdf

    def _get_n_rows(self) -> int:
        return int(np.ceil(np.pi * self.EARTH_RADIUS / self.dist))

    def _get_row_lat(self, row_idx: NDArray[np.int_]) -> NDArray[np.floating]:
        return -90.0 + row_idx * self.lat_spacing

    def _get_n_cols(self, lat: NDArray[np.floating]) -> NDArray[np.int_]:
        # Clip latitude to prevent singularity
        lat_rad = np.deg2rad(np.clip(lat, -89.0, 89.0))
        circumference = 2 * np.pi * self.EARTH_RADIUS * np.cos(lat_rad)
        return int(np.ceil(circumference / self.dist))

    def _get_col_lon(
        self, col_idx: NDArray[np.int_], lon_spacing: float
    ) -> NDArray[np.floating]:
        return -180.0 + col_idx * lon_spacing

    def _get_utm_code(
        self,
        gdf: gpd.GeoDataFrame,
        utm_definition: Literal["center", "bottomleft"] = "center",
    ) -> pd.Series:
        if utm_definition == "center":
            centroid = gdf.to_crs(self.EA_PROJ).centroid.to_crs(gdf.crs)
            lon, lat = centroid.x, centroid.y
        elif utm_definition == "bottomleft":
            lon, lat = gdf.lon, gdf.lat
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

    def _get_geohash(self, df: pd.DataFrame) -> pd.Series:
        def get_hash(row):
            lon = row["lon"]
            lat = row["lat"]
            return pygeohash.encode(lat, lon, precision=self.geohash_precision)

        return df.apply(get_hash, axis=1)

    def _get_cell_geometry(
        self, gdf: pd.DataFrame, buffer_ratio: float = 0.0
    ) -> gpd.GeoSeries:
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
