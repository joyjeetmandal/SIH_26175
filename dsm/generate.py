import os
import argparse

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


# ============================================================
# DSM GENERATION
# ============================================================


def generate_dsm(agl_path, dem_path, output_path):
    """
    Generate DSM from:

        DSM = DEM + AGL

    The DEM is automatically reprojected/resampled onto the
    AGL raster grid.

    Parameters
    ----------
    agl_path : str
        Path to predicted AGL GeoTIFF.

    dem_path : str
        Path to DEM GeoTIFF.

    output_path : str
        Path for output DSM GeoTIFF.
    """

    print("=" * 70)
    print("DSM GENERATION")
    print("=" * 70)

    print("\nAGL:", agl_path)
    print("DEM:", dem_path)
    print("Output:", output_path)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.isfile(agl_path):
        raise FileNotFoundError(f"AGL file not found:\n{agl_path}")

    if not os.path.isfile(dem_path):
        raise FileNotFoundError(f"DEM file not found:\n{dem_path}")

    # --------------------------------------------------------
    # Open AGL
    # --------------------------------------------------------

    with rasterio.open(agl_path) as agl_src:
        agl = agl_src.read(1).astype(np.float32)

        agl_profile = agl_src.profile.copy()

        agl_crs = agl_src.crs
        agl_transform = agl_src.transform

        agl_width = agl_src.width
        agl_height = agl_src.height

        agl_nodata = agl_src.nodata

        print("\n" + "-" * 70)
        print("AGL INFORMATION")
        print("-" * 70)

        print("Shape:", agl.shape)
        print("CRS:", agl_crs)
        print("Resolution:", agl_src.res)
        print("Bounds:", agl_src.bounds)
        print("NoData:", agl_nodata)

    # --------------------------------------------------------
    # Validate AGL
    # --------------------------------------------------------

    if agl_crs is None:
        raise ValueError(
            "AGL GeoTIFF has no CRS.\n"
            "A georeferenced AGL raster is required "
            "for DSM generation."
        )

    # --------------------------------------------------------
    # Open DEM
    # --------------------------------------------------------

    with rasterio.open(dem_path) as dem_src:
        dem_crs = dem_src.crs
        dem_transform = dem_src.transform

        dem_width = dem_src.width
        dem_height = dem_src.height

        dem_nodata = dem_src.nodata

        print("\n" + "-" * 70)
        print("DEM INFORMATION")
        print("-" * 70)

        print("Shape:", (dem_height, dem_width))
        print("CRS:", dem_crs)
        print("Resolution:", dem_src.res)
        print("Bounds:", dem_src.bounds)
        print("NoData:", dem_nodata)

        if dem_crs is None:
            raise ValueError("DEM GeoTIFF has no CRS.")

        # ----------------------------------------------------
        # Create DEM array matching AGL grid
        # ----------------------------------------------------

        aligned_dem = np.full((agl_height, agl_width), np.nan, dtype=np.float32)

        print("\n" + "-" * 70)
        print("ALIGNING DEM TO AGL GRID")
        print("-" * 70)

        print("Target CRS:", agl_crs)
        print("Target shape:", (agl_height, agl_width))

        reproject(
            source=rasterio.band(dem_src, 1),
            destination=aligned_dem,
            src_transform=dem_transform,
            src_crs=dem_crs,
            dst_transform=agl_transform,
            dst_crs=agl_crs,
            src_nodata=dem_nodata,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    # --------------------------------------------------------
    # Create validity masks
    # --------------------------------------------------------

    valid_agl = np.isfinite(agl)

    if agl_nodata is not None:
        valid_agl &= agl != agl_nodata

    valid_dem = np.isfinite(aligned_dem)

    valid = valid_agl & valid_dem

    valid_pixels = int(valid.sum())

    total_pixels = agl.size

    print("\nValid pixels:", valid_pixels)
    print("Total pixels:", total_pixels)

    if valid_pixels == 0:
        raise ValueError("No overlapping valid pixels between AGL and DEM.")

    # --------------------------------------------------------
    # DSM = DEM + AGL
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("GENERATING DSM")
    print("-" * 70)

    dsm = np.full(agl.shape, np.nan, dtype=np.float32)

    dsm[valid] = aligned_dem[valid] + agl[valid]

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    valid_dsm = dsm[valid]

    print("\nDSM statistics:")

    print("Min:", float(np.min(valid_dsm)))

    print("Max:", float(np.max(valid_dsm)))

    print("Mean:", float(np.mean(valid_dsm)))

    print("Median:", float(np.median(valid_dsm)))

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------
    # Output GeoTIFF profile
    # --------------------------------------------------------

    profile = agl_profile.copy()

    profile.update(
        {
            "driver": "GTiff",
            "height": agl_height,
            "width": agl_width,
            "count": 1,
            "dtype": "float32",
            "nodata": np.nan,
            "compress": "lzw",
        }
    )

    # --------------------------------------------------------
    # Save DSM
    # --------------------------------------------------------

    print("\nSaving DSM...")

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(dsm, 1)

    print("\n" + "=" * 70)
    print("DSM GENERATED SUCCESSFULLY")
    print("=" * 70)

    print("Output:", output_path)
    print("CRS:", profile["crs"])
    print("Resolution:", profile["transform"])
    print("Shape:", dsm.shape)

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================


def main():

    parser = argparse.ArgumentParser(
        description=("Generate a DSM by combining predicted AGL with a DEM.")
    )

    parser.add_argument("--agl", required=True, help="Path to predicted AGL GeoTIFF")

    parser.add_argument("--dem", required=True, help="Path to DEM GeoTIFF")

    parser.add_argument(
        "--output", default="outputs/dsm/dsm.tif", help="Output DSM GeoTIFF path"
    )

    args = parser.parse_args()

    generate_dsm(agl_path=args.agl, dem_path=args.dem, output_path=args.output)


if __name__ == "__main__":
    main()
