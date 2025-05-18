import os
from pathlib import Path

import geopandas as gpd
import icechunk
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from icechunk.xarray import to_icechunk


def create_ds(root, key, id_to_area, date_range):
    local_divides = root[key]["divide_id"][:]
    runoff = root[key]["Qr"][:].astype(np.float32)
    local_divide_ids = [int(float(_id.split("-")[1])) for _id in local_divides]

    areas = np.zeros_like(local_divide_ids, dtype=np.float32)
    for idx, divide in enumerate(local_divide_ids):
        try:
            areas[idx] = id_to_area[divide]
        except KeyError as e:
            print(f"problem finding {divide} in Areas Dictionary")
    areas_array = areas * 1000 / 86400

    areas_array_reshaped = areas_array.reshape(-1, 1)
    streamflow_m3_s_data = runoff * areas_array_reshaped
    streamflow_m3_s_data = np.nan_to_num(
        streamflow_m3_s_data, nan=1e-6, posinf=1e-6, neginf=1e-6
    )
    mask = streamflow_m3_s_data == 0
    streamflow_m3_s_data[mask] = 1e-6

    ds = xr.Dataset(
        data_vars=dict(
            Qr=(["divide_id", "time"], streamflow_m3_s_data)
        ),
        coords=dict(
            divide_id=(["divide_id"], local_divides),
            time=(["time"], date_range)
        ),
        attrs=dict(description="Runoff outputs from dhbv2.0 at the HFv2.2 catchment scale"),
    )
    return ds


def main():
    profile_name = "CIROH_USER"  # Replace with your AWS credentials file profile name
    os.environ['AWS_PROFILE'] = profile_name

    file_path = Path("/projects/mhpi/yxs275/DM_output/HydroFabric_forward_1980_2019_From_dPL_local_daymet_v6_2v18_2_oneGPU_dynamic_k0_1980_1995_new")
    bucket="mhpi-spatial"
    prefix="hydrofabric_v2.2_dhbv_retrospective"
    storage_config = icechunk.s3_storage(
        bucket=bucket, prefix=prefix, region="us-east-2", from_env=True
    )
    repo = icechunk.Repository.create(storage_config)
    session = repo.writable_session("main")

    if file_path.exists() is False:
        raise FileNotFoundError(f"Cannot find: {file_path}")
    root = zarr.open_group(file_path)

    gdf = gpd.read_file("/projects/mhpi/data/hydrofabric/v2.2/conus_nextgen.gpkg", layer="flowpaths")
    gdf["_id"] = [int(float(_id.split("-")[1])) for _id in gdf["divide_id"]]
    gdf = gdf.set_index("_id")

    id_to_area = gdf["areasqkm"].to_dict()
    date_range = pd.date_range(
        start="01-01-1980",
        end="12-31-2019",
        freq="D",
    )
    
    print("Reading all zone data")
    zone_keys = [key for key in root.keys()]
    ds = create_ds(
        root=root,
        key=zone_keys[0],
        id_to_area=id_to_area,
        date_range=date_range
    )
    to_icechunk(ds, session)
    latest_snapshot = session.commit("initial commit")
    print(f"Data uploaded: {latest_snapshot}")
    session = repo.writable_session("main")
    for i, key in enumerate(zone_keys[1:]):
        ds = create_ds(
            root=root,
            key=key,
            id_to_area=id_to_area,
            date_range=date_range
        )
        to_icechunk(ds, session, append_dim='divide_id')
        if i % 5 == 0:
            msg = f"Uploaded groups {zone_keys[i-5:i]} to S3"
            latest_snapshot = session.commit(msg)
            print(f"{msg}: {latest_snapshot}")
            session = repo.writable_session("main")

if __name__ == "__main__":
    main()
