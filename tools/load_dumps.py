from pathlib import Path
import numpy as np
import xarray as xr

from .convert_scale_letkf_var import convert_scale_letkf_var

_UNDEF = -9.99e33  # SCALE missing sentinel
Z_LEVELS = [   55.     ,   165.     ,   275.     ,   385.     ,   495.     ,
             608.08   ,   727.49245,   853.592  ,   986.75315,  1127.37135,
            1275.86415,  1432.67255,  1598.2623 ,  1773.1251 ,  1957.78025,
            2152.77615,  2358.6919 ,  2576.13905,  2805.7633 ,  3048.24655,
            3304.30895,  3574.7108 ,  3860.2551 ,  4161.7898 ,  4480.2102 ,
            4816.46215,  5171.5442 ,  5546.51075,  5942.47535,  6360.61375,
            6802.16795,  7268.4492 ,  7760.842  ,  8280.80855,  8829.89305,
            9409.7261 , 10022.0298 , 10668.62255, 11351.4243 , 12072.4629 ,
           12842.8018 , 13642.8018 , 14442.8018 , 15242.8018 , 16042.8013 ]

x_offset = {"pe" + str(i).zfill(6) : 50 + (100  * (i % 4)) * 320 for i in range(20)}
y_offset = {"pe" + str(i).zfill(6) : 50 + (100  * (i // 4)) * 256 for i in range(20)}

X_LEVELS =  {k: v+np.arange(320)*100 for k,v in x_offset.items()}
Y_LEVELS =  {k: v+np.arange(256)*100 for k,v in y_offset.items()}

def _read_state(path: Path) -> np.ndarray:
    with path.open("rb") as fh:
        nd = np.fromfile(fh, dtype=">i4", count=1)[0]
        dims = tuple(np.fromfile(fh, dtype=">i4", count=nd))
        data = np.fromfile(fh, dtype=">f8")
    size = int(np.prod(dims, dtype=np.int64))
    if data.size < size:
        raise ValueError(f"{path} truncated: expected {size} values, found {data.size}")
    if data.size > size:
        data = data[:size]
    return data.reshape(dims, order="F")


def _parse_meta(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    with path.open() as fh:
        for line in fh:
            if "=" not in line:
                continue
            key, value = line.strip().split("=", 1)
            if key in {"nlon", "nlat"}:
                out[key] = int(value)
    return out

def load_rank_members(
    dump_dir: str | Path,
    prefix: str,
    pe_tag: str | int
    ) -> xr.DataArray:
    mems = ["mem0001", "mem0002", "memmean"]
    x = [_read_state(dump_dir / f"{prefix}_{pe_tag}.{mem}.bin") for mem in mems]

    y = np.zeros((256, 320, 45, 3, 11))
    y_flat = y.reshape(-1, 45, 3, 11)
    
    y_flat[::3] = x[0]
    y_flat[1::3] = x[1]
    y_flat[2::3] = x[2]
    
    y = y_flat.reshape((256, 320, 45, 3, 11))
    
    dims=["y", "x", "z", "ens", "variable"]
    da = xr.DataArray(y, dims=dims,
                      coords={"variable":['U', 'V', 'W', 'T', 'P', 'QV', 'QC', 'QR', 'QI', 'QS', 'QG'],
                              "ens":["0001", "0002", "mean"],
                              "x":X_LEVELS[pe_tag],
                              "y":Y_LEVELS[pe_tag],
                              "z":Z_LEVELS
                             }
                      )
    
    da = da.where(da>-10**30)
    return da

def load_and_convert_rank_members(dump_dir, prefix, pe_tag):
    das = []
    for member in ["0001", "0002", "mean"]:
        fname = f"init_20210730-060030.000.{pe_tag}.nc"
        nc_path = dump_dir / ".." / "anal_f" / member / fname
        ds = xr.open_dataset(nc_path, engine="netcdf4")
        da_comp = convert_scale_letkf_var(ds)
        das.append(da_comp.expand_dims(ens=[member]))
    return xr.concat(das, dim="ens").transpose("y", "x", "z", "ens", "variable")