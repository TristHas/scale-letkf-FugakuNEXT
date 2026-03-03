from __future__ import annotations

import torch
import xtensor as xt

from sc23 import Grid, Tile
from pyletkf import ScaleLetkfConverter, GridInterpolator, LETKF, ObsOperator
from pyletkf.obs_op.radar import ref_operator, vr_operator, filter_ref, filter_vr
from pyletkf.logging import configure_rank_file_logging
from pyletkf.dist import init_distributed_from_env

CPU = torch.device("cpu")

OBS_TENSOR_FIELDS = (
    "hx_mean",
    "hdx",
    "innov",
    "err",
    "obs",
    "ri_global",
    "rj_global",
    "lev",
)

OBS_FIELD_DIMS = {
    "hx_mean": ("obs",),
    "hdx": ("obs", "ens"),
    "innov": ("obs",),
    "err": ("obs",),
    "obs": ("obs",),
    "ri_global": ("obs",),
    "rj_global": ("obs",),
    "lev": ("obs",),
}

DTYPE_CODE = {
    torch.float32: 0,
    torch.float64: 1,
    torch.int32: 2,
    torch.int64: 3,
}
CODE_DTYPE = {code: dtype for dtype, code in DTYPE_CODE.items()}
    
@dataclass
class DistributedContextConfig:
    grid: Grid
    state_conv: ScaleLetkfConverter
    obs_op: ObsOperator
    interp: GridInterpolator
    core: LETKF

def slice_len(slc, dim):
    start, stop, step = slc.indices(dim)
    return max(0, (stop - start + (step - 1)) // step)

class DistributedSC23Tile(SC23Tile):
    """
        Torch-distributed implementation that exchanges halos/observations on CPU.
    """
    STATE_TAG = 11
    HEIGHT_TAG = 13
    OBS_META_TAG = 21
    OBS_FIELD_META_TAG = 22
    OBS_DATA_TAG = 23

    def neighbor_ranks(self, k_neighbors=1):
        base = super().neighbor_ranks(k_neighbors)
        world_size = dist.get_world_size()
        return [entry for entry in base if entry[0] < world_size]

    def share_halos(self, ds, select_rank=None):
        neighbours = self._select_neighbours(select_rank)
        if not neighbours: return ds

        state = ds["state"].values
        height = ds["height"].values
        state_dev = state.device
        height_dev = height.device

        x_total = state.shape[3]
        y_total = state.shape[2]

        for peer, dx, dy in neighbours:
            send_x = self.lookup_send_x[dx]
            send_y = self.lookup_send_y[dy]
            recv_x = self.lookup_self_x[dx]
            recv_y = self.lookup_self_y[dy]

            state_send = state[:, :, send_y, send_x, :].contiguous().to(CPU)
            state_recv = torch.empty(
                (
                    state.shape[0], state.shape[1],
                    slice_len(recv_y, y_total),
                    slice_len(recv_x, x_total),
                    state.shape[4],
                ),
                dtype=state.dtype,
                device=CPU,
            )
            self._exchange_tensor(state_send, state_recv, peer, self.STATE_TAG)
            state[:, :, recv_y, recv_x, :] = state_recv.to(state_dev)

            height_send = height[send_y, send_x, :].contiguous().to(CPU)
            height_recv = torch.empty(
                (
                    slice_len(recv_y, y_total),
                    slice_len(recv_x, x_total),
                    height.shape[2],
                ),
                dtype=height.dtype,
                device=CPU,
            )
            self._exchange_tensor(height_send, height_recv, peer, self.HEIGHT_TAG)
            height[recv_y, recv_x, :] = height_recv.to(height_dev)

        return ds

    def share_obs(self, obs, neighbour_obs = None):
        neighbours = self._select_neighbours(None)
        local_payload, n_obs, n_ens, ens_coord = self._prepare_obs_payload(obs)
        datasets = [
            self._payload_to_dataset(local_payload, n_obs, n_ens, ens_coord)
        ]

        meta_tensor = torch.tensor([n_obs, n_ens], dtype=torch.int64, device=CPU)

        for peer, _, _ in neighbours:
            peer_meta = torch.empty_like(meta_tensor)
            self._exchange_tensor(meta_tensor, peer_meta, peer, self.OBS_META_TAG)
            peer_n_obs, peer_n_ens = map(int, peer_meta.tolist())
            recv_payload = self._exchange_obs_payload(
                peer, peer_n_obs, peer_n_ens, local_payload
            )
            if recv_payload is not None and peer_n_obs > 0:
                datasets.append(
                    self._payload_to_dataset(
                        recv_payload, peer_n_obs, peer_n_ens, ens_coord
                    )
                )

        if neighbour_obs:
            for ext in neighbour_obs:
                payload, ext_n_obs, ext_n_ens, _ = self._prepare_obs_payload(ext)
                if ext_n_obs > 0:
                    datasets.append(
                        self._payload_to_dataset(payload, ext_n_obs, ext_n_ens, ens_coord)
                    )

        if not datasets:
            return None
        combined = datasets[0] if len(datasets) == 1 else xt.concat(datasets, dim="obs")
        return combined.to(self.device)

    def _select_neighbours(self, select_rank):
        neighbours = self.neighbor_ranks()
        if select_rank is None:
            return neighbours
        selected = set(select_rank)
        return [entry for entry in neighbours if entry[0] in selected]

    def _exchange_tensor(self, send_tensor, recv_tensor, peer, tag):
        if self.rank < peer:
            dist.send(send_tensor, peer, tag=tag)
            dist.recv(recv_tensor, peer, tag=tag)
        else:
            dist.recv(recv_tensor, peer, tag=tag)
            dist.send(send_tensor, peer, tag=tag)

    def _prepare_obs_payload(self, obs):
        n_obs = obs.sizes.get("obs", 0)
        n_ens = obs.sizes.get("ens", 0)
        ens_coord = obs.coords.get("ens")
        payload = {}
        for key in OBS_TENSOR_FIELDS:
            source_key = key
            if key == "hx_mean" and key not in obs.data_vars and "hx" in obs.data_vars:
                source_key = "hx"
            data_var = obs.data_vars.get(source_key)
            if data_var is None:
                continue
            payload[key] = data_var.values.contiguous().to(CPU)
        return payload, n_obs, n_ens, ens_coord

    def _exchange_obs_payload(self, peer, peer_n_obs, peer_n_ens, local_payload):
        received: dict[str, torch.Tensor] = {}
        placeholder = torch.empty(0, dtype=torch.float32, device=CPU)
        for key in OBS_TENSOR_FIELDS:
            local_tensor = local_payload.get(key)
            has_local = 1 if local_tensor is not None else 0
            dtype_code = (
                self._dtype_code(local_tensor.dtype) if local_tensor is not None else -1
            )
            meta_send = torch.tensor([has_local, dtype_code], dtype=torch.int32, device=CPU)
            meta_recv = torch.empty_like(meta_send)
            self._exchange_tensor(meta_send, meta_recv, peer, self.OBS_FIELD_META_TAG)
            peer_flag = int(meta_recv[0].item())
            peer_dtype = self._dtype_from_code(int(meta_recv[1].item()))

            send_buf = local_tensor if local_tensor is not None else placeholder
            recv_shape = self._field_shape(key, peer_n_obs, peer_n_ens)
            if peer_flag and recv_shape is not None:
                recv_buf = torch.empty(recv_shape, dtype=peer_dtype, device=CPU)
            else:
                recv_buf = torch.empty(0, dtype=peer_dtype, device=CPU)
            self._exchange_tensor(send_buf, recv_buf, peer, self.OBS_DATA_TAG)
            if peer_flag and recv_shape is not None:
                received[key] = recv_buf
        return received if received else None

    def _payload_to_dataset(self, payload, n_obs, n_ens, ens_coord):
        coords = {"obs": xt.arange_index(n_obs, device=self.device)}
        if n_ens > 0:
            coords["ens"] = ens_coord if ens_coord is not None else xt.arange_index(
                n_ens, device=self.device
            )
        dataset = xt.Dataset(coords=coords)
        for key, dims in OBS_FIELD_DIMS.items():
            tensor = payload.get(key)
            if tensor is None:
                continue
            dataset[key] = (dims, tensor.to(self.device))
        return dataset

    def _field_shape(self, key, n_obs, n_ens):
        dims = OBS_FIELD_DIMS[key]
        shape = []
        for dim in dims:
            if dim == "obs":
                shape.append(n_obs)
            elif dim == "ens":
                shape.append(n_ens)
        return tuple(shape)

    @staticmethod
    def _dtype_code(dtype):
        if dtype not in DTYPE_CODE:
            raise ValueError(f"Unsupported dtype for obs sharing: {dtype}")
        return DTYPE_CODE[dtype]

    @staticmethod
    def _dtype_from_code(code):
        return CODE_DTYPE.get(code, torch.float32)

def distributed_tile_workflow(rank: int, config: DistributedContextConfig):
    """
    Run the SC23 pipeline on one rank using the distributed comm manager.
    """
    logger = configure_rank_file_logging(rank=rank, logger_name="test")

    ctx = config.grid[rank]
    ctx.device = torch.device(f"cuda:{rank}")

    with log_timing(logger, "Reading observations"):
        obs = ctx.read_obs().to(ctx.device)
        obs = ctx.populate_obs_coords(obs)
        obs = ctx.filter_obs_to_tile(obs)

    with log_timing(logger, "Loading state"):
        state = ctx.read_state().to(ctx.device)

    with log_timing(logger, "Sharing halos"):
        state = ctx.share_halos(state)

    with log_timing(logger, "State Conversion"):
        state = config.state_conv.model_to_da(state)

    with log_timing(logger, "Interpolation"):
        obs_state = config.interp.map_state_to_obs(obs.to(ctx.device), state)

    with log_timing(logger, "Halo Striping"):
        state = ctx.strip_halo(state)

    with log_timing(logger, "Observation Operator"):
        obs = config.obs_op(obs_state.to(ctx.device))

    with log_timing(logger, "Obs Filter"):
        obs = config.obs_op.filter(obs)
        logger.info("Remaining observations: %d", obs["obs"].shape[0])

    with log_timing(logger, "Obs Sharing"):
        obs = ctx.share_obs(obs)
        logger.info("Shared observations: %d", obs["obs"].shape[0])

    with log_timing(logger, "Obs mapping"):
        da_state = config.interp.map_obs_to_state(obs, state)

    with log_timing(logger, "LETKF Solver"):
        da_params = config.core.infer_update_params(da_state)

    with log_timing(logger, "LETKF Update"):
        output = config.core.apply_update(state, da_params)

    return output


def build_default_config(n_tiles_x=4, n_tiles_y=5) -> DistributedContextConfig:
    grid = SC23Grid(n_tiles_x=n_tiles_x, n_tiles_y=n_tiles_y)
    state_conv = ScaleLetkfConverter()
    obs_op = ObsOperator(
        {4001: ref_operator, 4002: vr_operator},
        {4001: filter_ref, 4002: filter_vr},
    )
    interp = GridInterpolator(grid)
    core = LETKF()
    return DistributedContextConfig(grid, state_conv, obs_op, interp, core)

if __name__ == "__main__":
    rank, _ = init_distributed_from_env()
    cfg = build_default_config()
    output = distributed_tile_workflow(rank, cfg)
    if output is None:
        print(f"[rank {rank}] No observations in tile.")
    else:
        print(f"[rank {rank}] Finished assimilation step.")
