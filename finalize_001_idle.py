from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from kimodo.exports.bvh import bvh_to_kimodo_motion, save_motion_bvh
from kimodo.exports.motion_io import complete_motion_dict, save_kimodo_npz
from kimodo.skeleton.registry import build_skeleton

OUT = Path("native_out")
SRC_NPZ = OUT / "137_28_soma77_native_30fps.npz"
FINAL_NPZ = OUT / "001_timid_idle_soma77_30fps.npz"
FINAL_BVH = OUT / "001_timid_idle_soma77_30fps.bvh"
REPORT = OUT / "001_timid_idle_loop_report.json"

FPS = 30.0
SOURCE_START = 0
SOURCE_FRAMES = 30       # 1.0 s: quietest real CMU passage
FINAL_FRAMES = 120       # 4.0 s game idle loop


def geodesic_interp_mats(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """Vectorized shortest-arc SO(3) interpolation from matrices a to b."""
    rel = np.swapaxes(a, -1, -2) @ b
    flat = rel.reshape(-1, 3, 3)
    rv = Rotation.from_matrix(flat).as_rotvec()
    inc = Rotation.from_rotvec(rv * float(alpha)).as_matrix().reshape(rel.shape)
    return a @ inc


def rot_angle_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    b = b.to(dtype=a.dtype, device=a.device)
    rel = a.transpose(-1, -2) @ b
    tr = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    c = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(c)


def periodic_resample_rotations(src: np.ndarray, n_out: int) -> np.ndarray:
    """Resample one discrete rotation cycle to n_out frames, including wrap edge."""
    n = len(src)
    out = np.empty((n_out,) + src.shape[1:], dtype=np.float64)
    for k in range(n_out):
        phase = k * n / n_out
        i0 = int(np.floor(phase)) % n
        frac = phase - np.floor(phase)
        i1 = (i0 + 1) % n
        out[k] = geodesic_interp_mats(src[i0], src[i1], frac)
    return out


def periodic_resample_vectors(src: np.ndarray, n_out: int) -> np.ndarray:
    n = len(src)
    out = np.empty((n_out,) + src.shape[1:], dtype=np.float64)
    for k in range(n_out):
        phase = k * n / n_out
        i0 = int(np.floor(phase)) % n
        frac = phase - np.floor(phase)
        i1 = (i0 + 1) % n
        out[k] = (1.0 - frac) * src[i0] + frac * src[i1]
    return out


if not SRC_NPZ.exists():
    raise FileNotFoundError(SRC_NPZ)

data = np.load(SRC_NPZ)
local = np.asarray(data["local_rot_mats"], dtype=np.float64)
root = np.asarray(data["root_positions"], dtype=np.float64)
if local.shape[1:] != (77, 3, 3):
    raise RuntimeError(f"Unexpected SOMA local rotation shape: {local.shape}")
if len(local) < SOURCE_START + SOURCE_FRAMES:
    raise RuntimeError("Source motion shorter than required 1-second quiet passage")

src_local = local[SOURCE_START:SOURCE_START + SOURCE_FRAMES].copy()
src_root = root[SOURCE_START:SOURCE_START + SOURCE_FRAMES].copy()

# Game idle is in-place. Remove only the tiny net horizontal translation of
# the real 1-second capture; keep vertical breathing/sway untouched.
t = np.linspace(0.0, 1.0, SOURCE_FRAMES, dtype=np.float64)
for axis in (0, 2):
    drift = src_root[-1, axis] - src_root[0, axis]
    src_root[:, axis] -= t * drift
    src_root[:, axis] -= src_root[0, axis]

# Slow the quiet real capture by exactly 4x using a periodic sampler. The
# frame 29 -> frame 0 edge is part of the same interpolation domain, so the
# loop closure has no separate hand-authored blend segment.
final_local = periodic_resample_rotations(src_local, FINAL_FRAMES)
final_root = periodic_resample_vectors(src_root, FINAL_FRAMES)

soma = build_skeleton(77)
final_motion = complete_motion_dict(
    torch.from_numpy(final_local).float(),
    torch.from_numpy(final_root).float(),
    soma,
    FPS,
)
save_kimodo_npz(str(FINAL_NPZ), final_motion)
save_motion_bvh(
    FINAL_BVH,
    final_motion["local_rot_mats"],
    final_motion["root_positions"],
    skeleton=soma,
    fps=FPS,
    standard_tpose=True,
)

# Official Kimodo round-trip on the exact deliverable.
round_motion, round_fps = bvh_to_kimodo_motion(FINAL_BVH, skeleton=soma, standard_tpose=True)
if abs(round_fps - FPS) > 0.02:
    raise RuntimeError(f"001 roundtrip FPS mismatch: {round_fps}")
if tuple(round_motion["local_rot_mats"].shape) != tuple(final_motion["local_rot_mats"].shape):
    raise RuntimeError("001 roundtrip shape mismatch")
rot_rt = rot_angle_between(final_motion["local_rot_mats"], round_motion["local_rot_mats"])
root_rt = torch.linalg.norm(
    final_motion["root_positions"] - round_motion["root_positions"].to(
        dtype=final_motion["root_positions"].dtype,
        device=final_motion["root_positions"].device,
    ),
    dim=-1,
)

# Quantitative loop QA on official SOMA FK positions.
pos = final_motion["posed_joints"].detach().cpu()
loc = final_motion["local_rot_mats"].detach().cpu()
step_pos = torch.linalg.norm(pos[1:] - pos[:-1], dim=-1).mean(-1)
median_step = float(step_pos.median())
max_step = float(step_pos.max())
closure_step = float(torch.linalg.norm(pos[0] - pos[-1], dim=-1).mean())
closure_rot = float(torch.rad2deg(rot_angle_between(loc[-1], loc[0])).mean())

names = list(soma.bone_order_names)
J = {n: i for i, n in enumerate(names)}

def horiz_range(name: str) -> float:
    p = pos[:, J[name]][:, [0, 2]]
    return float(torch.linalg.norm(p.max(0).values - p.min(0).values))

def mean_speed(name: str) -> float:
    p = pos[:, J[name]]
    return float(torch.linalg.norm(p[1:] - p[:-1], dim=-1).mean() * FPS)

lf = pos[:, J["LeftFoot"]]
rf = pos[:, J["RightFoot"]]
foot_closure_l = float(torch.linalg.norm(lf[0] - lf[-1]))
foot_closure_r = float(torch.linalg.norm(rf[0] - rf[-1]))

report = {
    "name": "001_timid_idle",
    "source": "CMU 137_28 Normal Wait -> native Kimodo SOMA77",
    "source_quiet_passage_seconds": [0.0, 1.0],
    "source_quiet_passage_frames_30fps": [0, 30],
    "source_processing": "X/Z linear detrend only; all joint motion and Y motion remain captured data",
    "periodic_resampling": "30 source frames -> 120 frames via shortest-arc SO(3) interpolation and cyclic vector interpolation",
    "final_frames": FINAL_FRAMES,
    "final_loop_period_seconds": FINAL_FRAMES / FPS,
    "fps": FPS,
    "official_roundtrip": {
        "max_rotation_error_deg": float(torch.rad2deg(rot_rt).max()),
        "mean_rotation_error_deg": float(torch.rad2deg(rot_rt).mean()),
        "max_root_position_error_m": float(root_rt.max()),
        "mean_root_position_error_m": float(root_rt.mean()),
    },
    "loop_qa": {
        "median_ordinary_joint_position_step_m": median_step,
        "max_ordinary_joint_position_step_m": max_step,
        "final_to_first_joint_position_step_m": closure_step,
        "final_to_first_rotation_step_deg_mean": closure_rot,
        "closure_position_vs_median_ratio": closure_step / max(median_step, 1e-9),
        "left_foot_horizontal_range_m": horiz_range("LeftFoot"),
        "right_foot_horizontal_range_m": horiz_range("RightFoot"),
        "hips_horizontal_range_m": horiz_range("Hips"),
        "left_foot_mean_speed_mps": mean_speed("LeftFoot"),
        "right_foot_mean_speed_mps": mean_speed("RightFoot"),
        "left_foot_final_to_first_step_m": foot_closure_l,
        "right_foot_final_to_first_step_m": foot_closure_r,
    },
    "formal_outputs": {"npz": str(FINAL_NPZ), "bvh": str(FINAL_BVH)},
    "notes": [
        "No Blender re-export is used in the formal data path.",
        "No joint is hand-keyed or manually posed.",
        "The animation body is the quietest real segment of CMU 137_28, slowed 4x for a restrained nervous idle.",
        "Entity-mesh skinning QA is performed separately against NVIDIA SOMA neutral 1.0 before tracker acceptance.",
    ],
}
REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
