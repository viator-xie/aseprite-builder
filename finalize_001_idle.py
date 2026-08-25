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
BASE_START = 0
BASE_FRAMES = 75       # 2.5 s of the quietest source passage
BLEND_FRAMES = 15      # 0.5 s cyclic overlap
REPEATS = 2            # 60-frame seamless cycle x2 = 4.0 s period


def geodesic_blend_mats(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """Vectorized SO(3) interpolation from matrices a to b."""
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


if not SRC_NPZ.exists():
    raise FileNotFoundError(SRC_NPZ)

data = np.load(SRC_NPZ)
local = np.asarray(data["local_rot_mats"], dtype=np.float64)
root = np.asarray(data["root_positions"], dtype=np.float64)
if local.shape[1:] != (77, 3, 3):
    raise RuntimeError(f"Unexpected SOMA local rotation shape: {local.shape}")
if len(local) < BASE_START + BASE_FRAMES:
    raise RuntimeError("Source motion shorter than required quiet passage")

base_local = local[BASE_START:BASE_START + BASE_FRAMES].copy()
base_root = root[BASE_START:BASE_START + BASE_FRAMES].copy()
N = len(base_local)
B = BLEND_FRAMES
if not (0 < B < N // 2):
    raise RuntimeError(f"Invalid cyclic overlap B={B}, N={N}")

# Idle is authored in-place. Preserve the captured sway, but remove only the
# tiny net X/Z translation across the 2.5-second source passage. Y is untouched.
t = np.linspace(0.0, 1.0, N, dtype=np.float64)
for axis in (0, 2):
    drift = base_root[-1, axis] - base_root[0, axis]
    base_root[:, axis] -= t * drift
    base_root[:, axis] -= base_root[0, axis]

# Standard cyclic overlap-add. The cycle begins at original frame B, follows
# the untouched middle passage, then blends original tail -> original head.
# The last output frame therefore lands on original frame B-1 and the next
# loop starts on original frame B: a normal adjacent-frame transition.
body_local = base_local[B:N-B]
body_root = base_root[B:N-B]
seam_local = []
seam_root = []
for k in range(B):
    alpha = (k + 1) / B
    tail_i = N - B + k
    head_i = k
    seam_local.append(geodesic_blend_mats(base_local[tail_i], base_local[head_i], alpha))
    seam_root.append((1.0 - alpha) * base_root[tail_i] + alpha * base_root[head_i])

cycle_local = np.concatenate([body_local, np.stack(seam_local)], axis=0)
cycle_root = np.concatenate([body_root, np.stack(seam_root)], axis=0)
if len(cycle_local) != 60:
    raise RuntimeError(f"Expected 60-frame / 2.0 s seamless base cycle, got {len(cycle_local)}")

final_local = np.concatenate([cycle_local] * REPEATS, axis=0)
final_root = np.concatenate([cycle_root] * REPEATS, axis=0)
if len(final_local) != 120:
    raise RuntimeError(f"Expected 120-frame / 4.0 s final loop, got {len(final_local)}")

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

# Official Kimodo reader round-trip on the final deliverable.
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

# Seam quality: compare the repeat boundary and final->first loop closure with
# ordinary adjacent-frame motion. A clean loop should not contain an outlier.
pos = final_motion["posed_joints"].detach().cpu()
loc = final_motion["local_rot_mats"].detach().cpu()
adj_pos = torch.linalg.norm(pos[1:] - pos[:-1], dim=-1).mean(-1)
median_adj_pos = float(adj_pos.median())
repeat_boundary_pos = float(torch.linalg.norm(pos[60] - pos[59], dim=-1).mean())
final_boundary_pos = float(torch.linalg.norm(pos[0] - pos[-1], dim=-1).mean())
repeat_boundary_rot = float(torch.rad2deg(rot_angle_between(loc[59], loc[60])).mean())
final_boundary_rot = float(torch.rad2deg(rot_angle_between(loc[-1], loc[0])).mean())

# The two repeated cycles must be numerically identical by construction.
cycle_rot_repeat_err = float(torch.rad2deg(rot_angle_between(loc[:60], loc[60:])).max())
cycle_root_repeat_err = float(torch.linalg.norm(final_motion["root_positions"][:60] - final_motion["root_positions"][60:], dim=-1).max())

report = {
    "name": "001_timid_idle",
    "source": "CMU 137_28 Normal Wait -> native Kimodo SOMA77",
    "source_quiet_passage_seconds": [0.0, 2.5],
    "source_quiet_passage_frames_30fps": [0, 75],
    "cyclic_overlap_seconds": BLEND_FRAMES / FPS,
    "base_cycle_frames": 60,
    "base_cycle_seconds": 60 / FPS,
    "final_frames": len(final_local),
    "final_loop_period_seconds": len(final_local) / FPS,
    "fps": FPS,
    "root_motion_policy": "in-place X/Z linear detrend only; Y and captured body sway preserved",
    "official_roundtrip": {
        "max_rotation_error_deg": float(torch.rad2deg(rot_rt).max()),
        "mean_rotation_error_deg": float(torch.rad2deg(rot_rt).mean()),
        "max_root_position_error_m": float(root_rt.max()),
        "mean_root_position_error_m": float(root_rt.mean()),
    },
    "seam_qa": {
        "median_ordinary_joint_position_step_m": median_adj_pos,
        "repeat_boundary_joint_position_step_m": repeat_boundary_pos,
        "final_to_first_joint_position_step_m": final_boundary_pos,
        "repeat_boundary_rotation_step_deg_mean": repeat_boundary_rot,
        "final_to_first_rotation_step_deg_mean": final_boundary_rot,
        "repeat_boundary_position_vs_median_ratio": repeat_boundary_pos / max(median_adj_pos, 1e-9),
        "final_boundary_position_vs_median_ratio": final_boundary_pos / max(median_adj_pos, 1e-9),
        "cycle_repeat_max_rotation_error_deg": cycle_rot_repeat_err,
        "cycle_repeat_max_root_error_m": cycle_root_repeat_err,
    },
    "formal_outputs": {
        "npz": str(FINAL_NPZ),
        "bvh": str(FINAL_BVH),
    },
    "notes": [
        "No Blender re-export is used in the formal data path.",
        "No joint is hand-keyed; the body motion remains sourced from the real CMU capture.",
        "Only cyclic SO(3) overlap and in-place root detrending are applied for game-loop usability.",
    ],
}
REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
