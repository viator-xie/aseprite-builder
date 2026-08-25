from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from kimodo.exports.bvh import bvh_to_kimodo_motion, parse_bvh_motion, save_motion_bvh
from kimodo.exports.motion_io import complete_motion_dict, save_kimodo_npz
from kimodo.skeleton.bvh import Bvh, SkeletonBvh, load_bvh_animation
from kimodo.skeleton.registry import build_skeleton

SRC = Path("cmu_source/137_28.bvh")
TARGET_TPOSE = Path("kimodo/kimodo/assets/skeletons/somaskel77/somaskel77_standard_tpose.bvh")
OUT = Path("native_out")
OUT.mkdir(parents=True, exist_ok=True)

SRC_FPS = 120.0
OUT_FPS = 30.0
STEP = 4
WINDOW_SECONDS = 5.0
WINDOW = int(WINDOW_SECONDS * OUT_FPS)
WINDOW_STRIDE = int(0.5 * OUT_FPS)

# Target SOMA77 -> source CMU. Zero-offset CMU hip helper joints are skipped.
MAP = {
    "Hips": "Hips",
    "Spine1": "LowerBack",
    "Spine2": "Spine",
    "Chest": "Spine1",
    "Neck1": "Neck",
    "Neck2": "Neck1",
    "Head": "Head",
    "LeftShoulder": "LeftShoulder",
    "LeftArm": "LeftArm",
    "LeftForeArm": "LeftForeArm",
    "LeftHand": "LeftHand",
    "RightShoulder": "RightShoulder",
    "RightArm": "RightArm",
    "RightForeArm": "RightForeArm",
    "RightHand": "RightHand",
    "LeftLeg": "LeftUpLeg",
    "LeftShin": "LeftLeg",
    "LeftFoot": "LeftFoot",
    "LeftToeBase": "LeftToeBase",
    "RightLeg": "RightUpLeg",
    "RightShin": "RightLeg",
    "RightFoot": "RightFoot",
    "RightToeBase": "RightToeBase",
}


def load_source():
    text = SRC.read_text(encoding="utf-8", errors="replace")
    mocap = Bvh(text, backend="np")
    fps = 1.0 / mocap.frame_time
    if abs(fps - SRC_FPS) > 0.1:
        raise RuntimeError(f"Expected CMU source ~120 FPS; got {fps}")
    sk = SkeletonBvh()
    sk.load_from_bvh(str(SRC), mocap=mocap)
    root_trans, rots = load_bvh_animation(str(SRC), sk, mocap=mocap)
    names = sk.get_bones_names()
    return mocap, sk, names, np.asarray(root_trans, np.float64), np.asarray(rots, np.float64), fps


def offset_len(sk: SkeletonBvh, name: str) -> float:
    return float(np.linalg.norm(sk.name2bone[name].offset))


def rot_angle_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Geodesic angle between rotation matrices (...,3,3), radians."""
    rel = a.transpose(-1, -2) @ b
    tr = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    c = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(c)


mocap, src_sk, src_names, src_root, src_rots, detected_fps = load_source()
Tsrc = src_rots.shape[0]
if Tsrc != 3723:
    raise RuntimeError(f"Unexpected source frame count: {Tsrc}")

# Source frame 0 is the CGSpeed-inserted T-pose and is calibration only.
# Keep motion frames 1,5,9,... => exact 120/30 sampling, 931 frames / 31 s.
sample_idx = np.arange(1, Tsrc, STEP, dtype=np.int64)
src_sample = src_rots[sample_idx]
src_rest = src_rots[0]

soma = build_skeleton(77)
tgt_names = list(soma.bone_order_names)
if len(tgt_names) != 77:
    raise RuntimeError(f"Expected SOMA77, got {len(tgt_names)} joints")
src_idx = {n: i for i, n in enumerate(src_names)}
tgt_idx = {n: i for i, n in enumerate(tgt_names)}
missing_src = [s for s in MAP.values() if s not in src_idx]
missing_tgt = [t for t in MAP if t not in tgt_idx]
if missing_src or missing_tgt:
    raise RuntimeError(f"Mapping mismatch source={missing_src}, target={missing_tgt}")

Tout = len(sample_idx)
I = np.eye(3, dtype=np.float64)
tgt_local = np.broadcast_to(I, (Tout, 77, 3, 3)).copy()

# BVH has no separate joint-orient transform. Remove the inserted source T-pose
# local orientation, then apply that local animation delta to the zero-rotation
# SOMA standard T-pose. This stays entirely in the original Y-up BVH space.
for tname, sname in MAP.items():
    si = src_idx[sname]
    ti = tgt_idx[tname]
    r0_inv = src_rest[si].T
    tgt_local[:, ti] = r0_inv[None, :, :] @ src_sample[:, si]

# Scale source root trajectory to the fixed Kimodo SOMA body proportion using
# knee/ankle chains. Source units are not assumed; ratio is measured from rigs.
src_leg = 0.5 * (
    offset_len(src_sk, "LeftLeg") + offset_len(src_sk, "LeftFoot") +
    offset_len(src_sk, "RightLeg") + offset_len(src_sk, "RightFoot")
)
neutral_m = soma.neutral_joints.detach().cpu().numpy().astype(np.float64)
def dist(a, b):
    return float(np.linalg.norm(neutral_m[tgt_idx[a]] - neutral_m[tgt_idx[b]]))
tgt_leg_m = 0.5 * (
    dist("LeftLeg", "LeftShin") + dist("LeftShin", "LeftFoot") +
    dist("RightLeg", "RightShin") + dist("RightShin", "RightFoot")
)
meters_per_source_unit = tgt_leg_m / src_leg

# Use the official SOMA standard-T-pose BVH to get Kimodo's canonical pelvis
# world height (normally [0,1,0] m). Only motion delta is taken from CMU.
_, target_tpose_root, target_tpose_fps = parse_bvh_motion(str(TARGET_TPOSE))
base_root = target_tpose_root[0].detach().cpu().numpy().astype(np.float64)
root_delta = src_root[sample_idx] - src_root[0]
root_positions = base_root[None, :] + root_delta * meters_per_source_unit

local_t = torch.from_numpy(tgt_local).float()
root_t = torch.from_numpy(root_positions).float()
motion = complete_motion_dict(local_t, root_t, soma, OUT_FPS)

full_npz = OUT / "137_28_soma77_native_30fps.npz"
full_bvh = OUT / "137_28_soma77_native_30fps.bvh"
save_kimodo_npz(str(full_npz), motion)
save_motion_bvh(
    full_bvh,
    motion["local_rot_mats"],
    motion["root_positions"],
    skeleton=soma,
    fps=OUT_FPS,
    standard_tpose=True,
)

# Official round-trip parser validation. This catches wrong hierarchy, joint
# count/order, channel layout, units, frame rate, and serialization mistakes.
round_motion, round_fps = bvh_to_kimodo_motion(full_bvh, skeleton=soma, standard_tpose=True)
if abs(round_fps - OUT_FPS) > 0.02:
    raise RuntimeError(f"Official roundtrip FPS mismatch: {round_fps}")
if tuple(round_motion["local_rot_mats"].shape) != tuple(motion["local_rot_mats"].shape):
    raise RuntimeError(
        f"Official roundtrip shape mismatch {tuple(round_motion['local_rot_mats'].shape)} "
        f"vs {tuple(motion['local_rot_mats'].shape)}"
    )
rot_round_err = rot_angle_between(motion["local_rot_mats"], round_motion["local_rot_mats"])
root_round_err = torch.linalg.norm(motion["root_positions"] - round_motion["root_positions"], dim=-1)
roundtrip = {
    "max_rotation_error_deg": float(torch.rad2deg(rot_round_err).max()),
    "mean_rotation_error_deg": float(torch.rad2deg(rot_round_err).mean()),
    "max_root_position_error_m": float(root_round_err.max()),
    "mean_root_position_error_m": float(root_round_err.mean()),
}

# -------------------------------------------------------------------------
# 5-second idle-window scoring on official SOMA FK joint positions.
# Lower is better. We reward stable root/feet, restrained hands/head, hands
# hanging below the pelvis, and start/end similarity for looping.
# -------------------------------------------------------------------------
pos = motion["posed_joints"].detach().cpu()  # T,J,3, Y-up
locrot = motion["local_rot_mats"].detach().cpu()
J = tgt_idx
tracked = ["Hips", "LeftFoot", "RightFoot", "LeftHand", "RightHand", "Head"]

def speed_mps(x: torch.Tensor) -> float:
    if len(x) < 2:
        return 0.0
    return float(torch.linalg.norm(x[1:] - x[:-1], dim=-1).mean() * OUT_FPS)

def horiz_range(x: torch.Tensor) -> float:
    # X/Z ground plane in Kimodo Y-up.
    xz = x[:, [0, 2]]
    return float(torch.linalg.norm(xz.max(0).values - xz.min(0).values))

def score_window(start: int):
    end = start + WINDOW
    p = pos[start:end]
    hips = p[:, J["Hips"]]
    lf = p[:, J["LeftFoot"]]
    rf = p[:, J["RightFoot"]]
    lh = p[:, J["LeftHand"]]
    rh = p[:, J["RightHand"]]
    head = p[:, J["Head"]]

    root_speed = speed_mps(hips)
    foot_speed = 0.5 * (speed_mps(lf) + speed_mps(rf))
    hand_speed = 0.5 * (speed_mps(lh) + speed_mps(rh))
    head_speed = speed_mps(head)
    root_range = horiz_range(hips)

    # Natural relaxed hang: wrists generally below pelvis and not glued to it.
    down_l = hips[:,1] - lh[:,1]
    down_r = hips[:,1] - rh[:,1]
    down_pen = float((torch.abs(down_l - 0.16).mean() + torch.abs(down_r - 0.16).mean()) * 0.5)
    hand_dist = 0.5 * (
        torch.linalg.norm(lh - hips, dim=-1).mean() +
        torch.linalg.norm(rh - hips, dim=-1).mean()
    )
    hand_dist_pen = float(torch.abs(hand_dist - 0.28))

    ids = [J[n] for n in tracked]
    loop_pos = float(torch.linalg.norm(p[-1, ids] - p[0, ids], dim=-1).mean())
    loop_rot = float(torch.rad2deg(rot_angle_between(locrot[start, ids], locrot[end-1, ids])).mean())

    score = (
        5.0 * root_speed +
        4.0 * foot_speed +
        0.8 * hand_speed +
        0.8 * head_speed +
        3.0 * root_range +
        0.9 * down_pen +
        0.5 * hand_dist_pen +
        2.0 * loop_pos +
        0.01 * loop_rot
    )
    return {
        "start_frame_0based": start,
        "end_frame_exclusive_0based": end,
        "start_seconds": start / OUT_FPS,
        "end_seconds": end / OUT_FPS,
        "score": float(score),
        "root_speed_mps": root_speed,
        "foot_speed_mps": foot_speed,
        "hand_speed_mps": hand_speed,
        "head_speed_mps": head_speed,
        "root_horizontal_range_m": root_range,
        "hands_down_penalty_m": down_pen,
        "hand_distance_penalty_m": hand_dist_pen,
        "loop_position_error_m": loop_pos,
        "loop_rotation_error_deg": loop_rot,
    }

windows = [score_window(s) for s in range(0, Tout - WINDOW + 1, WINDOW_STRIDE)]
windows.sort(key=lambda x: x["score"])

# Greedy top-3, requiring starts >=3 seconds apart so QA candidates are not
# just the same quiet patch shifted by half a second.
selected = []
for w in windows:
    if all(abs(w["start_seconds"] - s["start_seconds"]) >= 3.0 for s in selected):
        selected.append(w)
    if len(selected) == 3:
        break

candidates = OUT / "candidates"
candidates.mkdir(exist_ok=True)
for rank, w in enumerate(selected, start=1):
    a = int(w["start_frame_0based"])
    b = int(w["end_frame_exclusive_0based"])
    local_clip = motion["local_rot_mats"][a:b].clone()
    root_clip = motion["root_positions"][a:b].clone()
    # Rebase only initial X/Z location; preserve vertical height and all drift.
    root_clip[:,0] -= root_clip[0,0]
    root_clip[:,2] -= root_clip[0,2]
    clip = complete_motion_dict(local_clip, root_clip, soma, OUT_FPS)
    stem = f"candidate_{rank}_s{w['start_seconds']:.1f}_e{w['end_seconds']:.1f}"
    save_kimodo_npz(str(candidates / f"{stem}.npz"), clip)
    save_motion_bvh(
        candidates / f"{stem}.bvh",
        clip["local_rot_mats"],
        clip["root_positions"],
        skeleton=soma,
        fps=OUT_FPS,
        standard_tpose=True,
    )

report = {
    "source_frames": Tsrc,
    "source_fps_detected": detected_fps,
    "source_tpose_frame_used_for_calibration_only": 0,
    "output_frames": Tout,
    "output_fps": OUT_FPS,
    "output_duration_seconds": (Tout - 1) / OUT_FPS,
    "meters_per_source_unit": meters_per_source_unit,
    "source_leg_measure_units": src_leg,
    "target_leg_measure_m": tgt_leg_m,
    "target_base_root_m": base_root.tolist(),
    "mapping": MAP,
    "official_roundtrip": roundtrip,
    "top_idle_windows": selected,
    "top_10_window_scores": windows[:10],
    "formal_output": {
        "npz": str(full_npz),
        "bvh": str(full_bvh),
    },
    "notes": [
        "Core retarget performed in original BVH Y-up coordinates; Blender is not in the formal data path.",
        "First CGSpeed T-pose is calibration only and omitted from output.",
        "Exact 120->30 Hz sampling takes source frames 1,5,9,... (zero-based after T-pose).",
        "Unmapped SOMA face/finger joints remain identity in standard T-pose for this body-motion pass.",
        "BVH is serialized by Kimodo official save_motion_bvh and read back by official bvh_to_kimodo_motion.",
    ],
}
(OUT / "native_retarget_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
