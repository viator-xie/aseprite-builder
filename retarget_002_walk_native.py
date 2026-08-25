from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from kimodo.exports.bvh import bvh_to_kimodo_motion, save_motion_bvh
from kimodo.exports.motion_io import complete_motion_dict, save_kimodo_npz
from kimodo.skeleton.bvh import Bvh, SkeletonBvh, load_bvh_animation, parse_bvh_motion
from kimodo.skeleton.registry import build_skeleton

SRC = Path('walk002_source/91_20.bvh')
TARGET_TPOSE = Path('kimodo/kimodo/assets/skeletons/somaskel77/somaskel77_standard_tpose.bvh')
OUT = Path('walk002_out'); OUT.mkdir(parents=True, exist_ok=True)
OUT_FPS = 30.0

MAP = {
    'Hips':'Hips','Spine1':'LowerBack','Spine2':'Spine','Chest':'Spine1',
    'Neck1':'Neck','Neck2':'Neck1','Head':'Head',
    'LeftShoulder':'LeftShoulder','LeftArm':'LeftArm','LeftForeArm':'LeftForeArm','LeftHand':'LeftHand',
    'RightShoulder':'RightShoulder','RightArm':'RightArm','RightForeArm':'RightForeArm','RightHand':'RightHand',
    'LeftLeg':'LeftUpLeg','LeftShin':'LeftLeg','LeftFoot':'LeftFoot','LeftToeBase':'LeftToeBase',
    'RightLeg':'RightUpLeg','RightShin':'RightLeg','RightFoot':'RightFoot','RightToeBase':'RightToeBase',
}

def rot_angle_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    b = b.to(dtype=a.dtype, device=a.device)
    rel = a.transpose(-1,-2) @ b
    tr = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(((tr-1.0)*0.5).clamp(-1.0,1.0))

def offset_len(sk, name):
    return float(np.linalg.norm(sk.name2bone[name].offset))

text = SRC.read_text(encoding='utf-8', errors='replace')
mocap = Bvh(text, backend='np')
src_fps = 1.0 / mocap.frame_time
ratio = src_fps / OUT_FPS
step = int(round(ratio))
if step < 1 or abs(ratio-step) > 0.01:
    raise RuntimeError(f'Expected integer source->30fps ratio, got source_fps={src_fps}')

src_sk = SkeletonBvh(); src_sk.load_from_bvh(str(SRC), mocap=mocap)
src_root, src_rots = load_bvh_animation(str(SRC), src_sk, mocap=mocap)
src_root = np.asarray(src_root, np.float64); src_rots = np.asarray(src_rots, np.float64)
src_names = src_sk.get_bones_names(); Tsrc = len(src_rots)
if Tsrc < 20: raise RuntimeError(f'Source too short: {Tsrc}')

# CGSpeed MotionBuilder-friendly files contain a single inserted T-pose at f0.
# It is calibration only; real captured motion starts at f1.
sample_idx = np.arange(1, Tsrc, step, dtype=np.int64)
src_rest = src_rots[0]; src_sample = src_rots[sample_idx]

soma = build_skeleton(77)
tgt_names = list(soma.bone_order_names); ti = {n:i for i,n in enumerate(tgt_names)}; si = {n:i for i,n in enumerate(src_names)}
missing_src = [s for s in MAP.values() if s not in si]; missing_tgt = [t for t in MAP if t not in ti]
if missing_src or missing_tgt: raise RuntimeError(f'Mapping mismatch source={missing_src} target={missing_tgt}')

Tout = len(sample_idx)
tgt_local = np.broadcast_to(np.eye(3), (Tout,77,3,3)).copy()
for tname,sname in MAP.items():
    tgt_local[:,ti[tname]] = src_rest[si[sname]].T[None] @ src_sample[:,si[sname]]

src_leg = 0.5*(offset_len(src_sk,'LeftLeg')+offset_len(src_sk,'LeftFoot')+offset_len(src_sk,'RightLeg')+offset_len(src_sk,'RightFoot'))
neutral = soma.neutral_joints.detach().cpu().numpy().astype(np.float64)
def d(a,b): return float(np.linalg.norm(neutral[ti[a]]-neutral[ti[b]]))
tgt_leg = 0.5*(d('LeftLeg','LeftShin')+d('LeftShin','LeftFoot')+d('RightLeg','RightShin')+d('RightShin','RightFoot'))
scale = tgt_leg/src_leg
_, tpose_root, _ = parse_bvh_motion(str(TARGET_TPOSE))
base_root = tpose_root[0].detach().cpu().numpy().astype(np.float64)
root_positions = base_root[None] + (src_root[sample_idx]-src_root[0])*scale

motion = complete_motion_dict(torch.from_numpy(tgt_local).float(), torch.from_numpy(root_positions).float(), soma, OUT_FPS)
npz = OUT/'002_W01_Walk_F_Sh yWalk_SOMA77_30fps.npz'
# avoid a space typo in public filenames
npz = OUT/'002_W01_Walk_F_ShyWalk_SOMA77_30fps.npz'
bvh = OUT/'002_W01_Walk_F_ShyWalk_SOMA77_30fps.bvh'
save_kimodo_npz(str(npz), motion)
save_motion_bvh(bvh, motion['local_rot_mats'], motion['root_positions'], skeleton=soma, fps=OUT_FPS, standard_tpose=True)

rt, rt_fps = bvh_to_kimodo_motion(bvh, skeleton=soma, standard_tpose=True)
if abs(rt_fps-OUT_FPS)>0.02 or tuple(rt['local_rot_mats'].shape)!=tuple(motion['local_rot_mats'].shape):
    raise RuntimeError('Official Kimodo roundtrip mismatch')
rot_err = rot_angle_between(motion['local_rot_mats'],rt['local_rot_mats'])
root_err = torch.linalg.norm(motion['root_positions']-rt['root_positions'].to(motion['root_positions'].dtype),dim=-1)

pos = motion['posed_joints'].detach().cpu(); J = ti
hips = pos[:,J['Hips']]; lf=pos[:,J['LeftFoot']]; rf=pos[:,J['RightFoot']]
root_xz = hips[:,[0,2]]; delta = root_xz[-1]-root_xz[0]
travel = float(torch.linalg.norm(delta)); duration=(Tout-1)/OUT_FPS
forward = (delta/torch.linalg.norm(delta).clamp_min(1e-9)).tolist()

def speed(p): return torch.linalg.norm(p[1:]-p[:-1],dim=-1)*OUT_FPS
lf_sp=speed(lf); rf_sp=speed(rf)
# Low-speed foot frames are useful anchors for later same-foot gait-cycle extraction.
q_l=float(torch.quantile(lf_sp,0.2)); q_r=float(torch.quantile(rf_sp,0.2))
anchors_l=torch.nonzero(lf_sp<=q_l).flatten().tolist(); anchors_r=torch.nonzero(rf_sp<=q_r).flatten().tolist()

report={
 'name':'002_W01_Walk_F','source':'CMU 91_20 ShyWalk','source_frames':Tsrc,'source_fps':src_fps,
 'source_tpose_frame_calibration_only':0,'sampling_step':step,'output_frames':Tout,'output_fps':OUT_FPS,'duration_seconds':duration,
 'meters_per_source_unit':scale,'root_motion':{'horizontal_displacement_m':travel,'average_horizontal_speed_mps':travel/max(duration,1e-9),'principal_forward_xz':forward,'delta_xz_m':delta.tolist()},
 'foot_motion':{'left_mean_speed_mps':float(lf_sp.mean()),'right_mean_speed_mps':float(rf_sp.mean()),'left_low_speed_threshold_mps':q_l,'right_low_speed_threshold_mps':q_r,'left_low_speed_frame_sample':anchors_l[:40],'right_low_speed_frame_sample':anchors_r[:40]},
 'official_roundtrip':{'max_rotation_error_deg':float(torch.rad2deg(rot_err).max()),'mean_rotation_error_deg':float(torch.rad2deg(rot_err).mean()),'max_root_error_m':float(root_err.max()),'mean_root_error_m':float(root_err.mean())},
 'mapping':MAP,
 'formal_outputs':{'npz':str(npz),'bvh':str(bvh)},
 'notes':['Full real forward root motion is preserved in Stage A.','No gait cycle is cut or blended in Stage A.','Frame 0 T-pose is calibration only.','Unmapped SOMA face/finger joints remain identity for the body-motion pass.']
}
(OUT/'002_stageA_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
