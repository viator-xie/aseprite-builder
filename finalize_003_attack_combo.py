from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation, Slerp

from kimodo.exports.bvh import bvh_to_kimodo_motion, save_motion_bvh
from kimodo.exports.motion_io import complete_motion_dict, save_kimodo_npz
from kimodo.skeleton.registry import build_skeleton

SRC = Path('stageA/attack003_stageA/02_09/02_09_Swordplay_SOMA77_30fps.npz')
OUT = Path('attack003_final'); OUT.mkdir(parents=True, exist_ok=True)
FPS = 30.0
SRC_START = 40
SRC_END_EXCLUSIVE = 154
SRC_HITS = [63, 89, 125]
OUT_FRAMES = 75  # 2.5 s at 30 FPS

soma = build_skeleton(77)
names = list(soma.bone_order_names); J = {n:i for i,n in enumerate(names)}
raw = np.load(SRC)
src_local = raw['local_rot_mats'][SRC_START:SRC_END_EXCLUSIVE].astype(np.float64)
src_root = raw['root_positions'][SRC_START:SRC_END_EXCLUSIVE].astype(np.float64)
N = len(src_local)
if N != SRC_END_EXCLUSIVE-SRC_START:
    raise RuntimeError(f'Unexpected source clip length: {N}')

# Time-compress the single continuous real mocap combo. Every joint follows
# shortest-arc quaternion interpolation; there is no pose-by-pose hand editing.
src_t = np.arange(N, dtype=np.float64)
out_t = np.linspace(0.0, N-1.0, OUT_FRAMES)
out_local = np.empty((OUT_FRAMES,77,3,3),dtype=np.float64)
for j in range(77):
    key = Rotation.from_matrix(src_local[:,j])
    out_local[:,j] = Slerp(src_t,key)(out_t).as_matrix()

out_root = np.empty((OUT_FRAMES,3),dtype=np.float64)
for k in range(3):
    out_root[:,k] = np.interp(out_t,src_t,src_root[:,k])

# Rebase the root horizontal origin. Root-motion version preserves the real
# 16-cm-ish lunge/stance drift; in-place version removes horizontal travel.
out_root_rm = out_root.copy()
out_root_rm[:,0] -= out_root_rm[0,0]
out_root_rm[:,2] -= out_root_rm[0,2]
out_root_ip = out_root_rm.copy()
out_root_ip[:,0] = 0.0
out_root_ip[:,2] = 0.0

local_t = torch.from_numpy(out_local).float()
rm_motion = complete_motion_dict(local_t,torch.from_numpy(out_root_rm).float(),soma,FPS)
ip_motion = complete_motion_dict(local_t,torch.from_numpy(out_root_ip).float(),soma,FPS)

files = {}
for tag,motion in [('RootMotion',rm_motion),('InPlace',ip_motion)]:
    npz = OUT/f'003_A01_Attack_Combo_3Hit_{tag}_SOMA77_30fps.npz'
    bvh = OUT/f'003_A01_Attack_Combo_3Hit_{tag}_SOMA77_30fps.bvh'
    save_kimodo_npz(str(npz),motion)
    save_motion_bvh(bvh,motion['local_rot_mats'],motion['root_positions'],skeleton=soma,fps=FPS,standard_tpose=True)
    rt,rt_fps = bvh_to_kimodo_motion(bvh,skeleton=soma,standard_tpose=True)
    if abs(rt_fps-FPS)>0.02 or tuple(rt['local_rot_mats'].shape)!=tuple(motion['local_rot_mats'].shape):
        raise RuntimeError(f'{tag}: official roundtrip mismatch')
    rel = motion['local_rot_mats'].transpose(-1,-2) @ rt['local_rot_mats'].to(motion['local_rot_mats'].dtype)
    tr = rel.diagonal(dim1=-2,dim2=-1).sum(-1)
    ang = torch.acos(((tr-1.0)*0.5).clamp(-1,1))
    rerr = torch.linalg.norm(motion['root_positions']-rt['root_positions'].to(motion['root_positions'].dtype),dim=-1)
    files[tag] = {
        'npz':str(npz),'bvh':str(bvh),
        'roundtrip_max_rotation_error_deg':float(torch.rad2deg(ang).max()),
        'roundtrip_max_root_error_m':float(rerr.max())
    }

# Map the three natural slash peaks into the time-compressed output.
def map_frame(f):
    u=(f-SRC_START)/(SRC_END_EXCLUSIVE-1-SRC_START)
    return int(round(u*(OUT_FRAMES-1)))
hits=[map_frame(f) for f in SRC_HITS]

# Motion QA from the root-motion version.
pos=rm_motion['posed_joints'].detach().cpu()
hips=pos[:,J['Hips']]; rh=pos[:,J['RightHand']]; lh=pos[:,J['LeftHand']]
rh_rel=rh-hips; lh_rel=lh-hips
rh_speed=torch.linalg.norm(rh_rel[1:]-rh_rel[:-1],dim=-1)*FPS
lh_speed=torch.linalg.norm(lh_rel[1:]-lh_rel[:-1],dim=-1)*FPS
hand_sep=torch.linalg.norm(rh-lh,dim=-1)
root_delta=rm_motion['root_positions'][-1,[0,2]]-rm_motion['root_positions'][0,[0,2]]

# Rotation purity: no bone scaling/shear introduced by resampling.
R=rm_motion['local_rot_mats']
I=torch.eye(3,dtype=R.dtype)
orth_err=torch.linalg.norm(R.transpose(-1,-2)@R-I,dim=(-2,-1))
det=torch.linalg.det(R)

# Contact sheet: start / each wind-up-hit-recovery region / end.
EDGES=[('Hips','Spine1'),('Spine1','Spine2'),('Spine2','Chest'),('Chest','Neck1'),('Neck1','Neck2'),('Neck2','Head'),('Chest','LeftShoulder'),('LeftShoulder','LeftArm'),('LeftArm','LeftForeArm'),('LeftForeArm','LeftHand'),('Chest','RightShoulder'),('RightShoulder','RightArm'),('RightArm','RightForeArm'),('RightForeArm','RightHand'),('Hips','LeftLeg'),('LeftLeg','LeftShin'),('LeftShin','LeftFoot'),('LeftFoot','LeftToeBase'),('Hips','RightLeg'),('RightLeg','RightShin'),('RightShin','RightFoot'),('RightFoot','RightToeBase')]
frames=sorted(set([0,max(0,hits[0]-6),hits[0],min(OUT_FRAMES-1,hits[0]+5),hits[1],min(OUT_FRAMES-1,hits[1]+5),hits[2],OUT_FRAMES-1]))
p=pos.numpy()
fig,axes=plt.subplots(2,len(frames),figsize=(2.15*len(frames),5.6))
for c,f in enumerate(frames):
    q=p[f].copy(); q[:,0]-=q[J['Hips'],0]; q[:,2]-=q[J['Hips'],2]
    for r,(a0,a1,label) in enumerate([(0,1,'front'),(2,1,'side')]):
        ax=axes[r,c]
        for u,v in EDGES:
            z=q[[J[u],J[v]]]; ax.plot(z[:,a0],z[:,a1],linewidth=2)
        ax.scatter([q[J['RightHand'],a0]],[q[J['RightHand'],a1]],s=22)
        ax.set_aspect('equal',adjustable='box'); ax.axis('off')
        marker=' HIT' if f in hits else ''
        ax.set_title(f'f{f} {f/FPS:.2f}s{marker}\n{label}',fontsize=8)
fig.suptitle('003 A01 — 3-Hit Right-Hand Knife/Sword Slash Combo (no weapon mesh)')
fig.tight_layout(); fig.savefig(OUT/'003_combo_contact_sheet.png',dpi=170); plt.close(fig)

report={
    'name':'003_A01_Attack_Combo_3Hit',
    'source':'CMU 02_09 swordplay -> native Kimodo SOMA77',
    'selection_reason':'02_09 has strong right-hand dominance and three natural consecutive slash peaks in one continuous capture; no cross-clip stitching.',
    'source_frames_30fps':[SRC_START,SRC_END_EXCLUSIVE],
    'source_duration_seconds':N/FPS,
    'output_frames':OUT_FRAMES,'fps':FPS,'duration_seconds':OUT_FRAMES/FPS,
    'time_compression_ratio':(OUT_FRAMES-1)/(N-1),
    'source_hit_frames':SRC_HITS,'output_hit_frames':hits,'output_hit_seconds':[h/FPS for h in hits],
    'root_motion':{'delta_xz_m':root_delta.tolist(),'distance_m':float(torch.linalg.norm(root_delta))},
    'qa':{
        'right_hand_speed_max_mps':float(rh_speed.max()),
        'left_hand_speed_max_mps':float(lh_speed.max()),
        'minimum_hand_separation_m':float(hand_sep.min()),
        'max_rotation_orthonormal_error':float(orth_err.max()),
        'determinant_min':float(det.min()),'determinant_max':float(det.max())
    },
    'outputs':files,
    'notes':[
        'No knife/sword mesh is added; the right hand is treated as the weapon hand.',
        'All three hits come from one continuous real mocap passage, so body transitions are authentic rather than stitched from unrelated poses.',
        'The only animation processing is SOMA77 retargeting, time compression via SO(3) Slerp, root rebasing/in-place conversion, and serialization.',
        'Finger/face joints remain neutral in this body-motion pass.'
    ]
}
(OUT/'003_final_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
