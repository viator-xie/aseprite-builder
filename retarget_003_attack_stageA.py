from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from kimodo.exports.bvh import bvh_to_kimodo_motion, save_motion_bvh
from kimodo.exports.motion_io import complete_motion_dict, save_kimodo_npz
from kimodo.skeleton.bvh import Bvh, SkeletonBvh, load_bvh_animation, parse_bvh_motion
from kimodo.skeleton.registry import build_skeleton

SRC_DIR = Path('attack003_source')
OUT = Path('attack003_stageA'); OUT.mkdir(parents=True, exist_ok=True)
TARGET_TPOSE = Path('kimodo/kimodo/assets/skeletons/somaskel77/somaskel77_standard_tpose.bvh')
SOURCES = ['02_07', '02_08', '02_09']
OUT_FPS = 30.0

MAP = {
    'Hips':'Hips','Spine1':'LowerBack','Spine2':'Spine','Chest':'Spine1',
    'Neck1':'Neck','Neck2':'Neck1','Head':'Head',
    'LeftShoulder':'LeftShoulder','LeftArm':'LeftArm','LeftForeArm':'LeftForeArm','LeftHand':'LeftHand',
    'RightShoulder':'RightShoulder','RightArm':'RightArm','RightForeArm':'RightForeArm','RightHand':'RightHand',
    'LeftLeg':'LeftUpLeg','LeftShin':'LeftLeg','LeftFoot':'LeftFoot','LeftToeBase':'LeftToeBase',
    'RightLeg':'RightUpLeg','RightShin':'RightLeg','RightFoot':'RightFoot','RightToeBase':'RightToeBase',
}

EDGES = [
    ('Hips','Spine1'),('Spine1','Spine2'),('Spine2','Chest'),('Chest','Neck1'),('Neck1','Neck2'),('Neck2','Head'),
    ('Chest','LeftShoulder'),('LeftShoulder','LeftArm'),('LeftArm','LeftForeArm'),('LeftForeArm','LeftHand'),
    ('Chest','RightShoulder'),('RightShoulder','RightArm'),('RightArm','RightForeArm'),('RightForeArm','RightHand'),
    ('Hips','LeftLeg'),('LeftLeg','LeftShin'),('LeftShin','LeftFoot'),('LeftFoot','LeftToeBase'),
    ('Hips','RightLeg'),('RightLeg','RightShin'),('RightShin','RightFoot'),('RightFoot','RightToeBase')
]


def rot_angle_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    b = b.to(dtype=a.dtype, device=a.device)
    rel = a.transpose(-1,-2) @ b
    tr = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(((tr-1.0)*0.5).clamp(-1.0,1.0))


def offset_len(sk, name):
    return float(np.linalg.norm(sk.name2bone[name].offset))


def render_candidate(pos: np.ndarray, J: dict[str,int], start: int, end: int, peak: int, path: Path, title: str):
    # six poses over the window, front and side. Root horizontal position is rebased per pose for readability.
    frames = np.linspace(start, max(start, end-1), 6).round().astype(int)
    fig, axes = plt.subplots(2, len(frames), figsize=(14, 5.4))
    for c, f in enumerate(frames):
        q = pos[f].copy()
        q[:,0] -= q[J['Hips'],0]
        q[:,2] -= q[J['Hips'],2]
        for r,(ax0,ax1,label) in enumerate([(0,1,'front'),(2,1,'side')]):
            ax = axes[r,c]
            for u,v in EDGES:
                z=q[[J[u],J[v]]]
                ax.plot(z[:,ax0], z[:,ax1], linewidth=2)
            ax.scatter([q[J['RightHand'],ax0]],[q[J['RightHand'],ax1]],s=16)
            ax.set_aspect('equal',adjustable='box'); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'f{f} {f/OUT_FPS:.2f}s {label}',fontsize=7)
    fig.suptitle(title + f' | peak f{peak} {peak/OUT_FPS:.2f}s')
    fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)


def process_source(stem: str):
    src = SRC_DIR / f'{stem}.bvh'
    text = src.read_text(encoding='utf-8', errors='replace')
    mocap = Bvh(text, backend='np')
    src_fps = 1.0 / mocap.frame_time
    ratio = src_fps / OUT_FPS
    step = int(round(ratio))
    if step < 1 or abs(ratio-step) > 0.01:
        raise RuntimeError(f'{stem}: expected integer source->30fps ratio, got {src_fps}')

    src_sk = SkeletonBvh(); src_sk.load_from_bvh(str(src), mocap=mocap)
    src_root, src_rots = load_bvh_animation(str(src), src_sk, mocap=mocap)
    src_root=np.asarray(src_root,np.float64); src_rots=np.asarray(src_rots,np.float64)
    src_names=src_sk.get_bones_names(); Tsrc=len(src_rots)
    if Tsrc < 40: raise RuntimeError(f'{stem}: source too short {Tsrc}')

    # Bruce Hahn / CGSpeed BVH files use frame 0 as the added MotionBuilder-compatible T-pose.
    # It is calibration only; captured swordplay starts at frame 1.
    sample_idx=np.arange(1,Tsrc,step,dtype=np.int64)
    src_rest=src_rots[0]; src_sample=src_rots[sample_idx]

    soma=build_skeleton(77)
    names=list(soma.bone_order_names); J={n:i for i,n in enumerate(names)}; S={n:i for i,n in enumerate(src_names)}
    missing_src=[s for s in MAP.values() if s not in S]; missing_tgt=[t for t in MAP if t not in J]
    if missing_src or missing_tgt: raise RuntimeError(f'{stem}: mapping mismatch source={missing_src} target={missing_tgt}')

    Tout=len(sample_idx)
    tgt_local=np.broadcast_to(np.eye(3),(Tout,77,3,3)).copy()
    for tname,sname in MAP.items():
        tgt_local[:,J[tname]] = src_rest[S[sname]].T[None] @ src_sample[:,S[sname]]

    src_leg=0.5*(offset_len(src_sk,'LeftLeg')+offset_len(src_sk,'LeftFoot')+offset_len(src_sk,'RightLeg')+offset_len(src_sk,'RightFoot'))
    neutral=soma.neutral_joints.detach().cpu().numpy().astype(np.float64)
    def d(a,b): return float(np.linalg.norm(neutral[J[a]]-neutral[J[b]]))
    tgt_leg=0.5*(d('LeftLeg','LeftShin')+d('LeftShin','LeftFoot')+d('RightLeg','RightShin')+d('RightShin','RightFoot'))
    scale=tgt_leg/src_leg
    _,tpose_root,_=parse_bvh_motion(str(TARGET_TPOSE))
    base_root=tpose_root[0].detach().cpu().numpy().astype(np.float64)
    root=base_root[None]+(src_root[sample_idx]-src_root[0])*scale

    motion=complete_motion_dict(torch.from_numpy(tgt_local).float(),torch.from_numpy(root).float(),soma,OUT_FPS)
    full_dir=OUT/stem; full_dir.mkdir(exist_ok=True)
    npz=full_dir/f'{stem}_Swordplay_SOMA77_30fps.npz'; bvh=full_dir/f'{stem}_Swordplay_SOMA77_30fps.bvh'
    save_kimodo_npz(str(npz),motion)
    save_motion_bvh(bvh,motion['local_rot_mats'],motion['root_positions'],skeleton=soma,fps=OUT_FPS,standard_tpose=True)

    rt,rt_fps=bvh_to_kimodo_motion(bvh,skeleton=soma,standard_tpose=True)
    if abs(rt_fps-OUT_FPS)>0.02 or tuple(rt['local_rot_mats'].shape)!=tuple(motion['local_rot_mats'].shape):
        raise RuntimeError(f'{stem}: official Kimodo roundtrip mismatch')
    rot_err=rot_angle_between(motion['local_rot_mats'],rt['local_rot_mats'])
    root_err=torch.linalg.norm(motion['root_positions']-rt['root_positions'].to(motion['root_positions'].dtype),dim=-1)

    pos=motion['posed_joints'].detach().cpu().numpy()
    hips=pos[:,J['Hips']]
    rh=pos[:,J['RightHand']]-hips
    rf=pos[:,J['RightForeArm']]-hips
    lh=pos[:,J['LeftHand']]-hips
    def sp(x):
        v=np.linalg.norm(np.diff(x,axis=0),axis=-1)*OUT_FPS
        return np.r_[v,v[-1] if len(v) else 0.0]
    rhs=sp(rh); rfs=sp(rf); lhs=sp(lh)
    energy=0.75*rhs+0.25*rfs

    # Sword slash candidate peaks: right-hand/forearm motion, at least 0.55 s apart.
    prominence=max(float(np.quantile(energy,0.60))*0.35,0.05)
    peaks,_=find_peaks(energy,distance=max(12,int(0.55*OUT_FPS)),prominence=prominence)
    if len(peaks)==0:
        peaks=np.array([int(np.argmax(energy))],dtype=int)
    ranked=sorted(peaks.tolist(),key=lambda p:float(energy[p]),reverse=True)[:5]

    cdir=full_dir/'candidates'; cdir.mkdir(exist_ok=True)
    candidates=[]
    for rank,p in enumerate(ranked,1):
        a=max(0,p-14); b=min(Tout,p+28)
        if b-a<20: continue
        local_clip=motion['local_rot_mats'][a:b].clone()
        root_clip=motion['root_positions'][a:b].clone()
        root_clip[:,0]-=root_clip[0,0]; root_clip[:,2]-=root_clip[0,2]
        clip=complete_motion_dict(local_clip,root_clip,soma,OUT_FPS)
        cstem=f'{stem}_cand{rank}_f{a}_{b}_peak{p}'
        cbvh=cdir/f'{cstem}.bvh'; cnpz=cdir/f'{cstem}.npz'; cpng=cdir/f'{cstem}.png'
        save_kimodo_npz(str(cnpz),clip)
        save_motion_bvh(cbvh,clip['local_rot_mats'],clip['root_positions'],skeleton=soma,fps=OUT_FPS,standard_tpose=True)
        render_candidate(pos,J,a,b,p,cpng,f'003 attack candidate {stem} #{rank}')
        candidates.append({
            'rank':rank,'peak_frame':p,'peak_seconds':p/OUT_FPS,'start_frame':a,'end_frame_exclusive':b,
            'duration_seconds':(b-a)/OUT_FPS,'right_hand_peak_speed_mps':float(rhs[p]),
            'right_forearm_peak_speed_mps':float(rfs[p]),'left_hand_speed_at_peak_mps':float(lhs[p]),
            'energy':float(energy[p]),'bvh':str(cbvh),'npz':str(cnpz),'preview':str(cpng)
        })

    report={
        'source':stem,'description':'CMU swordplay','source_frames':Tsrc,'source_fps':src_fps,
        'source_tpose_frame_calibration_only':0,'sampling_step':step,'output_frames':Tout,'output_fps':OUT_FPS,
        'duration_seconds':(Tout-1)/OUT_FPS,'meters_per_source_unit':scale,
        'official_roundtrip':{
            'max_rotation_error_deg':float(torch.rad2deg(rot_err).max()),
            'mean_rotation_error_deg':float(torch.rad2deg(rot_err).mean()),
            'max_root_error_m':float(root_err.max()),'mean_root_error_m':float(root_err.mean())
        },
        'motion_stats':{
            'right_hand_speed_max_mps':float(rhs.max()),'right_hand_speed_p95_mps':float(np.quantile(rhs,0.95)),
            'left_hand_speed_max_mps':float(lhs.max())
        },
        'top_slash_candidates':candidates,
        'notes':['No weapon mesh is added. Right hand is treated as the hypothetical knife hand.',
                 'No manual bone posing. Stage A only retargets real mocap and extracts high-speed slash candidates.',
                 'Finger/face joints remain neutral in this body-motion pass.']
    }
    (full_dir/f'{stem}_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

all_reports=[]
for stem in SOURCES:
    all_reports.append(process_source(stem))

summary={'name':'003_A01_Attack_Combo_3Hit','stage':'A','sources':all_reports}
(OUT/'003_stageA_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
