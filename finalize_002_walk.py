from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from kimodo.exports.bvh import bvh_to_kimodo_motion, save_motion_bvh
from kimodo.exports.motion_io import complete_motion_dict, save_kimodo_npz
from kimodo.skeleton.registry import build_skeleton

OUT = Path('walk002_out')
SRC = OUT/'002_W01_Walk_F_ShyWalk_SOMA77_30fps.npz'
FPS = 30.0
START = 125
END = 171            # virtual equivalent phase; stored clip is [125,171)
L = END - START      # 46 frames = 1.5333 s


def rot_angle_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    b=b.to(dtype=a.dtype,device=a.device)
    rel=a.transpose(-1,-2)@b
    tr=rel.diagonal(dim1=-2,dim2=-1).sum(-1)
    return torch.acos(((tr-1.0)*0.5).clamp(-1.0,1.0))


def distributed_loop_close(src: np.ndarray) -> np.ndarray:
    """Distribute endpoint->start SO(3) correction over one real gait cycle."""
    # src includes the virtual endpoint, shape [L+1,J,3,3].
    a=src[0]; z=src[-1]
    delta=np.swapaxes(z,-1,-2)@a
    rv=Rotation.from_matrix(delta.reshape(-1,3,3)).as_rotvec().reshape(delta.shape[:-2]+(3,))
    out=np.empty_like(src,dtype=np.float64)
    for k in range(len(src)):
        alpha=k/(len(src)-1)
        corr=Rotation.from_rotvec((rv*alpha).reshape(-1,3)).as_matrix().reshape(delta.shape)
        out[k]=src[k]@corr
    return out


def save_and_roundtrip(stem: str, local: torch.Tensor, root: torch.Tensor, soma):
    motion=complete_motion_dict(local,root,soma,FPS)
    npz=OUT/f'{stem}.npz'; bvh=OUT/f'{stem}.bvh'
    save_kimodo_npz(str(npz),motion)
    save_motion_bvh(bvh,motion['local_rot_mats'],motion['root_positions'],skeleton=soma,fps=FPS,standard_tpose=True)
    rt,rtfps=bvh_to_kimodo_motion(bvh,skeleton=soma,standard_tpose=True)
    if abs(rtfps-FPS)>0.02 or tuple(rt['local_rot_mats'].shape)!=tuple(motion['local_rot_mats'].shape):
        raise RuntimeError(f'{stem} roundtrip mismatch')
    re=rot_angle_between(motion['local_rot_mats'],rt['local_rot_mats'])
    pe=torch.linalg.norm(motion['root_positions']-rt['root_positions'].to(motion['root_positions'].dtype),dim=-1)
    return motion,npz,bvh,{
        'max_rotation_error_deg':float(torch.rad2deg(re).max()),
        'mean_rotation_error_deg':float(torch.rad2deg(re).mean()),
        'max_root_error_m':float(pe.max()),'mean_root_error_m':float(pe.mean())}


def seam_metrics(local_ext: torch.Tensor, root_ext: torch.Tensor, soma):
    # L+1 frames: the final frame is the virtual first frame of the next cycle.
    m=complete_motion_dict(local_ext,root_ext,soma,FPS)
    pos=m['posed_joints'].detach().cpu()
    loc=m['local_rot_mats'].detach().cpu()
    steps=torch.linalg.norm(pos[1:]-pos[:-1],dim=-1).mean(-1)
    rsteps=torch.rad2deg(rot_angle_between(loc[:-1],loc[1:])).mean(-1)
    return {
        'median_joint_position_step_m':float(steps[:-1].median()),
        'seam_joint_position_step_m':float(steps[-1]),
        'seam_position_vs_median_ratio':float(steps[-1]/steps[:-1].median().clamp_min(1e-9)),
        'median_rotation_step_deg':float(rsteps[:-1].median()),
        'seam_rotation_step_deg':float(rsteps[-1]),
        'seam_rotation_vs_median_ratio':float(rsteps[-1]/rsteps[:-1].median().clamp_min(1e-9)),
    }


data=np.load(SRC)
local=np.asarray(data['local_rot_mats'],np.float64)
root=np.asarray(data['root_positions'],np.float64)
if len(local)<=END: raise RuntimeError('Stage A clip shorter than selected gait cycle')

src_local=local[START:END+1].copy()
src_root=root[START:END+1].copy()
closed=distributed_loop_close(src_local)

# Remove only net vertical endpoint drift. Horizontal displacement is the real
# captured gait-cycle root motion and remains untouched in RootMotion output.
phase=np.linspace(0.0,1.0,L+1)
root_rm=src_root.copy()
root_rm[:,0]-=root_rm[0,0]; root_rm[:,2]-=root_rm[0,2]
y_drift=root_rm[-1,1]-root_rm[0,1]
root_rm[:,1]-=phase*y_drift
cycle_delta=root_rm[-1,[0,2]].copy()
travel=float(np.linalg.norm(cycle_delta)); speed=travel/(L/FPS)

# In-place companion: subtract the measured horizontal root progression over
# phase while retaining vertical/lateral gait sway around the trajectory.
root_ip=root_rm.copy()
root_ip[:,0]-=phase*cycle_delta[0]
root_ip[:,2]-=phase*cycle_delta[1]

soma=build_skeleton(77)
local_stored=torch.from_numpy(closed[:L]).float()
rm_root_stored=torch.from_numpy(root_rm[:L]).float()
ip_root_stored=torch.from_numpy(root_ip[:L]).float()

rm,rm_npz,rm_bvh,rm_rt=save_and_roundtrip('002_W01_Walk_F_RootMotion_SOMA77_30fps',local_stored,rm_root_stored,soma)
ip,ip_npz,ip_bvh,ip_rt=save_and_roundtrip('002_W01_Walk_F_InPlace_SOMA77_30fps',local_stored,ip_root_stored,soma)

rm_seam=seam_metrics(torch.from_numpy(closed).float(),torch.from_numpy(root_rm).float(),soma)
ip_seam=seam_metrics(torch.from_numpy(closed).float(),torch.from_numpy(root_ip).float(),soma)

# Contact-speed QA on RootMotion world joints. Kimodo contact channels are
# grouped by side; OR each side's two channels to be robust to heel/toe usage.
pos=rm['posed_joints'].detach().cpu(); contacts=rm['foot_contacts'].detach().cpu()
names=list(soma.bone_order_names); J={n:i for i,n in enumerate(names)}
lf=pos[:,J['LeftFoot']]; rf=pos[:,J['RightFoot']]
lf_speed=torch.linalg.norm(lf[1:]-lf[:-1],dim=-1)*FPS
rf_speed=torch.linalg.norm(rf[1:]-rf[:-1],dim=-1)*FPS
lc=(contacts[:-1,0]>0.5)|(contacts[:-1,1]>0.5)
rc=(contacts[:-1,2]>0.5)|(contacts[:-1,3]>0.5)

def contact_stats(sp,mask):
    vals=sp[mask]
    return {'frames':int(mask.sum()),'mean_speed_mps':float(vals.mean()) if len(vals) else None,'max_speed_mps':float(vals.max()) if len(vals) else None}

# How much loop-closing correction was required? This quantifies that cleanup
# remains small rather than re-authoring the capture.
orig=torch.from_numpy(src_local).float(); corr=torch.from_numpy(closed).float()
correction=torch.rad2deg(rot_angle_between(orig,corr))

report={
 'name':'002_W01_Walk_F','source':'CMU 91_20 ShyWalk -> native Kimodo SOMA77',
 'selected_stageA_frames':[START,END],'selected_source_seconds':[START/FPS,END/FPS],
 'stored_frames':L,'fps':FPS,'cycle_seconds':L/FPS,
 'root_motion':{'cycle_delta_xz_m':cycle_delta.tolist(),'distance_per_cycle_m':travel,'recommended_game_speed_mps':speed},
 'loop_cleanup':{'method':'distributed shortest-arc SO(3) endpoint correction; vertical endpoint detrend only','mean_rotation_correction_deg':float(correction.mean()),'max_rotation_correction_deg':float(correction.max())},
 'root_motion_seam_qa':rm_seam,'in_place_seam_qa':ip_seam,
 'root_motion_contact_qa':{'left':contact_stats(lf_speed,lc),'right':contact_stats(rf_speed,rc)},
 'official_roundtrip':{'root_motion':rm_rt,'in_place':ip_rt},
 'outputs':{'root_motion_npz':str(rm_npz),'root_motion_bvh':str(rm_bvh),'in_place_npz':str(ip_npz),'in_place_bvh':str(ip_bvh)},
 'notes':['The selected capture interval contains one complete shy walking gait cycle with arms naturally down.','No joint was manually posed.','RootMotion preserves real captured progression; InPlace subtracts only the measured horizontal cycle progression.','Final entity-mesh QA is performed against NVIDIA SOMA neutral 1.0 before tracker acceptance.']
}
(OUT/'002_final_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
