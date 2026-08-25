from __future__ import annotations

import json
from pathlib import Path
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from kimodo.exports.bvh import bvh_to_kimodo_motion, save_motion_bvh
from kimodo.exports.motion_io import complete_motion_dict, save_kimodo_npz
from kimodo.skeleton.bvh import Bvh, SkeletonBvh, load_bvh_animation, parse_bvh_motion
from kimodo.skeleton.registry import build_skeleton

SRC=Path('attack003_v2_source/86_05.bvh')
OUT=Path('attack003_v2_stage'); OUT.mkdir(parents=True, exist_ok=True)
TARGET=Path('kimodo/kimodo/assets/skeletons/somaskel77/somaskel77_standard_tpose.bvh')
FPS=30.0
MAP={
'Hips':'Hips','Spine1':'LowerBack','Spine2':'Spine','Chest':'Spine1','Neck1':'Neck','Neck2':'Neck1','Head':'Head',
'LeftShoulder':'LeftShoulder','LeftArm':'LeftArm','LeftForeArm':'LeftForeArm','LeftHand':'LeftHand',
'RightShoulder':'RightShoulder','RightArm':'RightArm','RightForeArm':'RightForeArm','RightHand':'RightHand',
'LeftLeg':'LeftUpLeg','LeftShin':'LeftLeg','LeftFoot':'LeftFoot','LeftToeBase':'LeftToeBase',
'RightLeg':'RightUpLeg','RightShin':'RightLeg','RightFoot':'RightFoot','RightToeBase':'RightToeBase'}
EDGES=[('Hips','Spine1'),('Spine1','Spine2'),('Spine2','Chest'),('Chest','Neck1'),('Neck1','Neck2'),('Neck2','Head'),('Chest','LeftShoulder'),('LeftShoulder','LeftArm'),('LeftArm','LeftForeArm'),('LeftForeArm','LeftHand'),('Chest','RightShoulder'),('RightShoulder','RightArm'),('RightArm','RightForeArm'),('RightForeArm','RightHand'),('Hips','LeftLeg'),('LeftLeg','LeftShin'),('LeftShin','LeftFoot'),('Hips','RightLeg'),('RightLeg','RightShin'),('RightShin','RightFoot')]

def offset_len(sk,n): return float(np.linalg.norm(sk.name2bone[n].offset))

def render(pos,J,a,b,path,title):
    fs=np.linspace(a,b-1,8).round().astype(int)
    fig,axs=plt.subplots(2,len(fs),figsize=(16,5.8))
    for c,f in enumerate(fs):
        q=pos[f].copy(); q[:,0]-=q[J['Hips'],0]; q[:,2]-=q[J['Hips'],2]
        for r,(u,v,label) in enumerate([(0,1,'front'),(2,1,'side')]):
            ax=axs[r,c]
            for x,y in EDGES:
                z=q[[J[x],J[y]]]; ax.plot(z[:,u],z[:,v],lw=2)
            ax.scatter([q[J['RightHand'],u]],[q[J['RightHand'],v]],s=15)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f'{f/FPS:.2f}s {label}',fontsize=7)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)

text=SRC.read_text(errors='replace')
mocap=Bvh(text,backend='np'); srcfps=1.0/mocap.frame_time; step=int(round(srcfps/FPS))
sk=SkeletonBvh(); sk.load_from_bvh(str(SRC),mocap=mocap)
root,rots=load_bvh_animation(str(SRC),sk,mocap=mocap); root=np.asarray(root,np.float64); rots=np.asarray(rots,np.float64)
names=sk.get_bones_names(); S={n:i for i,n in enumerate(names)}
idx=np.arange(1,len(rots),step,dtype=int); rest=rots[0]; samp=rots[idx]
soma=build_skeleton(77); tnames=list(soma.bone_order_names); J={n:i for i,n in enumerate(tnames)}
loc=np.broadcast_to(np.eye(3),(len(idx),77,3,3)).copy()
for tn,sn in MAP.items(): loc[:,J[tn]]=rest[S[sn]].T[None]@samp[:,S[sn]]
neutral=soma.neutral_joints.detach().cpu().numpy().astype(np.float64)
def d(a,b): return float(np.linalg.norm(neutral[J[a]]-neutral[J[b]]))
srcleg=.5*(offset_len(sk,'LeftLeg')+offset_len(sk,'LeftFoot')+offset_len(sk,'RightLeg')+offset_len(sk,'RightFoot'))
tgtleg=.5*(d('LeftLeg','LeftShin')+d('LeftShin','LeftFoot')+d('RightLeg','RightShin')+d('RightShin','RightFoot'))
scale=tgtleg/srcleg
_,troot,_=parse_bvh_motion(str(TARGET)); base=troot[0].cpu().numpy().astype(np.float64)
rp=base[None]+(root[idx]-root[0])*scale
motion=complete_motion_dict(torch.from_numpy(loc).float(),torch.from_numpy(rp).float(),soma,FPS)
save_kimodo_npz(str(OUT/'86_05_SOMA77_30fps.npz'),motion)
save_motion_bvh(OUT/'86_05_SOMA77_30fps.bvh',motion['local_rot_mats'],motion['root_positions'],skeleton=soma,fps=FPS,standard_tpose=True)
pos=motion['posed_joints'].cpu().numpy(); hips=pos[:,J['Hips']]; rh=pos[:,J['RightHand']]-hips; lf=pos[:,J['LeftFoot']]; rf=pos[:,J['RightFoot']]

# Jump detection: pelvis vertical excursion relative to rolling low baseline + both feet rising.
y=hips[:,1]; basey=np.quantile(y,0.20); peaks,_=find_peaks(y,distance=30,prominence=0.06)
jumps=[]
for p in peaks:
    a=max(0,p-18); b=min(len(y),p+24)
    rise=float(y[p]-min(y[a],y[b-1]))
    footrise=float(min(lf[p,1]-np.min(lf[a:b,1]),rf[p,1]-np.min(rf[a:b,1])))
    if rise>0.06:
        jumps.append((rise+0.4*max(0,footrise),a,b,p,rise,footrise))
jumps=sorted(jumps,reverse=True)[:6]

# Down-chop detection: hand begins above shoulder/head zone and drops strongly over 0.4-0.8s.
chops=[]
for w in [12,15,18,21,24]:
    for a in range(0,len(rh)-w):
        b=a+w; dy=float(rh[b,1]-rh[a,1]); dx=float(rh[b,0]-rh[a,0]); dz=float(rh[b,2]-rh[a,2])
        if rh[a,1]>0.50 and dy<-0.45:
            score=(-dy)+0.15*abs(dz)-0.15*abs(dx)
            chops.append((score,a,b,dy,dx,dz,rh[a].copy(),rh[b].copy()))
chops=sorted(chops,reverse=True)[:8]

for i,c in enumerate(jumps[:4],1):
    _,a,b,p,rise,fr=c; render(pos,J,a,b,OUT/f'jump_{i}_f{a}_{b}_peak{p}.png',f'86_05 jump candidate {i} rise={rise:.3f}m footrise={fr:.3f}m')
for i,c in enumerate(chops[:5],1):
    _,a,b,dy,dx,dz,s,e=c; render(pos,J,a,b,OUT/f'chop_{i}_f{a}_{b}.png',f'86_05 downward hand candidate {i} dy={dy:.3f}m')

rep={'source':'CMU 86_05','fps':FPS,'frames':len(pos),'jump_candidates':[{'rank':i+1,'start':c[1],'end':c[2],'peak':c[3],'pelvis_rise_m':c[4],'both_feet_rise_m':c[5]} for i,c in enumerate(jumps)],'chop_candidates':[{'rank':i+1,'start':c[1],'end':c[2],'hand_drop_m':-c[3],'dx_m':c[4],'dz_m':c[5],'start_xyz':c[6].tolist(),'end_xyz':c[7].tolist()} for i,c in enumerate(chops)]}
(OUT/'86_05_design_candidates.json').write_text(json.dumps(rep,indent=2),encoding='utf-8')
print(json.dumps(rep,indent=2))
