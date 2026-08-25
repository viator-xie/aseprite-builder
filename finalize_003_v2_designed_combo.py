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

FPS=30.0
OUT=Path('attack003_v2_final'); OUT.mkdir(parents=True, exist_ok=True)
soma=build_skeleton(77); names=list(soma.bone_order_names); J={n:i for i,n in enumerate(names)}

src09=next(Path('input_stageA').rglob('02_09_Swordplay_SOMA77_30fps.npz'))
src08=next(Path('input_stageA').rglob('02_08_Swordplay_SOMA77_30fps.npz'))
src86=next(Path('input_jump').rglob('86_05_SOMA77_30fps.npz'))

def load(p):
    d=np.load(p)
    return np.asarray(d['local_rot_mats'],np.float64), np.asarray(d['root_positions'],np.float64)
R09,P09=load(src09); R08,P08=load(src08); R86,P86=load(src86)

def resample_rots(R, n):
    if len(R)==n: return R.copy()
    old=np.linspace(0,1,len(R)); new=np.linspace(0,1,n)
    out=np.empty((n,R.shape[1],3,3),np.float64)
    for j in range(R.shape[1]):
        s=Slerp(old,Rotation.from_matrix(R[:,j]))
        out[:,j]=s(new).as_matrix()
    return out

def resample_pos(P,n):
    old=np.linspace(0,1,len(P)); new=np.linspace(0,1,n)
    return np.stack([np.interp(new,old,P[:,k]) for k in range(3)],axis=-1)

def segment(R,P,a,b,n):
    rr=resample_rots(R[a:b],n); pp=resample_pos(P[a:b],n)
    pp=pp-pp[0]
    return rr,pp

def blend_concat(A,AP,B,BP,nblend=5):
    # Rebase B's root to A's current endpoint, then SO(3) crossfade a few frames.
    BP=BP-BP[0]+AP[-1]
    if nblend<=0: return np.concatenate([A,B]),np.concatenate([AP,BP])
    n=min(nblend,len(A),len(B))
    pre=A[:-n]; preP=AP[:-n]; post=B[n:]; postP=BP[n:]
    blendR=[]; blendP=[]
    for i in range(n):
        t=(i+1)/(n+1)
        pair=np.stack([A[-n+i],B[i]],axis=0)
        frame=np.empty_like(pair[0])
        for j in range(pair.shape[1]):
            frame[j]=Slerp([0,1],Rotation.from_matrix(pair[:,j]))([t]).as_matrix()[0]
        blendR.append(frame); blendP.append((1-t)*AP[-n+i]+t*BP[i])
    return np.concatenate([pre,np.asarray(blendR),post]),np.concatenate([preP,np.asarray(blendP),postP])

# HIT 1 — explicit right-high -> left-low diagonal slash.
# 02_09 was measured: f140->160, dx +0.569 m, dy -0.678 m in hand/hips space.
R1,P1=segment(R09,P09,134,164,22)
# Keep first action mostly planted, only a small body advance.
P1[:,0]=np.linspace(0.0,0.02,len(P1)); P1[:,2]=np.linspace(0.0,-0.06,len(P1)); P1[:,1]-=P1[0,1]

# HIT 2 — explicit left -> right horizontal sweep from a DIFFERENT capture (02_08).
# Measured f76->96: dx -0.742 m, dy -0.019 m. Add a designed 0.65 m forward lunge.
R2,P2=segment(R08,P08,70,102,23)
t=np.linspace(0,1,len(P2)); ease=t*t*(3-2*t)
P2[:,0]=0.015*np.sin(np.pi*t)
P2[:,2]=-0.65*ease
P2[:,1]-=P2[0,1]

# HIT 3 — REAL jump lower body + REAL overhead/downward upper body from CMU 86_05.
# Jump f333:375, pelvis rise ~0.437 m, both feet rise ~0.302 m.
# Downward-arm f549:567, right hand drop ~0.97 m.
N3=34
Rjump,Pjump=segment(R86,P86,333,375,N3)
Rchop,Pchop=segment(R86,P86,545,572,N3)
R3=Rjump.copy()
upper=['Spine1','Spine2','Chest','Neck1','Neck2','Head','LeftShoulder','LeftArm','LeftForeArm','LeftHand','RightShoulder','RightArm','RightForeArm','RightHand']
for n in upper: R3[:,J[n]]=Rchop[:,J[n]]
# Preserve real vertical jump. Normalize start Y; add slight forward commitment on landing.
P3=Pjump.copy(); P3[:,0]-=P3[0,0]; P3[:,2]-=P3[0,2]; P3[:,1]-=P3[0,1]
t=np.linspace(0,1,N3); P3[:,2]+=-0.16*(t*t*(3-2*t))

R12,P12=blend_concat(R1,P1,R2,P2,5)
Rall,Pall=blend_concat(R12,P12,R3,P3,5)
# Rebase to SOMA neutral root height.
base_y=float(soma.neutral_joints[0,1].item())
Pall[:,1]+=base_y

# In-place keeps the designed jump height but removes horizontal translation.
Pip=Pall.copy(); Pip[:,0]=Pall[0,0]; Pip[:,2]=Pall[0,2]

def save_variant(label,P):
    m=complete_motion_dict(torch.from_numpy(Rall).float(),torch.from_numpy(P).float(),soma,FPS)
    npz=OUT/f'003_A01_Designed_Combo_3Hit_V2_{label}_SOMA77_30fps.npz'
    bvh=OUT/f'003_A01_Designed_Combo_3Hit_V2_{label}_SOMA77_30fps.bvh'
    save_kimodo_npz(str(npz),m); save_motion_bvh(bvh,m['local_rot_mats'],m['root_positions'],skeleton=soma,fps=FPS,standard_tpose=True)
    rt,rfps=bvh_to_kimodo_motion(bvh,skeleton=soma,standard_tpose=True)
    rel=m['local_rot_mats'].transpose(-1,-2)@rt['local_rot_mats'].to(m['local_rot_mats'].dtype)
    tr=rel.diagonal(dim1=-2,dim2=-1).sum(-1); ang=torch.acos(((tr-1)*.5).clamp(-1,1))
    re=torch.linalg.norm(m['root_positions']-rt['root_positions'].to(m['root_positions'].dtype),dim=-1)
    return m,npz,bvh,float(torch.rad2deg(ang).max()),float(re.max())

mrm,npzrm,bvhrm,rerr,perr=save_variant('RootMotion',Pall)
mip,npzip,bvhip,rerri,perri=save_variant('InPlace',Pip)
pos=mrm['posed_joints'].cpu().numpy(); hips=pos[:,J['Hips']]; rh=pos[:,J['RightHand']]-hips; lf=pos[:,J['LeftFoot']]; rf=pos[:,J['RightFoot']]

# Hard design QA using actual final joint trajectories.
# Locate approximate boundaries after 5-frame crossfades.
b1=len(R1)-5; b2=b1+len(R2)-5
# Attack 1: beginning high on character-right (-X), ending low on character-left (+X).
a1s=max(0,2); a1e=min(len(rh)-1,b1+2)
d1=rh[a1e]-rh[a1s]
# Attack 2: left(+X) to right(-X), near-horizontal, plus strong forward root displacement.
a2s=max(0,b1+1); a2e=min(len(rh)-1,b2+2); d2=rh[a2e]-rh[a2s]
# Attack 3: true airborne vertical excursion and overhead-to-downward hand path.
a3s=max(0,b2); a3e=len(rh)-1; seg3hips=hips[a3s:a3e+1]; jump_rise=float(seg3hips[:,1].max()-min(seg3hips[0,1],seg3hips[-1,1]))
seg3lf=lf[a3s:a3e+1]; seg3rf=rf[a3s:a3e+1]; feet_rise=float(min(seg3lf[:,1].max()-seg3lf[:,1].min(),seg3rf[:,1].max()-seg3rf[:,1].min()))
# hand drop from its highest point in attack3 to later minimum
r3=rh[a3s:a3e+1]; hi=int(np.argmax(r3[:,1])); hand_drop=float(r3[hi,1]-np.min(r3[hi:,1]))
forward2=float(Pall[a2e,2]-Pall[a2s,2])
qa={
 'attack1_right_high_to_left_low':{'dx_m':float(d1[0]),'dy_m':float(d1[1]),'pass':bool(d1[0]>0.35 and d1[1]<-0.35)},
 'attack2_left_to_right_horizontal_lunge':{'dx_m':float(d2[0]),'dy_m':float(d2[1]),'forward_delta_z_m':forward2,'pass':bool(d2[0]<-0.35 and abs(d2[1])<0.30 and forward2<-0.35)},
 'attack3_jump_overhead_down':{'pelvis_rise_m':jump_rise,'feet_vertical_range_m':feet_rise,'right_hand_drop_m':hand_drop,'pass':bool(jump_rise>0.22 and feet_rise>0.15 and hand_drop>0.45)},
}
if not all(x['pass'] for x in qa.values()): raise RuntimeError('Designed combo hard QA failed: '+json.dumps(qa,indent=2))

# Contact sheet at designed beats.
EDGES=[('Hips','Spine1'),('Spine1','Spine2'),('Spine2','Chest'),('Chest','Neck1'),('Neck1','Neck2'),('Neck2','Head'),('Chest','LeftShoulder'),('LeftShoulder','LeftArm'),('LeftArm','LeftForeArm'),('LeftForeArm','LeftHand'),('Chest','RightShoulder'),('RightShoulder','RightArm'),('RightArm','RightForeArm'),('RightForeArm','RightHand'),('Hips','LeftLeg'),('LeftLeg','LeftShin'),('LeftShin','LeftFoot'),('Hips','RightLeg'),('RightLeg','RightShin'),('RightShin','RightFoot')]
frames=np.unique(np.clip(np.array([0,5,b1-3,b1+3,b2-4,b2+2,a3s+8,a3s+16,a3s+24,len(pos)-1]),0,len(pos)-1))
fig,axs=plt.subplots(2,len(frames),figsize=(2.1*len(frames),6))
for c,f in enumerate(frames):
    q=pos[f].copy(); q[:,0]-=hips[f,0]; q[:,2]-=hips[f,2]
    for rr,(u,v,label) in enumerate([(0,1,'front'),(2,1,'side')]):
        ax=axs[rr,c]
        for a,b in EDGES:
            z=q[[J[a],J[b]]]; ax.plot(z[:,u],z[:,v],lw=2)
        ax.scatter([q[J['RightHand'],u]],[q[J['RightHand'],v]],s=20)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f'f{f} {f/FPS:.2f}s {label}',fontsize=7)
fig.suptitle('003 V2 — designed 3-hit: diagonal slash -> forward reverse sweep -> jumping overhead chop')
fig.tight_layout(); fig.savefig(OUT/'003_v2_designed_contact_sheet.png',dpi=160); plt.close(fig)

# approximate action markers based on measured source strike centers after resampling
hit1=13; hit2=b1+12; hit3=a3s+25
report={'name':'003_A01_Designed_Combo_3Hit_V2','fps':FPS,'frames':len(Rall),'duration_s':len(Rall)/FPS,
'design':['HIT1 right-high -> left-low diagonal','HIT2 left -> right horizontal with 0.65m forward lunge','HIT3 overhead raise + real jump + downward chop'],
'sources':{'hit1':'CMU 02_09 swordplay f134-164','hit2':'CMU 02_08 swordplay f70-102','hit3_lower':'CMU 86_05 real jump f333-375','hit3_upper':'CMU 86_05 real downward arm/chop f545-572'},
'action_frames':[hit1,hit2,hit3],'action_seconds':[hit1/FPS,hit2/FPS,hit3/FPS],'hard_design_qa':qa,
'rootmotion_total_xz_m':[float(Pall[-1,0]-Pall[0,0]),float(Pall[-1,2]-Pall[0,2])],
'roundtrip':{'rootmotion_max_rot_deg':rerr,'rootmotion_max_root_m':perr,'inplace_max_rot_deg':rerri,'inplace_max_root_m':perri},
'outputs':[str(npzrm),str(bvhrm),str(npzip),str(bvhip)]}
(OUT/'003_v2_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
