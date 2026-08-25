from pathlib import Path
import json
import bpy

FPS=30
IN=Path('input')
OUT=Path('blender_v2'); OUT.mkdir(parents=True, exist_ok=True)
report=json.loads((IN/'003_v2_report.json').read_text(encoding='utf-8'))
markers=report['action_frames']
JOBS=[
('003_A01_Designed_Combo_3Hit_V2_InPlace_SOMA77_30fps.bvh','003_A01_Designed_Combo_3Hit_V2_InPlace.blend','Designed_Combo_V2_InPlace'),
('003_A01_Designed_Combo_3Hit_V2_RootMotion_SOMA77_30fps.bvh','003_A01_Designed_Combo_3Hit_V2_RootMotion.blend','Designed_Combo_V2_RootMotion')]

def clear():
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
    for blocks in (bpy.data.actions,bpy.data.armatures,bpy.data.meshes,bpy.data.curves,bpy.data.materials):
        for b in list(blocks):
            if b.users==0: blocks.remove(b)

def build(srcname,outname,objname):
    clear(); sc=bpy.context.scene; sc.render.fps=FPS; sc.render.fps_base=1.0; sc.frame_start=1
    try: bpy.ops.preferences.addon_enable(module='io_anim_bvh')
    except Exception: pass
    src=IN/srcname
    r=bpy.ops.import_anim.bvh(filepath=str(src.resolve()),global_scale=1.0,frame_start=1,use_fps_scale=False,update_scene_fps=False,update_scene_duration=True,rotate_mode='NATIVE',axis_forward='-Z',axis_up='Y')
    if 'FINISHED' not in r: raise RuntimeError(r)
    arms=[o for o in sc.objects if o.type=='ARMATURE']
    if len(arms)!=1: raise RuntimeError(f'expected one armature, got {len(arms)}')
    arm=arms[0]; arm.name=objname; arm.show_in_front=True; arm.data.display_type='OCTAHEDRAL'
    act=arm.animation_data.action; act.name=objname+'_Action'
    a,b=act.frame_range; sc.frame_start=max(1,int(round(a))); sc.frame_end=int(round(b))
    # Source action frames are zero-based; Blender imported frame starts at 1.
    for i,f in enumerate(markers,1): sc.timeline_markers.new(f'HIT_{i}',frame=int(f)+1)
    sc['animation_name']='003_A01_Designed_Combo_3Hit_V2'
    sc['fps']=FPS
    sc['design_hit_1']='right-high to left-low diagonal'
    sc['design_hit_2']='left-to-right horizontal + forward lunge'
    sc['design_hit_3']='overhead + real airborne jump + downward chop'
    arm['motion_sources']='CMU 02_09 + 02_08 + 86_05'
    arm['weapon_mesh_added']=False
    bpy.ops.mesh.primitive_plane_add(size=5.0,location=(0,0,0)); g=bpy.context.object; g.name='Ground_Guide'; g.display_type='WIRE'; g.hide_render=True
    bpy.ops.object.select_all(action='DESELECT'); arm.select_set(True); bpy.context.view_layer.objects.active=arm; sc.frame_set(sc.frame_start)
    bpy.ops.wm.save_as_mainfile(filepath=str((OUT/outname).resolve()),compress=True)

for j in JOBS: build(*j)
print('003 v2 Blender build complete')
