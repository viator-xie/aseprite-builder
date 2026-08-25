from pathlib import Path
import json
import bpy

FPS = 30
IN_DIR = Path('input')
OUT_DIR = Path('blender_out')
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = json.loads((IN_DIR/'003_final_report.json').read_text(encoding='utf-8'))
MARKS = [int(x) + 1 for x in REPORT['output_hit_frames']]

JOBS = [
    ('003_A01_Attack_Combo_3Hit_InPlace_SOMA77_30fps.bvh', 'A01_Combo_3Hit_InPlace_SOMA77', '003_A01_Combo_3Hit_InPlace.blend'),
    ('003_A01_Attack_Combo_3Hit_RootMotion_SOMA77_30fps.bvh', 'A01_Combo_3Hit_RootMotion_SOMA77', '003_A01_Combo_3Hit_RootMotion.blend'),
]


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.actions, bpy.data.armatures, bpy.data.meshes, bpy.data.curves, bpy.data.materials):
        for block in list(datablocks):
            if block.users == 0:
                datablocks.remove(block)


def build(src_name, object_name, out_name):
    src = IN_DIR / src_name
    if not src.exists():
        raise FileNotFoundError(src)
    clear_scene()
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.render.fps_base = 1.0
    scene.frame_start = 1
    scene.timeline_markers.clear()
    try:
        bpy.ops.preferences.addon_enable(module='io_anim_bvh')
    except Exception:
        pass
    result = bpy.ops.import_anim.bvh(filepath=str(src.resolve()), global_scale=1.0, frame_start=1, use_fps_scale=False, update_scene_fps=False, update_scene_duration=True, rotate_mode='NATIVE', axis_forward='-Z', axis_up='Y')
    if 'FINISHED' not in result:
        raise RuntimeError(f'BVH import failed: {result}')
    arms = [o for o in scene.objects if o.type == 'ARMATURE']
    if len(arms) != 1:
        raise RuntimeError(f'Expected one armature, got {len(arms)}')
    arm = arms[0]
    arm.name = object_name
    arm.data.name = object_name + '_Armature'
    arm.show_in_front = True
    arm.data.display_type = 'OCTAHEDRAL'
    if not arm.animation_data or not arm.animation_data.action:
        raise RuntimeError('Imported armature has no Action')
    arm.animation_data.action.name = object_name + '_Action'
    start, end = arm.animation_data.action.frame_range
    scene.frame_start = max(1, int(round(start)))
    scene.frame_end = int(round(end))
    scene.frame_set(scene.frame_start)
    scene.timeline_markers.new('COMBO_START', frame=scene.frame_start)
    for i, f in enumerate(MARKS, 1):
        scene.timeline_markers.new(f'ACTION_{i}', frame=f)
    scene.timeline_markers.new('RECOVERY_END', frame=scene.frame_end)
    bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0.0, 0.0, 0.0))
    ground = bpy.context.object
    ground.name = 'Ground_Guide'
    ground.display_type = 'WIRE'
    ground.hide_render = True
    scene['animation_name'] = object_name
    scene['fps'] = FPS
    scene['action_frames'] = ','.join(str(x) for x in MARKS)
    scene['source_motion'] = 'CMU 02_09 swordplay -> native SOMA77'
    arm['primary_hand'] = 'RightHand'
    arm['manual_joint_posing'] = False
    bpy.ops.object.select_all(action='DESELECT')
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    out = OUT_DIR / out_name
    bpy.ops.wm.save_as_mainfile(filepath=str(out.resolve()), compress=True)
    print(f'SAVED {out} frames={scene.frame_start}-{scene.frame_end} marks={MARKS}')

for job in JOBS:
    build(*job)
print('003 Blender combo build complete')
