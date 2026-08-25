from pathlib import Path
import bpy

FPS = 30
IN_DIR = Path('input')
OUT_DIR = Path('blender_out')
OUT_DIR.mkdir(parents=True, exist_ok=True)

JOBS = [
    ('002_W01_Walk_F_InPlace_SOMA77_30fps.bvh', 'W01_Walk_F_InPlace_SOMA77', '002_W01_Walk_F_InPlace.blend'),
    ('002_W01_Walk_F_RootMotion_SOMA77_30fps.bvh', 'W01_Walk_F_RootMotion_SOMA77', '002_W01_Walk_F_RootMotion.blend'),
]


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.actions, bpy.data.armatures, bpy.data.meshes, bpy.data.curves, bpy.data.materials):
        for block in list(datablocks):
            if block.users == 0:
                datablocks.remove(block)


def import_and_save(src_name: str, object_name: str, out_name: str):
    src = IN_DIR / src_name
    if not src.exists():
        raise FileNotFoundError(src)

    clear_scene()
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.render.fps_base = 1.0
    scene.frame_start = 1

    try:
        bpy.ops.preferences.addon_enable(module='io_anim_bvh')
    except Exception:
        pass

    result = bpy.ops.import_anim.bvh(
        filepath=str(src.resolve()),
        global_scale=1.0,
        frame_start=1,
        use_fps_scale=False,
        update_scene_fps=False,
        update_scene_duration=True,
        rotate_mode='NATIVE',
        axis_forward='-Z',
        axis_up='Y',
    )
    if 'FINISHED' not in result:
        raise RuntimeError(f'BVH import failed: {result}')

    arms = [o for o in scene.objects if o.type == 'ARMATURE']
    if len(arms) != 1:
        raise RuntimeError(f'Expected exactly one armature, got {len(arms)}')
    arm = arms[0]
    arm.name = object_name
    arm.data.name = object_name + '_Armature'
    arm.show_in_front = True
    arm.data.display_type = 'OCTAHEDRAL'

    if not arm.animation_data or not arm.animation_data.action:
        raise RuntimeError('Imported armature has no Action')
    action = arm.animation_data.action
    action.name = object_name + '_Action'

    start, end = action.frame_range
    scene.frame_start = max(1, int(round(start)))
    scene.frame_end = int(round(end))
    scene.frame_set(scene.frame_start)

    # Ground guide only; it does not affect animation.
    bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0.0, 0.0, 0.0))
    ground = bpy.context.object
    ground.name = 'Ground_Guide'
    ground.display_type = 'WIRE'
    ground.hide_render = True

    # Small metadata for quick inspection in Blender.
    scene['animation_name'] = object_name
    scene['fps'] = FPS
    scene['source_bvh'] = src_name
    scene['loop_expected'] = True
    arm['motion_source'] = 'CMU 91_20 ShyWalk -> native SOMA77'
    arm['retarget_policy'] = 'No manual joint posing; automated native SOMA77 retarget'

    # Select armature on save so the user lands on the useful object.
    bpy.ops.object.select_all(action='DESELECT')
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm

    out = OUT_DIR / out_name
    bpy.ops.wm.save_as_mainfile(filepath=str(out.resolve()), compress=True)
    print(f'SAVED {out} frames={scene.frame_start}-{scene.frame_end} fps={scene.render.fps}')


for job in JOBS:
    import_and_save(*job)

print('002 Blender animation build complete')
