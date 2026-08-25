from __future__ import annotations

import json
import math
from pathlib import Path

import bpy
from mathutils import Vector

OUT = Path("artifacts")
PREVIEW = OUT / "qa_preview"
PREVIEW.mkdir(parents=True, exist_ok=True)

qa_glb = OUT / "qa_input" / "F01_uncompressed.glb"
glbs = [qa_glb] if qa_glb.exists() else sorted(OUT.glob("*.glb"))
if not glbs:
    raise RuntimeError("No generated GLB found in artifacts/")
glb = glbs[0]

# Clean scene and import generated animation.
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
result = bpy.ops.import_scene.gltf(filepath=str(glb.resolve()))
if "FINISHED" not in result:
    raise RuntimeError(f"GLB import failed: {result}")

scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = 720
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = False
scene.world.color = (0.055, 0.055, 0.065)
scene.render.fps = 30

armatures = [o for o in scene.objects if o.type == "ARMATURE"]
meshes = [o for o in scene.objects if o.type == "MESH"]
if not armatures:
    raise RuntimeError("Imported GLB has no armature")
if not meshes:
    raise RuntimeError("Imported GLB has no mesh")
arm = armatures[0]

action = arm.animation_data.action if arm.animation_data else None
if action:
    start = int(math.floor(action.frame_range[0]))
    end = int(math.ceil(action.frame_range[1]))
else:
    start, end = scene.frame_start, scene.frame_end
scene.frame_start, scene.frame_end = start, end

frames = sorted(set([
    start,
    round(start + (end-start)*0.25),
    round(start + (end-start)*0.50),
    round(start + (end-start)*0.75),
    end,
]))

def world_bounds(sample_frames):
    mn = Vector((1e9, 1e9, 1e9))
    mx = Vector((-1e9, -1e9, -1e9))
    for fr in sample_frames:
        scene.frame_set(fr)
        bpy.context.view_layer.update()
        for obj in meshes:
            for corner in obj.bound_box:
                p = obj.matrix_world @ Vector(corner)
                mn.x, mn.y, mn.z = min(mn.x,p.x), min(mn.y,p.y), min(mn.z,p.z)
                mx.x, mx.y, mx.z = max(mx.x,p.x), max(mx.y,p.y), max(mx.z,p.z)
    return mn, mx

mn, mx = world_bounds(frames)
center = (mn + mx) * 0.5
span = mx - mn
max_span = max(span.x, span.y, span.z, 1.0)

bpy.ops.object.camera_add()
cam = bpy.context.object
cam.data.type = "ORTHO"
cam.data.ortho_scale = max(span.z * 1.18, max(span.x, span.y) * 1.35, 2.2)
scene.camera = cam

def point_camera(pos: Vector):
    cam.location = pos
    cam.rotation_euler = (center - pos).to_track_quat('-Z', 'Y').to_euler()

for loc, energy, size in [
    (center + Vector((3.0,-4.0,4.5)), 1100, 5.0),
    (center + Vector((-4.0,-1.0,2.5)), 650, 4.0),
    (center + Vector((1.0,4.0,3.5)), 800, 4.0),
]:
    bpy.ops.object.light_add(type='AREA', location=loc)
    light = bpy.context.object
    light.data.energy = energy
    light.data.shape = 'DISK'
    light.data.size = size
    light.rotation_euler = (center - light.location).to_track_quat('-Z','Y').to_euler()

views = {
    "front": center + Vector((0, -max_span*3.0, 0)),
    "side": center + Vector((max_span*3.0, 0, 0)),
}
rendered = []
for fr in frames:
    scene.frame_set(fr)
    bpy.context.view_layer.update()
    for name, pos in views.items():
        point_camera(pos)
        path = PREVIEW / f"f{fr:04d}_{name}.png"
        scene.render.filepath = str(path.resolve())
        bpy.ops.render.render(write_still=True)
        if not path.exists():
            raise RuntimeError(f"Render did not produce {path}")
        rendered.append(str(path))

def bone_world_head(name: str):
    pb = arm.pose.bones.get(name)
    if pb is None:
        return None
    return arm.matrix_world @ pb.head

bone_names = [b.name for b in arm.pose.bones]
name_candidates = {
    "hips": ["mixamorig:Hips", "Hips"],
    "left_foot": ["mixamorig:LeftFoot", "LeftFoot"],
    "right_foot": ["mixamorig:RightFoot", "RightFoot"],
    "left_hand": ["mixamorig:LeftHand", "LeftHand"],
    "right_hand": ["mixamorig:RightHand", "RightHand"],
    "spine2": ["mixamorig:Spine2", "Spine2", "mixamorig:Spine1", "Spine1"],
}
def resolve(cands):
    for n in cands:
        if arm.pose.bones.get(n):
            return n
    return None
resolved = {k: resolve(v) for k,v in name_candidates.items()}

samples = []
for fr in range(start, end+1):
    scene.frame_set(fr)
    bpy.context.view_layer.update()
    row = {"frame": fr}
    for key, bname in resolved.items():
        p = bone_world_head(bname) if bname else None
        row[key] = list(map(float,p)) if p is not None else None
    samples.append(row)

def displacement(key):
    pts = [Vector(s[key]) for s in samples if s.get(key)]
    if not pts:
        return None
    base = pts[0]
    return max((p-base).length for p in pts)

report = {
    "source_glb": glb.name,
    "fps": scene.render.fps,
    "frame_start": start,
    "frame_end": end,
    "frame_count_inclusive": end-start+1,
    "duration_seconds_approx": (end-start)/scene.render.fps,
    "armature": arm.name,
    "bone_count": len(bone_names),
    "resolved_bones": resolved,
    "max_displacement_from_first_frame_m": {
        "hips": displacement("hips"),
        "left_foot": displacement("left_foot"),
        "right_foot": displacement("right_foot"),
        "left_hand": displacement("left_hand"),
        "right_hand": displacement("right_hand"),
    },
    "bounds": {"min": list(map(float,mn)), "max": list(map(float,mx))},
    "preview_frames": frames,
    "rendered": rendered,
    "bone_names": bone_names,
}
(OUT / "qa_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
