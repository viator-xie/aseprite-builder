from __future__ import annotations

import json
import math
from pathlib import Path

import bpy
from mathutils import Vector

SRC = Path("cmu_source/137_28.bvh")
TPOSE = Path("kimodo/kimodo/assets/skeletons/somaskel77/somaskel77_standard_tpose.bvh")
OUT = Path("retarget_out")
OUT.mkdir(parents=True, exist_ok=True)
PREVIEW = OUT / "preview"
PREVIEW.mkdir(exist_ok=True)

# CGSpeed's MotionBuilder-friendly CMU release inserts one T-pose frame before
# the untouched CMU motion. We use that frame *only* as the source rest-pose
# calibration and deliberately omit it from the exported animation.
SRC_REST_FRAME = 1
SRC_FIRST_MOTION_FRAME = 2
SRC_FPS = 120.0
OUT_FPS = 30.0
STEP = int(round(SRC_FPS / OUT_FPS))
if STEP != 4:
    raise RuntimeError(f"Expected exact 120->30 ratio of 4, got {STEP}")

# SOMA77 target joint -> CMU source joint.  The zero-offset CMU LHipJoint /
# RHipJoint helpers are intentionally skipped. Finger detail is kept static for
# this first body-retarget QA; CMU 137 has only a very coarse hand hierarchy.
MAP = {
    "Hips": "Hips",
    "Spine1": "LowerBack",
    "Spine2": "Spine",
    "Chest": "Spine1",
    "Neck1": "Neck",
    "Neck2": "Neck1",
    "Head": "Head",
    "LeftShoulder": "LeftShoulder",
    "LeftArm": "LeftArm",
    "LeftForeArm": "LeftForeArm",
    "LeftHand": "LeftHand",
    "RightShoulder": "RightShoulder",
    "RightArm": "RightArm",
    "RightForeArm": "RightForeArm",
    "RightHand": "RightHand",
    "LeftLeg": "LeftUpLeg",
    "LeftShin": "LeftLeg",
    "LeftFoot": "LeftFoot",
    "LeftToeBase": "LeftToeBase",
    "RightLeg": "RightUpLeg",
    "RightShin": "RightLeg",
    "RightFoot": "RightFoot",
    "RightToeBase": "RightToeBase",
}


def import_bvh(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(path)
    before = set(bpy.data.objects)
    # Keep import arguments deliberately minimal for compatibility with the
    # Ubuntu Blender package. Both files go through the same importer, so axis
    # conversion is identical.
    result = bpy.ops.import_anim.bvh(filepath=str(path.resolve()), frame_start=1)
    if "FINISHED" not in result:
        raise RuntimeError(f"BVH import failed for {path}: {result}")
    created = [o for o in bpy.data.objects if o not in before and o.type == "ARMATURE"]
    if len(created) != 1:
        raise RuntimeError(f"Expected one armature from {path}, got {len(created)}")
    arm = created[0]
    arm.name = label
    return arm


def pose_snapshot(arm, frame: int):
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    out = {}
    for pb in arm.pose.bones:
        loc, rot, scale = pb.matrix_basis.decompose()
        out[pb.name] = {
            "loc": loc.copy(),
            "rot": rot.copy(),
            "scale": scale.copy(),
        }
    return out


def bone_chain_length(arm, names):
    total = 0.0
    for n in names:
        b = arm.data.bones.get(n)
        if b is None:
            raise RuntimeError(f"Missing bone for scale estimate: {n}")
        total += b.length
    return total


# Clean default scene.
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

src = import_bvh(SRC, "CMU_137_28_Source")
# Preserve the source action's authored range before importing the 1-frame target.
src_action = src.animation_data.action if src.animation_data else None
if src_action is None:
    raise RuntimeError("CMU BVH imported without an action")
src_start = int(math.floor(src_action.frame_range[0]))
src_end = int(math.ceil(src_action.frame_range[1]))
if src_start > SRC_REST_FRAME or src_end < 3000:
    raise RuntimeError(f"Unexpected CMU action range: {src_start}..{src_end}")

tgt = import_bvh(TPOSE, "SOMA77_Target")

missing_tgt = [n for n in MAP if tgt.pose.bones.get(n) is None]
missing_src = [n for n in MAP.values() if src.pose.bones.get(n) is None]
if missing_tgt or missing_src:
    raise RuntimeError(f"Joint mapping incomplete. target_missing={missing_tgt}, source_missing={missing_src}")

src_rest = pose_snapshot(src, SRC_REST_FRAME)
tgt_rest = pose_snapshot(tgt, 1)

# Estimate translation scale from the major leg chains rather than assuming a
# CMU unit. This transfers root drift to the fixed Kimodo training proportion.
src_leg = 0.5 * (
    bone_chain_length(src, ["LeftUpLeg", "LeftLeg"]) +
    bone_chain_length(src, ["RightUpLeg", "RightLeg"])
)
tgt_leg = 0.5 * (
    bone_chain_length(tgt, ["LeftLeg", "LeftShin"]) +
    bone_chain_length(tgt, ["RightLeg", "RightShin"])
)
translation_scale = tgt_leg / src_leg
print("SOURCE_LEG", src_leg, "TARGET_LEG", tgt_leg, "TRANSLATION_SCALE", translation_scale)

# Clear the target file's one-frame action, then build a new 30 Hz action.
tgt.animation_data_clear()
action = bpy.data.actions.new("CMU_137_28_to_SOMA77_30fps")
tgt.animation_data_create()
tgt.animation_data.action = action

# Set all SOMA bones to their imported T-pose basis. Unmapped facial/finger
# bones remain static in that pose for this first retarget validation.
for pb in tgt.pose.bones:
    base = tgt_rest[pb.name]
    pb.rotation_mode = "QUATERNION"
    pb.location = base["loc"]
    pb.rotation_quaternion = base["rot"]
    pb.scale = base["scale"]

source_frames = list(range(SRC_FIRST_MOTION_FRAME, src_end + 1, STEP))
print("RETARGET_FRAME_COUNT", len(source_frames))

# Transfer animation as per-joint *local rest-pose deltas*. Because both source
# and target are T-posed and imported through the same Blender BVH importer,
# this preserves target bone lengths while transferring the actor's motion.
# The first CMU T-pose is calibration-only and never appears in output.
for out_frame, src_frame in enumerate(source_frames, start=1):
    bpy.context.scene.frame_set(src_frame)
    bpy.context.view_layer.update()

    for tgt_name, src_name in MAP.items():
        spb = src.pose.bones[src_name]
        tpb = tgt.pose.bones[tgt_name]
        _, srot, _ = spb.matrix_basis.decompose()
        s0 = src_rest[src_name]
        t0 = tgt_rest[tgt_name]

        # Local animation delta relative to the inserted CMU T-pose.
        delta = s0["rot"].inverted() @ srot
        tpb.rotation_mode = "QUATERNION"
        tpb.rotation_quaternion = t0["rot"] @ delta
        tpb.scale = t0["scale"]
        tpb.location = t0["loc"]

        if tgt_name == "Hips":
            # Preserve target rest height while transferring actor translation.
            sloc = spb.matrix_basis.to_translation()
            dloc = (sloc - s0["loc"]) * translation_scale
            tpb.location = t0["loc"] + dloc

        tpb.keyframe_insert("rotation_quaternion", frame=out_frame, group=tgt_name)
        if tgt_name == "Hips":
            tpb.keyframe_insert("location", frame=out_frame, group=tgt_name)

# Keep Root wrapper static, as in the SOMA standard target hierarchy.
root = tgt.pose.bones.get("Root")
if root:
    root.rotation_mode = "QUATERNION"
    root.location = tgt_rest["Root"]["loc"]
    root.rotation_quaternion = tgt_rest["Root"]["rot"]
    root.keyframe_insert("location", frame=1, group="Root")
    root.keyframe_insert("rotation_quaternion", frame=1, group="Root")

scene = bpy.context.scene
scene.render.fps = int(OUT_FPS)
scene.frame_start = 1
scene.frame_end = len(source_frames)

# Export SOMA77 BVH. Blender's BVH exporter writes the target hierarchy and the
# baked 30 Hz animation; this file is the input for Kimodo's official converter
# in the next stage.
bpy.ops.object.select_all(action="DESELECT")
tgt.select_set(True)
bpy.context.view_layer.objects.active = tgt
export_path = OUT / "137_28_soma77_30fps.bvh"
res = bpy.ops.export_anim.bvh(
    filepath=str(export_path.resolve()),
    frame_start=scene.frame_start,
    frame_end=scene.frame_end,
)
if "FINISHED" not in res or not export_path.exists():
    raise RuntimeError(f"SOMA77 BVH export failed: {res}")

# -------------------------------------------------------------------------
# Technical stick-figure QA renders (full 31 s clip at coarse intervals).
# -------------------------------------------------------------------------
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = 640
scene.render.resolution_y = 640
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.world.color = (0.025, 0.025, 0.035)

# Major visible bones only; fingers/face intentionally excluded from first pass.
DRAW_BONES = [
    "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase",
]

mat = bpy.data.materials.new("QA_Material")
mat.diffuse_color = (0.72, 0.75, 0.82, 1.0)
sticks = {}
for name in DRAW_BONES:
    bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=1.0, depth=2.0)
    obj = bpy.context.object
    obj.name = f"QA_{name}"
    obj.data.materials.append(mat)
    sticks[name] = obj

# Head marker makes head direction/height easier to read.
bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=5.0)
head_marker = bpy.context.object
head_marker.name = "QA_HeadMarker"
head_marker.data.materials.append(mat)


def update_sticks(frame):
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    for name, obj in sticks.items():
        pb = tgt.pose.bones[name]
        a = tgt.matrix_world @ pb.head
        b = tgt.matrix_world @ pb.tail
        vec = b - a
        length = max(vec.length, 1e-4)
        obj.location = (a + b) * 0.5
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = vec.to_track_quat("Z", "Y")
        obj.scale = (1.8, 1.8, length * 0.5)
    hp = tgt.pose.bones["Head"]
    head_marker.location = tgt.matrix_world @ hp.tail

# Sample roughly every 5 seconds plus first/last motion frames.
sample_frames = sorted(set([
    1,
    min(scene.frame_end, 30),
    min(scene.frame_end, 150),
    min(scene.frame_end, 300),
    min(scene.frame_end, 450),
    min(scene.frame_end, 600),
    min(scene.frame_end, 750),
    scene.frame_end,
]))

# Estimate bounds from target joints at sampled frames.
mn = Vector((1e9, 1e9, 1e9))
mx = Vector((-1e9, -1e9, -1e9))
for fr in sample_frames:
    scene.frame_set(fr)
    bpy.context.view_layer.update()
    for n in DRAW_BONES:
        pb = tgt.pose.bones[n]
        for p in (tgt.matrix_world @ pb.head, tgt.matrix_world @ pb.tail):
            mn.x, mn.y, mn.z = min(mn.x, p.x), min(mn.y, p.y), min(mn.z, p.z)
            mx.x, mx.y, mx.z = max(mx.x, p.x), max(mx.y, p.y), max(mx.z, p.z)
center = (mn + mx) * 0.5
span = mx - mn
max_span = max(span.x, span.y, span.z, 1.0)

bpy.ops.object.camera_add()
cam = bpy.context.object
cam.data.type = "ORTHO"
cam.data.ortho_scale = max(span.z * 1.2, span.x * 1.3, span.y * 1.3, 180.0)
scene.camera = cam

def point_cam(pos):
    cam.location = pos
    cam.rotation_euler = (center - pos).to_track_quat("-Z", "Y").to_euler()

# Soft light for the cylinders.
for pos, energy, size in [
    (center + Vector((220, -260, 300)), 900, 180),
    (center + Vector((-220, 100, 180)), 500, 160),
]:
    bpy.ops.object.light_add(type="AREA", location=pos)
    l = bpy.context.object
    l.data.energy = energy
    l.data.size = size
    l.rotation_euler = (center - l.location).to_track_quat("-Z", "Y").to_euler()

rendered = []
views = {
    "front": center + Vector((0, -max_span * 3.0, 0)),
    "side": center + Vector((max_span * 3.0, 0, 0)),
}
for fr in sample_frames:
    update_sticks(fr)
    for vname, pos in views.items():
        point_cam(pos)
        p = PREVIEW / f"f{fr:04d}_{vname}.png"
        scene.render.filepath = str(p.resolve())
        bpy.ops.render.render(write_still=True)
        rendered.append(str(p))

# Motion sanity metrics on the retargeted target.
TRACK = ["Hips", "LeftFoot", "RightFoot", "LeftHand", "RightHand", "Head"]
positions = {n: [] for n in TRACK}
for fr in range(scene.frame_start, scene.frame_end + 1):
    scene.frame_set(fr)
    bpy.context.view_layer.update()
    for n in TRACK:
        positions[n].append(tgt.matrix_world @ tgt.pose.bones[n].head)

def max_disp(name):
    pts = positions[name]
    base = pts[0]
    return max((p - base).length for p in pts)

def max_step(name):
    pts = positions[name]
    if len(pts) < 2:
        return 0.0
    return max((b-a).length for a,b in zip(pts, pts[1:]))

report = {
    "source": str(SRC),
    "source_fps": SRC_FPS,
    "source_frames": [src_start, src_end],
    "source_rest_frame_calibration_only": SRC_REST_FRAME,
    "output_fps": OUT_FPS,
    "output_frames": len(source_frames),
    "output_duration_seconds": (len(source_frames)-1)/OUT_FPS,
    "translation_scale": translation_scale,
    "mapping": MAP,
    "max_displacement_from_output_start_target_units": {n: max_disp(n) for n in TRACK},
    "max_single_frame_step_target_units": {n: max_step(n) for n in TRACK},
    "sample_frames": sample_frames,
    "rendered": rendered,
    "notes": [
        "First CGSpeed T-pose frame used only to compute local rotation deltas; not exported.",
        "120 Hz source sampled every fourth motion frame to exact 30 Hz without changing duration.",
        "Finger/face joints remain static during this first body-retarget QA pass.",
    ],
}
(OUT / "retarget_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

# Save a Blender debug scene so any failed pose can be inspected precisely.
bpy.ops.wm.save_as_mainfile(filepath=str((OUT / "137_28_soma77_retarget_debug.blend").resolve()))
print(json.dumps(report, indent=2))
