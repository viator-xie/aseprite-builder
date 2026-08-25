from __future__ import annotations

import json
from pathlib import Path

import bpy
from mathutils import Vector

BVH = Path("native_out/137_28_soma77_native_30fps.bvh")
REPORT = Path("native_out/native_retarget_report.json")
OUT = Path("native_out/qa_candidates")
OUT.mkdir(parents=True, exist_ok=True)

if not BVH.exists() or not REPORT.exists():
    raise FileNotFoundError("Native retarget output/report missing")
meta = json.loads(REPORT.read_text(encoding="utf-8"))
selected = meta["top_idle_windows"]
if len(selected) < 3:
    raise RuntimeError("Expected three candidate windows")

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
res = bpy.ops.import_anim.bvh(filepath=str(BVH.resolve()), frame_start=1)
if "FINISHED" not in res:
    raise RuntimeError(f"BVH import failed: {res}")
armatures = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
if len(armatures) != 1:
    raise RuntimeError(f"Expected one SOMA armature, got {len(armatures)}")
arm = armatures[0]
arm.name = "SOMA77_Native_QA"

scene = bpy.context.scene
scene.render.fps = 30
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = 640
scene.render.resolution_y = 640
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.world.color = (0.035, 0.035, 0.045)

DRAW = [
    "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase",
]
missing = [n for n in DRAW if arm.pose.bones.get(n) is None]
if missing:
    raise RuntimeError(f"SOMA QA missing bones: {missing}")

mat = bpy.data.materials.new("QA_Bright")
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
if bsdf:
    bsdf.inputs["Base Color"].default_value = (0.72, 0.78, 0.9, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.5

sticks = {}
for name in DRAW:
    bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=1.0, depth=2.0)
    o = bpy.context.object
    o.name = f"Stick_{name}"
    o.data.materials.append(mat)
    sticks[name] = o

bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12, radius=5.0)
head = bpy.context.object
head.name = "HeadMarker"
head.data.materials.append(mat)


def update_stick_pose(frame: int):
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    for name, o in sticks.items():
        pb = arm.pose.bones[name]
        a = arm.matrix_world @ pb.head
        b = arm.matrix_world @ pb.tail
        v = b - a
        L = max(v.length, 1e-4)
        o.location = (a + b) * 0.5
        o.rotation_mode = "QUATERNION"
        o.rotation_quaternion = v.to_track_quat("Z", "Y")
        o.scale = (1.9, 1.9, L * 0.5)
    hp = arm.pose.bones["Head"]
    head.location = arm.matrix_world @ hp.tail

# Candidate snapshots: start, middle, end-1 for each top-ranked 5 s window.
snapshots = []
for rank, w in enumerate(selected, start=1):
    a = int(w["start_frame_0based"]) + 1
    b = int(w["end_frame_exclusive_0based"])
    m = (a + b) // 2
    snapshots.extend([
        (rank, "start", a),
        (rank, "mid", m),
        (rank, "end", b),
    ])

# Bounds across every candidate snapshot.
mn = Vector((1e9, 1e9, 1e9))
mx = Vector((-1e9, -1e9, -1e9))
for _, _, fr in snapshots:
    scene.frame_set(fr)
    bpy.context.view_layer.update()
    for n in DRAW:
        pb = arm.pose.bones[n]
        for p in (arm.matrix_world @ pb.head, arm.matrix_world @ pb.tail):
            mn.x, mn.y, mn.z = min(mn.x,p.x), min(mn.y,p.y), min(mn.z,p.z)
            mx.x, mx.y, mx.z = max(mx.x,p.x), max(mx.y,p.y), max(mx.z,p.z)
center = (mn + mx) * 0.5
span = mx - mn
max_span = max(span.x, span.y, span.z, 1.0)

bpy.ops.object.camera_add()
cam = bpy.context.object
cam.data.type = "ORTHO"
cam.data.ortho_scale = max(span.z * 1.25, span.x * 1.35, span.y * 1.35, 180.0)
scene.camera = cam

def point_camera(pos: Vector):
    cam.location = pos
    cam.rotation_euler = (center - pos).to_track_quat("-Z", "Y").to_euler()

for pos, energy, size in [
    (center + Vector((250,-300,320)), 1900, 220),
    (center + Vector((-250,-80,180)), 1100, 180),
    (center + Vector((50,300,250)), 1300, 200),
]:
    bpy.ops.object.light_add(type="AREA", location=pos)
    l=bpy.context.object
    l.data.energy=energy
    l.data.size=size
    l.rotation_euler=(center-l.location).to_track_quat("-Z","Y").to_euler()

views = {
    "front": center + Vector((0, -max_span*3.0, 0)),
    "side": center + Vector((max_span*3.0, 0, 0)),
}
rendered=[]
for rank, phase, fr in snapshots:
    update_stick_pose(fr)
    for view, pos in views.items():
        point_camera(pos)
        p=OUT/f"candidate{rank}_{phase}_f{fr:04d}_{view}.png"
        scene.render.filepath=str(p.resolve())
        bpy.ops.render.render(write_still=True)
        rendered.append(str(p))

qa = {
    "selected_windows": selected,
    "snapshots": [{"rank":r,"phase":p,"frame":f} for r,p,f in snapshots],
    "rendered": rendered,
}
(OUT/"qa_candidates.json").write_text(json.dumps(qa,indent=2),encoding="utf-8")
print(json.dumps(qa,indent=2))
