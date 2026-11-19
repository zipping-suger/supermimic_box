"""Turn OmniContact's MuJoCo G1 (29 DoF, rubber hands) into the MJCF used for simulation and torch FK."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import mujoco

MJCF = "g1_description/g1_29dof.xml"  # inside an OmniContact_sim2sim checkout
MAX_HULL_VERTICES = 32  # collision uses each link mesh's convex hull, capped at this many vertices
VISUAL_GROUP, COLLISION_GROUP = 1, 3
# Collision bits. Robot geoms never touch each other (contype 0) and accept bit 1, which the ground
# and the object have. The object also has bit 2, which only the foot boxes accept. A platform has
# bit 4, which only the object accepts: as in OmniContact's MuJoCo scenes, a platform carries the
# box but is no obstacle for the robot (unused platforms lie at ground level under its feet).
WORLD_BIT, OBJECT_BIT, TABLE_BIT = 1, 2, 4
# A box around each foot that touches the object only: the four small sole spheres of the source
# model stand on the ground but cannot push a box. Centre and half extents in the ankle-roll frame.
FOOT_BOX = ((0.038, 0.0, -0.005), (0.104, 0.038, 0.029))


def prepare_robot(sim2sim: Path, output: Path) -> Path:
    """Write ``output/g1_rubber.xml`` and its meshes. Kinematics and inertias are the source's.

    The source's motors, sensors and passive joint terms are dropped: mjlab adds one PD position
    actuator per joint with its own armature. Collision geoms touch the ground and the object, but
    neither each other nor the platforms.
    """
    source = Path(sim2sim) / MJCF
    spec = mujoco.MjSpec.from_file(str(source))
    spec.modelname = "g1_rubber"
    spec.option.integrator, spec.option.timestep = mujoco.mjtIntegrator.mjINT_EULER, 0.002  # MuJoCo's defaults; the simulation sets its own
    for element in (*spec.actuators, *spec.sensors):
        spec.delete(element)
    for joint in spec.joints:
        if joint.type == mujoco.mjtJoint.mjJNT_HINGE:
            joint.damping, joint.armature, joint.frictionloss = [0.0, 0.0, 0.0], 0.0, 0.0

    collision_meshes = set()
    for body in spec.bodies:
        counts = {"visual": 0, "collision": 0}
        for geom in body.geoms:
            role = "visual" if geom.contype == 0 and geom.conaffinity == 0 else "collision"
            geom.name = f"{body.name}_{role}_{counts[role]}"
            counts[role] += 1
            if role == "visual":
                geom.group = VISUAL_GROUP
                continue
            geom.group, geom.contype, geom.conaffinity = COLLISION_GROUP, 0, WORLD_BIT
            if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                collision_meshes.add(geom.meshname)
        if body.name.endswith("_ankle_roll_link"):
            center, half = FOOT_BOX
            body.add_geom(name=f"{body.name}_object_box", type=mujoco.mjtGeom.mjGEOM_BOX, pos=center, size=half,
                          group=COLLISION_GROUP, contype=0, conaffinity=OBJECT_BIT, density=0, rgba=[0.2, 0.2, 0.2, 0.3])  # fmt: skip

    output.mkdir(parents=True, exist_ok=True)
    mesh_dir = (output / "meshes").resolve()
    mesh_dir.mkdir(exist_ok=True)
    for mesh in spec.meshes:
        file = source.parent / "meshes" / Path(mesh.file).name
        shutil.copyfile(file, mesh_dir / file.name)
        mesh.file = file.name
    spec.meshdir = str(mesh_dir)
    spec.compile()
    xml, count = re.subn(r'meshdir="[^"]*"', 'meshdir="meshes"', spec.to_xml())
    if count != 1:
        raise ValueError("Could not make the mesh directory relative")
    # MuJoCo 3.11 does not serialize maxhullvert, so set it in the text.
    for name in collision_meshes:
        xml, count = re.subn(f'<mesh name="{name}" ', f'<mesh name="{name}" maxhullvert="{MAX_HULL_VERTICES}" ', xml)
        if count != 1:
            raise ValueError(f"Collision mesh not found in MJCF: {name}")
    path = output / "g1_rubber.xml"
    path.write_text(xml)
    mujoco.MjModel.from_xml_path(str(path))  # must load on its own
    return path
