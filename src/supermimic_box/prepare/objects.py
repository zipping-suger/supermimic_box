"""Box objects and the two support platforms. A platform carries the box and touches nothing else.

An object is a single box collider with a uniform-density inertial. Its body frame is the frame the
motions were recorded in, so the half extents are given along that frame's axes. The registry also
holds what a simulated world draws for it: the nominal mass and the range of its sliding friction,
which follow the task the box is used for (a pushed box slides easily; a carried one must not).
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .robot import OBJECT_BIT, TABLE_BIT, WORLD_BIT

# Half extents [m] of a support platform: the footprint of OmniContact's training scene and the
# thickness of its MuJoCo scenes, on which the recorded boxes rest exactly.
TABLE_HALF = (0.15, 0.125, 0.01)
TORSIONAL_FRICTION, ROLLING_FRICTION = 0.005, 0.0001
# Per kind of use: nominal mass [kg] and range of the sliding friction against ground and platforms.
# Every box weighs 2 kg, as in OmniContact's training: the pushed boxes are large but are also
# lifted by hand in the carry-then-push clips.
KINDS = {"box": dict(mass=2.0, friction=(0.4, 0.7)), "push": dict(mass=2.0, friction=(0.15, 0.35))}


def box_name(half, kind: str) -> str:
    """``box_30-30-30`` or ``push_46-50-52``: the kind and the full extents in centimetres."""
    return kind + "_" + "-".join(f"{round(200 * h):d}" for h in half)


def box_inertia(mass: float, half) -> np.ndarray:
    x, y, z = (2 * np.asarray(half, dtype=np.float64)) ** 2
    return mass / 12 * np.array([y + z, x + z, x + y])


def write_objects(output: Path, boxes: list[tuple[tuple[float, float, float], str]]) -> dict:
    """Write one MJCF per box, the platform MJCF and ``registry.json``. ``boxes``: (half extents, kind);
    the sorted names define the object keys."""
    import mujoco

    output.mkdir(parents=True, exist_ok=True)
    values = lambda x: " ".join(f"{v:.9g}" for v in np.asarray(x).flatten())  # noqa: E731
    entries, files = [], {}
    for half, kind in sorted(set(boxes), key=lambda b: box_name(*b)):
        name, mass = box_name(half, kind), KINDS[kind]["mass"]
        lo, hi = KINDS[kind]["friction"]
        xml = ET.Element("mujoco", model=name)
        ET.SubElement(xml, "compiler", angle="radian", inertiafromgeom="false")
        body = ET.SubElement(ET.SubElement(xml, "worldbody"), "body", name="object")
        ET.SubElement(body, "freejoint", name="root")
        ET.SubElement(body, "inertial", pos="0 0 0", mass=f"{mass:.9g}", diaginertia=values(box_inertia(mass, half)))
        ET.SubElement(body, "geom", name="collision", type="box", size=values(half), contype=str(WORLD_BIT | OBJECT_BIT),
                      conaffinity=str(WORLD_BIT | TABLE_BIT), group="3",
                      friction=values(((lo + hi) / 2, TORSIONAL_FRICTION, ROLLING_FRICTION)),
                      rgba="0.8 0.6 0.4 1")  # fmt: skip
        path = output / f"{name}.xml"
        ET.indent(xml)
        ET.ElementTree(xml).write(path, encoding="unicode")
        if abs(float(mujoco.MjModel.from_xml_path(str(path)).body_mass.sum()) - mass) > 1e-6 * mass:
            raise ValueError(f"Compiled mass differs for {name}")
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(dict(name=name, xml=path.name, kind=kind, half=[float(h) for h in half], mass=mass, friction=[lo, hi]))

    table = ET.Element("mujoco", model="table")
    top = ET.SubElement(ET.SubElement(table, "worldbody"), "body", name="table")
    ET.SubElement(top, "geom", name="top", type="box", size=values(TABLE_HALF),
                  contype=str(TABLE_BIT), conaffinity="0", group="3", rgba="0.6 0.1 0.1 1")  # fmt: skip
    ET.indent(table)
    ET.ElementTree(table).write(output / "table.xml", encoding="unicode")
    files["table.xml"] = hashlib.sha256((output / "table.xml").read_bytes()).hexdigest()

    registry = dict(format_version=1, objects=entries, table=dict(xml="table.xml", half=list(TABLE_HALF)), files=files)
    (output / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    return registry
