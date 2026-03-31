from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import cadquery as cq


def naca4_points(code: str, chord: float = 1.0, n: int = 161, te_frac: float = 0.002):
    """Return a closed blunt-TE NACA 4-digit airfoil polyline in the XZ plane.

    Output points are ordered around the perimeter and end at the start point.
    Coordinates are tuples (x, z) for use on a CadQuery workplane.
    """
    if len(code) != 4 or not code.isdigit():
        raise ValueError(f"Invalid NACA code: {code}")

    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0

    beta = [math.pi * i / (n - 1) for i in range(n)]
    x = [0.5 * chord * (1.0 - math.cos(b)) for b in beta]

    # Thickness distribution with finite trailing-edge thickness for robustness
    yt = []
    for xi in x:
        xc = xi / chord if chord else 0.0
        y = 5.0 * t * chord * (
            0.2969 * math.sqrt(max(xc, 0.0))
            - 0.1260 * xc
            - 0.3516 * xc**2
            + 0.2843 * xc**3
            - 0.1015 * xc**4
        )
        yt.append(y)
    yt[-1] = te_frac * chord

    yc, dyc_dx = [], []
    for xi in x:
        xc = xi / chord if chord else 0.0
        if p > 0 and xc < p:
            yc.append((m / p**2) * (2 * p * xc - xc**2) * chord)
            dyc_dx.append((2 * m / p**2) * (p - xc))
        elif p > 0:
            yc.append((m / (1 - p) ** 2) * ((1 - 2 * p) + 2 * p * xc - xc**2) * chord)
            dyc_dx.append((2 * m / (1 - p) ** 2) * (p - xc))
        else:
            yc.append(0.0)
            dyc_dx.append(0.0)

    theta = [math.atan(s) for s in dyc_dx]

    xu, zu, xl, zl = [], [], [], []
    for xi, yci, yti, th in zip(x, yc, yt, theta):
        xu.append(xi - yti * math.sin(th))
        zu.append(yci + yti * math.cos(th))
        xl.append(xi + yti * math.sin(th))
        zl.append(yci - yti * math.cos(th))

    # Hard-clamp endpoints for a finite blunt trailing edge
    xu[0], zu[0] = 0.0, 0.0
    xl[0], zl[0] = 0.0, 0.0
    xu[-1], zu[-1] = chord, te_frac * chord
    xl[-1], zl[-1] = chord, -te_frac * chord

    upper = [(float(xu[i]), float(zu[i])) for i in range(len(xu) - 1, -1, -1)]
    lower = [(float(xl[i]), float(zl[i])) for i in range(1, len(xl))]
    pts = upper + lower
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts


def wire_on_plane(points_xz: Iterable[tuple[float, float]], y_offset: float = 0.0) -> cq.Wire:
    wp = cq.Workplane("XZ").workplane(offset=y_offset)
    wire = wp.polyline(list(points_xz)).close().wire().val()
    if not wire.isValid():
        raise RuntimeError("Generated airfoil wire is invalid")
    return wire


def rotate_wire_about_x(wire: cq.Wire, angle_deg: float, origin=(0.0, 0.0, 0.0)) -> cq.Wire:
    return wire.rotate(origin, (1.0, 0.0, 0.0), angle_deg)


def translate_shape(shape: cq.Shape, dx=0.0, dy=0.0, dz=0.0):
    return shape.translate((dx, dy, dz))


def build_wing(
    root_code: str = "2412",
    tip_code: str = "0012",
    span: float = 5.0,
    root_chord: float = 1.0,
    tip_chord: float = 0.55,
    sweep: float = 1.2,
    dihedral_deg: float = 0.0,
    washout_deg: float = 0.0,
    npts: int = 161,
) -> cq.Solid:
    root_pts = naca4_points(root_code, chord=root_chord, n=npts)
    tip_pts = naca4_points(tip_code, chord=tip_chord, n=npts)

    root_wire = wire_on_plane(root_pts, y_offset=0.0)
    tip_wire = wire_on_plane(tip_pts, y_offset=span)
    tip_wire = translate_shape(tip_wire, dx=sweep, dz=span * math.tan(math.radians(dihedral_deg)))
    if abs(washout_deg) > 1e-12:
        quarter = (0.25 * tip_chord + sweep, span, span * math.tan(math.radians(dihedral_deg)))
        tip_wire = rotate_wire_about_x(tip_wire, washout_deg, origin=quarter)

    solid = cq.Solid.makeLoft([root_wire, tip_wire], ruled=False)
    return solid


def export_and_validate(solid: cq.Solid, out_path: str | Path):
    out_path = Path(out_path)
    wp = cq.Workplane(obj=solid)
    print("Raw shape type:", solid.ShapeType())
    print("Raw is valid:", solid.isValid())
    print("Raw volume:", solid.Volume())

    cleaned = wp.clean().val()
    print("Cleaned shape type:", cleaned.ShapeType())
    print("Cleaned is valid:", cleaned.isValid())
    print("Cleaned volume:", cleaned.Volume())

    cq.exporters.export(cq.Workplane(obj=cleaned), str(out_path))
    print(f"Exported STEP: {out_path.resolve()}")

    if not cleaned.isValid():
        raise RuntimeError(
            "STEP exported, but cleaned solid is still invalid. Reduce washout/sweep or use 0012 at both stations."
        )


def main():
    # Conservative geometry chosen to favor a valid loft for CFD meshing.
    solid = build_wing(
        root_code="0012",
        tip_code="0012",
        span=5.0,
        root_chord=1.0,
        tip_chord=0.60,
        sweep=1.0,
        dihedral_deg=0.0,
        washout_deg=0.0,
        npts=181,
    )
    export_and_validate(solid, "f35_wing.step")


if __name__ == "__main__":
    main()
