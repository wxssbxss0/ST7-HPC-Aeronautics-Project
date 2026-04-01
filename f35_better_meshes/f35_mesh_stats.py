"""
aircraft_mesh.py
================
Reads the Meshmixer STL surface mesh and performs a quality analysis.

For each triangle, computes:
  - Aspect Ratio (AR) : AR=1 for equilateral, AR→∞ for degenerate slivers
  - Petit h           : equivalent local mesh size = sqrt(2 * area)

Exports a .vtu file readable in ParaView with 4 scalar fields:
  - AspectRatio_1to5  : AR clamped to [1,5]  → readable colormap
  - LogAspectRatio    : log10(AR)             → fine gradient among good triangles
  - PoorQuality_AR5   : binary flag, 1 if AR > 5  → problem zones in red
  - PetitH            : local mesh size

Usage:
  python f35_mesh_stats.py

Output:
  meshed_f35_AR_quality.vtu  (same directory as the STL)
  → open in ParaView, color by PoorQuality_AR5 to spot bad triangles
"""

import numpy as np
import struct
import os
import sys

# ---------------------------------------------------------------------------
# PATH
# ---------------------------------------------------------------------------

STL_FILE = "/mnt/c/Users/Owner/OneDrive/Desktop/centrale/ST7_project/f35/f35_better_meshes/f35_better_1.stl"

if not os.path.isfile(STL_FILE):
    print(f"ERROR: file not found: {STL_FILE}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 1. Read STL (binary, as exported by Meshmixer)
# ---------------------------------------------------------------------------

print(f"Reading STL: {STL_FILE}")

points    = []
triangles = []
pt_index  = {}

def get_or_add(px, py, pz):
    key = (round(px, 8), round(py, 8), round(pz, 8))
    if key not in pt_index:
        pt_index[key] = len(points)
        points.append([px, py, pz])
    return pt_index[key]

try:
    with open(STL_FILE, 'rb') as f:
        f.read(80)
        n_tri = int.from_bytes(f.read(4), 'little')
        print(f"  Binary STL: {n_tri:,} triangles")
        for _ in range(n_tri):
            f.read(12)
            verts = []
            for _ in range(3):
                x, y, z = struct.unpack('<fff', f.read(12))
                verts.append(get_or_add(x, y, z))
            f.read(2)
            triangles.append(tuple(verts))
except Exception:
    print("  Trying ASCII STL ...")
    points.clear(); triangles.clear(); pt_index.clear()
    with open(STL_FILE, 'r', errors='replace') as f:
        buf = []
        for line in f:
            line = line.strip()
            if line.startswith('vertex'):
                v = line.split()
                buf.append((float(v[1]), float(v[2]), float(v[3])))
            elif line.startswith('endloop') and len(buf) == 3:
                triangles.append(tuple(get_or_add(*v) for v in buf))
                buf = []
    print(f"  ASCII STL: {len(triangles):,} triangles")

points = np.array(points, dtype=np.float64)
n_tri  = len(triangles)
print(f"  Unique vertices: {len(points):,}")

# ---------------------------------------------------------------------------
# 2. Compute AR and petit h
# ---------------------------------------------------------------------------

print(f"Computing quality metrics for {n_tri:,} triangles ...")

AR     = np.zeros(n_tri, dtype=np.float64)
h_mesh = np.zeros(n_tri, dtype=np.float64)

for k, (i0, i1, i2) in enumerate(triangles):
    p0, p1, p2 = points[i0], points[i1], points[i2]
    e0 = p1 - p0
    e1 = p2 - p1
    e2 = p0 - p2
    a  = np.linalg.norm(e0)
    b  = np.linalg.norm(e1)
    c  = np.linalg.norm(e2)

    cross     = np.cross(e0, -e2)
    area      = 0.5 * np.linalg.norm(cross)
    h_mesh[k] = np.sqrt(2.0 * area) if area > 0 else 0.0

    s     = (a + b + c) / 2.0
    denom = 8.0 * max(s-a, 1e-30) * max(s-b, 1e-30) * max(s-c, 1e-30)
    AR[k] = (a * b * c) / denom if denom > 1e-30 else 1e6

# ---------------------------------------------------------------------------
# 3. Statistics
# ---------------------------------------------------------------------------

print(f"\n{'='*55}")
print(f"ASPECT RATIO — {n_tri:,} triangles:")
print(f"  Min    : {AR.min():.3f}")
print(f"  Max    : {AR.max():.1f}")
print(f"  Mean   : {AR.mean():.3f}")
print(f"  Median : {np.median(AR):.3f}")
print(f"  AR < 2   excellent  : {(AR<2).sum():>8,}  ({100*(AR<2).mean():.1f}%)")
print(f"  AR 2-5   acceptable : {((AR>=2)&(AR<5)).sum():>8,}  ({100*((AR>=2)&(AR<5)).mean():.1f}%)")
print(f"  AR 5-10  poor       : {((AR>=5)&(AR<10)).sum():>8,}  ({100*((AR>=5)&(AR<10)).mean():.1f}%)")
print(f"  AR > 10  degenerate : {(AR>=10).sum():>8,}  ({100*(AR>=10).mean():.1f}%)")
print(f"\nPETIT H:")
print(f"  Min    : {h_mesh.min():.6f}  (same units as STL)")
print(f"  Max    : {h_mesh.max():.6f}")
print(f"  Mean   : {h_mesh.mean():.6f}")
print(f"  Median : {np.median(h_mesh):.6f}")
print(f"{'='*55}")

pct_poor = 100 * (AR >= 5).mean()
if pct_poor < 2:
    print(f"Mesh quality: GOOD ({pct_poor:.1f}% poor triangles)")
elif pct_poor < 10:
    print(f"Mesh quality: ACCEPTABLE ({pct_poor:.1f}% poor)")
else:
    print(f"Mesh quality: POOR ({pct_poor:.1f}% poor — Remesh recommended in Meshmixer)")

# ---------------------------------------------------------------------------
# 4. Write VTU
# ---------------------------------------------------------------------------

AR_clamped = np.clip(AR, 1.0, 5.0).astype(np.float32)
AR_log     = np.log10(np.clip(AR, 1.0, None)).astype(np.float32)
poor_flag  = (AR > 5.0).astype(np.float32)
h_vis      = h_mesh.astype(np.float32)

stl_dir  = os.path.dirname(STL_FILE)
stl_base = os.path.splitext(os.path.basename(STL_FILE))[0]
VTU_FILE = os.path.join(stl_dir, stl_base + "_AR_quality.vtu")

print(f"\nWriting VTU: {VTU_FILE}")

with open(VTU_FILE, 'w') as f:
    f.write('<?xml version="1.0"?>\n')
    f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
    f.write('  <UnstructuredGrid>\n')
    f.write(f'    <Piece NumberOfPoints="{len(points)}" NumberOfCells="{n_tri}">\n')
    f.write('      <Points>\n')
    f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
    for p in points:
        f.write(f'          {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')
    f.write('        </DataArray>\n')
    f.write('      </Points>\n')
    f.write('      <Cells>\n')
    f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
    for i0, i1, i2 in triangles:
        f.write(f'          {i0} {i1} {i2}\n')
    f.write('        </DataArray>\n')
    f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
    for k in range(1, n_tri + 1):
        f.write(f'          {3*k}\n')
    f.write('        </DataArray>\n')
    f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
    for _ in range(n_tri):
        f.write('          5\n')
    f.write('        </DataArray>\n')
    f.write('      </Cells>\n')
    f.write('      <CellData>\n')
    for name, data in [
        ("AspectRatio_1to5", AR_clamped),
        ("LogAspectRatio",   AR_log),
        ("PoorQuality_AR5",  poor_flag),
        ("PetitH",           h_vis),
    ]:
        f.write(f'        <DataArray type="Float32" Name="{name}" format="ascii">\n')
        for v in data:
            f.write(f'          {v:.4f}\n')
        f.write('        </DataArray>\n')
    f.write('      </CellData>\n')
    f.write('    </Piece>\n')
    f.write('  </UnstructuredGrid>\n')
    f.write('</VTKFile>\n')

print(f"""
Done. Open in ParaView:
  File -> Open -> {os.path.basename(VTU_FILE)}

  Color by PoorQuality_AR5  -> blue=good, red=AR>5
  Color by AspectRatio_1to5 -> gradient [1,5]
  Color by PetitH           -> mesh density map
""")
