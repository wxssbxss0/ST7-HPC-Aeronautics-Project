"""
Mesh conversion script for the Corsair 3D simulation (Euler & Stokes).

Input:  corsair_domain.msh  — MSH 2.2 mesh containing the full boxed domain
                               (plane body + surrounding fluid box).
Output: corsair_tetra.xdmf  — Volumetric mesh (tetrahedral cells, "Fluid" region)
        corsair_tri.xdmf  — Surface mesh (triangular cells, boundary tags)

Physical tags expected in the .msh file (define these in Gmsh):
    Volume:
        1  →  Fluid  (the entire box minus the plane solid)
    Surfaces:
        1  →  Corsair   (plane skin — no-slip wall for Stokes, slip wall for Euler)
        2  →  Inlet     (inflow face of the bounding box)
        3  →  Outlet    (outflow face of the bounding box)
        4  →  Walls     (remaining box faces — symmetry or slip wall)

If your Gmsh script uses different tag numbers, update the TAG_* constants below.
"""

import meshio
import numpy as np

# ---------------------------------------------------------------------------
# Tag constants — edit here if your Gmsh physical groups use different IDs
# ---------------------------------------------------------------------------
TAG_FLUID   = 1   # Volume physical group

TAG_CORSAIR = 1   # Surface: plane skin
TAG_INLET   = 2   # Surface: inflow
TAG_OUTLET  = 3   # Surface: outflow
TAG_WALLS   = 4   # Surface: side/top/bottom box faces


def convert(msh_file: str = "corsair_domain.msh") -> None:
    """
    Reads a Gmsh MSH 2.2 mesh and writes two XDMF files ready for
    FEniCSx (dolfinx):
        - corsair_domain.xdmf  : tetrahedral volume mesh
        - corsair_facets.xdmf  : triangular boundary/facet mesh
    Both files carry the physical-group integer tags in the 'Grid' cell-data
    array, which dolfinx reads via meshtags.
    """

    print(f"Reading mesh: {msh_file}")
    msh = meshio.read(msh_file)

    # ------------------------------------------------------------------
    # 1. VOLUME MESH — tetrahedral cells (3-D fluid domain)
    # ------------------------------------------------------------------
    # Extract all tetrahedral cells and their physical-group tags.
    # For the Corsair we have a single fluid volume (tag = TAG_FLUID),
    # but we keep the full tag array so dolfinx can mark subdomains later
    # (useful if you add, e.g., a refined wake region as a separate group).
    tetra_cells = msh.get_cells_type("tetra")
    tetra_data  = msh.get_cell_data("gmsh:physical", "tetra")

    volume_mesh = meshio.Mesh(
        points=msh.points,
        cells={"tetra": tetra_cells},
        cell_data={"Grid": [tetra_data]},
    )
    meshio.write("corsair_tetra.xdmf", volume_mesh)
    print("Written: corsair_tetra.xdmf")

    # ------------------------------------------------------------------
    # 2. FACET/SURFACE MESH — triangular cells (boundary conditions)
    # ------------------------------------------------------------------
    # The surface mesh carries ALL boundary patches in one file.
    # dolfinx distinguishes them via the integer tags:
    #   TAG_CORSAIR → no-slip (Stokes) or slip wall (Euler)
    #   TAG_INLET   → Dirichlet inflow velocity / total pressure
    #   TAG_OUTLET  → Neumann outflow / pressure outlet
    #   TAG_WALLS   → symmetry plane or free-slip wall
    tri_cells = msh.get_cells_type("triangle")
    tri_data  = msh.get_cell_data("gmsh:physical", "triangle")

    facet_mesh = meshio.Mesh(
        points=msh.points,
        cells={"triangle": tri_cells},
        cell_data={"Grid": [tri_data]},
    )
    meshio.write("corsair_tri.xdmf", facet_mesh)
    print("Written: corsair_tri.xdmf")

    # ------------------------------------------------------------------
    # 3. DEBUG / SANITY CHECK
    # ------------------------------------------------------------------
    print("\n--- CONVERSION DEBUG ---")
    print(f"  Total points          : {len(msh.points)}")
    print(f"  Volume cells (tetra)  : {len(tetra_cells)}")
    print(f"  Surface cells (tri)   : {len(tri_cells)}")

    print("\n  Volume tags found     :", np.unique(tetra_data))
    print("  Surface tags found    :", np.unique(tri_data))

    # Map tag IDs back to human-readable names for quick verification
    tag_names = {
        TAG_CORSAIR : "Corsair (plane skin)",
        TAG_INLET   : "Inlet",
        TAG_OUTLET  : "Outlet",
        TAG_WALLS   : "Walls",
    }
    print("\n  Surface tag breakdown:")
    for tag, name in tag_names.items():
        count = np.sum(tri_data == tag)
        print(f"    Tag {tag:>2d}  ({name:<25s}): {count:>8d} triangles")

    # Warn if any unexpected tags appear (typos in Gmsh script, etc.)
    expected_surface_tags = set(tag_names.keys())
    found_surface_tags    = set(np.unique(tri_data).tolist())
    unexpected = found_surface_tags - expected_surface_tags
    if unexpected:
        print(f"\n  WARNING: unexpected surface tags {unexpected} — "
              "check your Gmsh physical groups.")


if __name__ == "__main__":
    convert()
