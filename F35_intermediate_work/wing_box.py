import gmsh
import math

gmsh.initialize()
gmsh.model.add("wing_fluid")

# --------------------------------------------------
# 1. Import wing STEP
# --------------------------------------------------
wing = gmsh.model.occ.importShapes("swept_wing_washout.step")
gmsh.model.occ.synchronize()

wing = gmsh.model.occ.importShapes("swept_wing_washout.step")
gmsh.model.occ.synchronize()

print("Imported entities:", wing)
print("Volumes:", gmsh.model.getEntities(3))
print("Surfaces:", gmsh.model.getEntities(2))

# --------------------------------------------------
# 2. Bounding box of imported wing
# --------------------------------------------------
xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(
    wing[0][0], wing[0][1]
)

Lx = xmax - xmin
Ly = ymax - ymin
Lz = zmax - zmin

# Use chord-like size as reference
ref = max(Lx, Ly, Lz)

# --------------------------------------------------
# 3. Create outer box around wing
# --------------------------------------------------
pad_inlet  = 5.0 * ref
pad_outlet = 10.0 * ref
pad_side   = 5.0 * ref
pad_top    = 5.0 * ref
pad_bottom = 5.0 * ref

box_x0 = xmin - pad_inlet
box_y0 = ymin - pad_side
box_z0 = zmin - pad_bottom

box_dx = (xmax - xmin) + pad_inlet + pad_outlet
box_dy = (ymax - ymin) + 2.0 * pad_side
box_dz = (zmax - zmin) + pad_bottom + pad_top

box = gmsh.model.occ.addBox(box_x0, box_y0, box_z0, box_dx, box_dy, box_dz)

# --------------------------------------------------
# 4. Subtract wing from box => fluid domain
# --------------------------------------------------
cut = gmsh.model.occ.cut(
    [(3, box)],
    wing,
    removeObject=True,
    removeTool=False
)

gmsh.model.occ.synchronize()

fluid_volumes = cut[0]
assert len(fluid_volumes) == 1, f"Expected one fluid volume, got {len(fluid_volumes)}"
fluid = fluid_volumes[0]

# --------------------------------------------------
# 5. Identify boundary surfaces by bounding box
# --------------------------------------------------
surfaces = gmsh.model.getBoundary([fluid], oriented=False, recursive=False)

tol = 1e-6
inlet_surfs = []
outlet_surfs = []
farfield_surfs = []
body_surfs = []

for dim, tag in surfaces:
    bxmin, bymin, bzmin, bxmax, bymax, bzmax = gmsh.model.occ.getBoundingBox(dim, tag)

    # Outer box faces
    if abs(bxmin - box_x0) < tol and abs(bxmax - box_x0) < tol:
        inlet_surfs.append(tag)
    elif abs(bxmin - (box_x0 + box_dx)) < tol and abs(bxmax - (box_x0 + box_dx)) < tol:
        outlet_surfs.append(tag)
    elif (
        abs(bymin - box_y0) < tol and abs(bymax - box_y0) < tol
    ) or (
        abs(bymin - (box_y0 + box_dy)) < tol and abs(bymax - (box_y0 + box_dy)) < tol
    ) or (
        abs(bzmin - box_z0) < tol and abs(bzmax - box_z0) < tol
    ) or (
        abs(bzmin - (box_z0 + box_dz)) < tol and abs(bzmax - (box_z0 + box_dz)) < tol
    ):
        farfield_surfs.append(tag)
    else:
        body_surfs.append(tag)

# --------------------------------------------------
# 6. Physical groups
# --------------------------------------------------
FLUID = 10
INLET = 1
OUTLET = 2
FARFIELD = 3
BODY = 4

gmsh.model.addPhysicalGroup(3, [fluid[1]], FLUID)
gmsh.model.setPhysicalName(3, FLUID, "Fluid")

if inlet_surfs:
    gmsh.model.addPhysicalGroup(2, inlet_surfs, INLET)
    gmsh.model.setPhysicalName(2, INLET, "Inlet")

if outlet_surfs:
    gmsh.model.addPhysicalGroup(2, outlet_surfs, OUTLET)
    gmsh.model.setPhysicalName(2, OUTLET, "Outlet")

if farfield_surfs:
    gmsh.model.addPhysicalGroup(2, farfield_surfs, FARFIELD)
    gmsh.model.setPhysicalName(2, FARFIELD, "Farfield")

if body_surfs:
    gmsh.model.addPhysicalGroup(2, body_surfs, BODY)
    gmsh.model.setPhysicalName(2, BODY, "Body")

# --------------------------------------------------
# 7. Mesh sizes
# --------------------------------------------------
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.08 * ref)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.25 * ref)

gmsh.model.mesh.generate(3)
gmsh.write("wing_fluid.msh")

# Optional GUI
# gmsh.fltk.run()

gmsh.finalize()
