import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, io, mesh as dmesh
from dolfinx.io import XDMFFile
import ufl
from basix.ufl import element, mixed_element

# --------------------------------------------------
# 1. Read 3D mesh
# --------------------------------------------------
with XDMFFile(MPI.COMM_WORLD, "f35_mesh.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")

tdim = mesh.topology.dim
fdim = tdim - 1
gdim = mesh.geometry.dim
cell = mesh.basix_cell()

assert gdim == 3, f"Expected 3D mesh, got gdim={gdim}"

# --------------------------------------------------
# 2. Compute box bounds from mesh coordinates
# --------------------------------------------------
x = mesh.geometry.x

xmin = np.min(x[:, 0])
xmax = np.max(x[:, 0])
ymin = np.min(x[:, 1])
ymax = np.max(x[:, 1])
zmin = np.min(x[:, 2])
zmax = np.max(x[:, 2])

Lref = max(xmax - xmin, ymax - ymin, zmax - zmin)
tol = 1e-6 * Lref

if mesh.comm.rank == 0:
    print("Mesh bounds:")
    print(f"  x in [{xmin:.6g}, {xmax:.6g}]")
    print(f"  y in [{ymin:.6g}, {ymax:.6g}]")
    print(f"  z in [{zmin:.6g}, {zmax:.6g}]")
    print(f"  tolerance = {tol:.6g}")

# Build boundary connectivity
mesh.topology.create_connectivity(fdim, tdim)

# --------------------------------------------------
# 3. Locate boundary facets geometrically
# --------------------------------------------------
def inlet(x):
    return np.isclose(x[0], xmin, atol=tol)

def outlet(x):
    return np.isclose(x[0], xmax, atol=tol)

def outer_walls(x):
    return (
        np.isclose(x[1], ymin, atol=tol)
        | np.isclose(x[1], ymax, atol=tol)
        | np.isclose(x[2], zmin, atol=tol)
        | np.isclose(x[2], zmax, atol=tol)
    )

def any_boundary(x):
    return np.full(x.shape[1], True, dtype=bool)

inlet_facets = dmesh.locate_entities_boundary(mesh, fdim, inlet)
outlet_facets = dmesh.locate_entities_boundary(mesh, fdim, outlet)
wall_facets = dmesh.locate_entities_boundary(mesh, fdim, outer_walls)
all_boundary_facets = dmesh.locate_entities_boundary(mesh, fdim, any_boundary)

outer_facets = np.union1d(np.union1d(inlet_facets, outlet_facets), wall_facets)
body_facets = np.setdiff1d(all_boundary_facets, outer_facets)

if mesh.comm.rank == 0:
    print("Boundary facet counts:")
    print(f"  inlet   = {len(inlet_facets)}")
    print(f"  outlet  = {len(outlet_facets)}")
    print(f"  walls   = {len(wall_facets)}")
    print(f"  body    = {len(body_facets)}")
    print(f"  all bnd = {len(all_boundary_facets)}")

# --------------------------------------------------
# 4. Taylor-Hood space
# --------------------------------------------------
P2 = element("Lagrange", cell, 2, shape=(gdim,))
P1 = element("Lagrange", cell, 1)
TH = mixed_element([P2, P1])
W = fem.functionspace(mesh, TH)

W0 = W.sub(0)
W1 = W.sub(1)

V, _ = W0.collapse()
Q, _ = W1.collapse()

# --------------------------------------------------
# 5. Boundary data
# --------------------------------------------------
mu = fem.Constant(mesh, PETSc.ScalarType(1.0))
f = fem.Constant(mesh, PETSc.ScalarType((0.0, 0.0, 0.0)))

U0 = 1.0

def inlet_velocity(x):
    values = np.zeros((3, x.shape[1]), dtype=PETSc.ScalarType)
    values[0] = U0
    values[1] = 0.0
    values[2] = 0.0
    return values

u_in = fem.Function(V)
u_in.interpolate(inlet_velocity)

u_zero = fem.Function(V)
u_zero.x.array[:] = 0.0

p_out = fem.Function(Q)
p_out.x.array[:] = 0.0

# --------------------------------------------------
# 6. Dofs on geometrically detected boundaries
# --------------------------------------------------
inlet_dofs = fem.locate_dofs_topological((W0, V), fdim, inlet_facets)
wall_dofs = fem.locate_dofs_topological((W0, V), fdim, wall_facets)
body_dofs = fem.locate_dofs_topological((W0, V), fdim, body_facets)
outlet_dofs = fem.locate_dofs_topological((W1, Q), fdim, outlet_facets)

if mesh.comm.rank == 0:
    print("Boundary dof counts:")
    print(f"  inlet velocity dofs  = {len(inlet_dofs)}")
    print(f"  wall velocity dofs   = {len(wall_dofs)}")
    print(f"  body velocity dofs   = {len(body_dofs)}")
    print(f"  outlet pressure dofs = {len(outlet_dofs)}")

bc_inlet = fem.dirichletbc(u_in, inlet_dofs, W0)

# First-pass farfield approximation:
# impose freestream velocity on outer box walls too
bc_walls = fem.dirichletbc(u_in, wall_dofs, W0)

# No-slip on aircraft
bc_body = fem.dirichletbc(u_zero, body_dofs, W0)

# Pressure reference at outlet
bc_outlet = fem.dirichletbc(p_out, outlet_dofs, W1)

bcs = [bc_inlet, bc_walls, bc_body, bc_outlet]

# --------------------------------------------------
# 7. Variational problem
# --------------------------------------------------
(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)

a = (
    mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    - ufl.inner(p, ufl.div(v)) * ufl.dx
    + ufl.inner(ufl.div(u), q) * ufl.dx
)

L = ufl.inner(f, v) * ufl.dx

if mesh.comm.rank == 0:
    print("Solving Stokes system...")

problem = fem.petsc.LinearProblem(
    a,
    L,
    bcs=bcs,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu"
    }
)

wh = problem.solve()
uh, ph = wh.split()
uh.name = "velocity"
ph.name = "pressure"

if mesh.comm.rank == 0:
    print("Solve complete.")

# --------------------------------------------------
# 8. Export
# --------------------------------------------------
with io.XDMFFile(mesh.comm, "velocity_3d.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(uh)

with io.XDMFFile(mesh.comm, "pressure_3d.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(ph)

if mesh.comm.rank == 0:
    print("Wrote velocity_3d.xdmf and pressure_3d.xdmf")
