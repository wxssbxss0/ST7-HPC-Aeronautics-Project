"""
solver3D_corsair.py — Steady Stokes CFD Solver for the Corsair 3D simulation.
Adapted from the F35 wing solver (solver3D.py) by [teammate].

Physics:   Steady incompressible Stokes equations (linear, viscous-dominated)
Solver:    MUMPS direct solver via PETSc (LU factorisation)
Outputs:   corsair_results.xdmf  — velocity + pressure fields (open in ParaView)
           Terminal output        — Drag, Lift, Side-force on the Corsair skin

Boundary tag convention (must match convert_corsair.py and your Gmsh script):
    TAG_CORSAIR = 1  →  Corsair skin     — no-slip wall  (u = 0)
    TAG_INLET   = 2  →  Inflow face      — prescribed uniform velocity
    TAG_OUTLET  = 3  →  Outflow face     — free (natural Neumann BC)
    TAG_WALLS   = 4  →  Box side faces   — free-slip / symmetry (u·n = 0 weakly)

Physical parameters:
    U_INF = 10.0 m/s   — free-stream speed (adjust to your test condition)
    MU    =  1.0 Pa·s  — dynamic viscosity (set to air ~1.8e-5 for real units)
"""

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, io, default_scalar_type
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from petsc4py import PETSc
from basix.ufl import element, mixed_element

# ---------------------------------------------------------------------------
# Tag constants — keep in sync with convert_corsair.py
# ---------------------------------------------------------------------------
TAG_CORSAIR = 1
TAG_INLET   = 2
TAG_OUTLET  = 3
TAG_WALLS   = 4

# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------
U_INF = 10.0   # [m/s]  free-stream inlet velocity along X
MU    = 1.0    # [Pa·s] dynamic viscosity (Stokes regime — change for real air)

# ---------------------------------------------------------------------------
# 1. Load Mesh
# ---------------------------------------------------------------------------
with io.XDMFFile(MPI.COMM_WORLD, "corsair_domain.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")

# Build facet-to-cell connectivity (required for boundary-condition look-ups)
domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

with io.XDMFFile(MPI.COMM_WORLD, "corsair_facets.xdmf", "r") as xdmf:
    facet_tags = xdmf.read_meshtags(domain, name="Grid")

# ---------------------------------------------------------------------------
# 2. Function Space  —  Taylor-Hood P2/P1 (velocity P2, pressure P1)
# ---------------------------------------------------------------------------
v_el = element("Lagrange", domain.topology.cell_name(), 2, shape=(3,))  # velocity
p_el = element("Lagrange", domain.topology.cell_name(), 1)               # pressure
W = fem.functionspace(domain, mixed_element([v_el, p_el]))

# ---------------------------------------------------------------------------
# 3. Boundary Conditions
# ---------------------------------------------------------------------------
V, _ = W.sub(0).collapse()

# -- Inlet: uniform flow in the +X direction ---------------------------------
u_inlet = fem.Function(V)
u_inlet.interpolate(
    lambda x: np.vstack((
        np.full(x.shape[1], U_INF),   # u_x = U_INF
        np.zeros(x.shape[1]),          # u_y = 0
        np.zeros(x.shape[1]),          # u_z = 0
    ))
)

# -- No-slip zero-velocity (used on Corsair skin and box walls) --------------
u_zero = fem.Function(V)   # defaults to zero everywhere

# Locate DOFs for each tagged boundary patch
corsair_dofs = fem.locate_dofs_topological((W.sub(0), V), 2, facet_tags.find(TAG_CORSAIR))
inlet_dofs   = fem.locate_dofs_topological((W.sub(0), V), 2, facet_tags.find(TAG_INLET))
walls_dofs   = fem.locate_dofs_topological((W.sub(0), V), 2, facet_tags.find(TAG_WALLS))
# TAG_OUTLET is left as a natural (Neumann) BC — no Dirichlet needed

bc_corsair = fem.dirichletbc(u_zero,   corsair_dofs, W.sub(0))  # no-slip on plane skin
bc_inlet   = fem.dirichletbc(u_inlet,  inlet_dofs,   W.sub(0))  # prescribed inflow
bc_walls   = fem.dirichletbc(u_zero,   walls_dofs,   W.sub(0))  # no-slip box walls
                                                                  # (swap for slip if preferred)

bcs = [bc_inlet, bc_walls, bc_corsair]

# ---------------------------------------------------------------------------
# 4. Variational Formulation  —  Steady Stokes
#    μ ∫ ∇u:∇v dx  −  ∫ p ∇·v dx  +  ∫ q ∇·u dx  =  ∫ f·v dx
# ---------------------------------------------------------------------------
(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)

f  = fem.Constant(domain, default_scalar_type((0, 0, 0)))        # body force (gravity off)
mu = fem.Constant(domain, default_scalar_type(MU))

a = (
      mu * ufl.inner(ufl.grad(u), ufl.grad(v))   # viscous stress
    - p  * ufl.div(v)                              # pressure-velocity coupling
    + q  * ufl.div(u)                              # incompressibility
) * ufl.dx

L = ufl.inner(f, v) * ufl.dx

# ---------------------------------------------------------------------------
# 5. Assembly and Solver
# ---------------------------------------------------------------------------
print("Assembling system matrices...")
a_form = fem.form(a)
L_form = fem.form(L)

A = assemble_matrix(a_form, bcs=bcs)
A.assemble()

b = assemble_vector(L_form)
fem.petsc.apply_lifting(b, [a_form], [bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
fem.petsc.set_bc(b, bcs)

print("Configuring MUMPS direct solver (LU)...")
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
pc = solver.getPC()
pc.setType(PETSc.PC.Type.LU)
pc.setFactorSolverType("mumps")

w_h = fem.Function(W)
print("Solving the flow field...")
solver.solve(b, w_h.x.petsc_vec)

# ---------------------------------------------------------------------------
# 6. Extract and Save Results
# ---------------------------------------------------------------------------
u_h = w_h.sub(0).collapse()
p_h = w_h.sub(1).collapse()
p_h.name = "Pressure"

# Interpolate velocity onto P1 for lighter ParaView output
v_el_p1 = element("Lagrange", domain.topology.cell_name(), 1, shape=(3,))
V_p1  = fem.functionspace(domain, v_el_p1)
u_out = fem.Function(V_p1)
u_out.name = "Velocity"
u_out.interpolate(u_h)

print("Saving results to corsair_results.xdmf ...")
with io.XDMFFile(domain.comm, "corsair_results.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u_out)
    xdmf.write_function(p_h)

# ---------------------------------------------------------------------------
# 7. Aerodynamic Force Calculation on the Corsair Skin
#
#    Traction vector on a surface with outward normal n:
#        t = p·n  −  μ (∇u)·n
#    Integrating over TAG_CORSAIR gives the total fluid force on the plane.
#
#    Axis convention (inherited from your CadQuery/Gmsh model):
#        X  →  streamwise   (Drag)
#        Y  →  vertical up  (Lift)
#        Z  →  spanwise     (Side force)
# ---------------------------------------------------------------------------
print("\nCalculating aerodynamic forces on Corsair skin...")

n  = ufl.FacetNormal(domain)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

# Force components: fluid traction integrated over the Corsair surface
T_x = (p_h * n[0] - mu * ufl.dot(ufl.grad(u_h), n)[0]) * ds(TAG_CORSAIR)
T_y = (p_h * n[1] - mu * ufl.dot(ufl.grad(u_h), n)[1]) * ds(TAG_CORSAIR)
T_z = (p_h * n[2] - mu * ufl.dot(ufl.grad(u_h), n)[2]) * ds(TAG_CORSAIR)

drag_val = domain.comm.allreduce(fem.assemble_scalar(fem.form(T_x)), op=MPI.SUM)
lift_val = domain.comm.allreduce(fem.assemble_scalar(fem.form(T_y)), op=MPI.SUM)
lat_val  = domain.comm.allreduce(fem.assemble_scalar(fem.form(T_z)), op=MPI.SUM)

print("=" * 45)
print(f"  Drag  (X-axis — streamwise) : {drag_val:>12.4f} N")
print(f"  Lift  (Y-axis — vertical)   : {lift_val:>12.4f} N")
print(f"  Side  (Z-axis — spanwise)   : {lat_val:>12.4f}  N")
print("=" * 45)
print("Done! Open corsair_results.xdmf in ParaView.")
