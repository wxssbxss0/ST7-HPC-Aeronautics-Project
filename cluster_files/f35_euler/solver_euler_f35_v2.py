# ======================
# 0 - IMPORT LIBRARIES
# ======================

import numpy as np
import ufl
import time
from mpi4py import MPI
from dolfinx import fem, io, default_scalar_type
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.mesh import locate_entities_boundary, meshtags
from petsc4py import PETSc
import basix.ufl

# ======================
# CONFIG
# ======================

TEST_MODE = False       # <- Set True for local testing (10 steps only), False for cluster

if TEST_MODE:
    T_final        = 0.05
    dt_val         = 0.005
    SAVE_EVERY     = 5          # print diagnostics every N steps
    CHECKPOINT_SEC = None       # no checkpoints in test mode
    print(">>> RUNNING IN TEST_MODE — 10 steps only <<<")
else:
    T_final        = 2.0
    dt_val         = 0.005      # dt = 0.005s -> 400 total timesteps
    SAVE_EVERY     = 50         # print diagnostics every 50 steps (~every 0.25s physical)
    CHECKPOINT_SEC = 90 * 60   # wall-clock checkpoint every 90 minutes

# ==============================================
# 1 - LOAD MESH
# ==============================================

print("Loading mesh...")
mesh_comm = MPI.COMM_WORLD

with io.XDMFFile(mesh_comm, "f35_domain_lvl2.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")
domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
print("Mesh loaded.")

# ================================
# 2 - FUNCTION SPACES (cG1cG1)
# ================================

v_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1, shape=(3,))
p_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
W    = fem.functionspace(domain, basix.ufl.mixed_element([v_el, p_el]))

# ========================
# 3 - GEOMETRIC BOUNDARY MARKERS
# ========================
# KEY FIX: We recompute all boundary tags geometrically from scratch.
# This is the correct approach for the full F-35 mesh because the facet
# tags in the .xdmf file may use different numbering conventions than
# what we assumed in the original script. By recomputing geometrically,
# we are guaranteed to correctly identify inlet, outlet, tunnel walls,
# and the aircraft skin regardless of how the mesh was generated.

print("Computing geometric boundary markers...")

# Compute bounding box of the entire domain
x_min = domain.comm.allreduce(domain.geometry.x[:, 0].min(), op=MPI.MIN)
x_max = domain.comm.allreduce(domain.geometry.x[:, 0].max(), op=MPI.MAX)
y_min = domain.comm.allreduce(domain.geometry.x[:, 1].min(), op=MPI.MIN)
y_max = domain.comm.allreduce(domain.geometry.x[:, 1].max(), op=MPI.MAX)
z_min = domain.comm.allreduce(domain.geometry.x[:, 2].min(), op=MPI.MIN)
z_max = domain.comm.allreduce(domain.geometry.x[:, 2].max(), op=MPI.MAX)

# Use a relative tolerance: 0.1% of domain length in x
# This works regardless of mesh units (mm, m, etc.)
extent_x = x_max - x_min
tol = extent_x * 0.001

fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)

# Tag 1: Inlet — upstream face (x = x_min)
facets_inlet  = locate_entities_boundary(
    domain, fdim, lambda x: np.isclose(x[0], x_min, atol=tol))

# Tag 2: Outlet — downstream face (x = x_max)
facets_outlet = locate_entities_boundary(
    domain, fdim, lambda x: np.isclose(x[0], x_max, atol=tol))

# Tag 3: Tunnel walls — the 4 lateral faces of the wind tunnel box
facets_walls  = locate_entities_boundary(domain, fdim, lambda x: (
    np.isclose(x[1], y_min, atol=tol) | np.isclose(x[1], y_max, atol=tol) |
    np.isclose(x[2], z_min, atol=tol) | np.isclose(x[2], z_max, atol=tol)
))

# Tag 4: F-35 aircraft skin — everything that is NOT inlet, outlet, or tunnel walls
# This is the key fix: instead of relying on mesh file tags (which may be wrong),
# we identify the aircraft surface as the "leftover" boundary facets
all_boundary = locate_entities_boundary(
    domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
facets_f35   = np.setdiff1d(
    all_boundary,
    np.concatenate([facets_inlet, facets_outlet, facets_walls]))

# Sanity check — print facet counts to verify tags are non-empty
if domain.comm.rank == 0:
    print(f"  Inlet   : {len(facets_inlet)} facets")
    print(f"  Outlet  : {len(facets_outlet)} facets")
    print(f"  Walls   : {len(facets_walls)} facets")
    print(f"  F-35    : {len(facets_f35)} facets")
    if len(facets_inlet)  == 0: print("WARNING: Inlet not found! Check mesh orientation.")
    if len(facets_outlet) == 0: print("WARNING: Outlet not found! Check mesh orientation.")
    if len(facets_f35)    == 0: print("WARNING: F-35 surface not found! Slip BC will not be applied.")

# Assemble all facet tags into a single MeshTags object
all_facets = np.concatenate([facets_inlet, facets_outlet, facets_walls, facets_f35])
all_tags   = np.concatenate([
    np.full(len(facets_inlet),  1, dtype=np.int32),
    np.full(len(facets_outlet), 2, dtype=np.int32),
    np.full(len(facets_walls),  3, dtype=np.int32),
    np.full(len(facets_f35),    4, dtype=np.int32),
])
order          = np.argsort(all_facets)
facet_tags_geo = meshtags(domain, fdim, all_facets[order], all_tags[order])
ds             = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags_geo)

print("Geometric boundary markers done.")

# ========================
# 3b - BOUNDARY CONDITIONS
# ========================

V, _ = W.sub(0).collapse()

# Freestream velocity: U_inf = 10 m/s in the x-direction (cruise-like condition)
U_inf       = 10.0
u_inlet_val = fem.Function(V)
u_inlet_val.interpolate(lambda x: np.vstack((
    np.full(x.shape[1], U_inf),
    np.zeros(x.shape[1]),
    np.zeros(x.shape[1])
)))

# Apply Dirichlet BC strongly at the inlet
inlet_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_inlet)
bc_inlet   = fem.dirichletbc(u_inlet_val, inlet_dofs, W.sub(0))
bcs        = [bc_inlet]

# ==========================================
# 4 - VARIATIONAL FORMULATION G2 (EULER)
# ==========================================
# G2 method: Galerkin/Least-Squares stabilised equal-order (velocity, pressure)
# formulation for the incompressible Euler equations.
# Slip BCs on aircraft skin and tunnel walls are applied weakly via a
# penalty term on the normal velocity component (Nitsche-type).

(u, p)   = ufl.TrialFunctions(W)
(v, q)   = ufl.TestFunctions(W)
w_old    = fem.Function(W)
u_old, _ = ufl.split(w_old)

dt    = fem.Constant(domain, default_scalar_type(dt_val))
h     = ufl.CellDiameter(domain)
n     = ufl.FacetNormal(domain)

# Penalty parameter for slip condition (Nitsche method)
# gamma=10 is standard for G2; increase if normal velocity leaks through walls
gamma = fem.Constant(domain, default_scalar_type(10.0))

# Reference pressure at outlet — pins the pressure to avoid a singular system
p_ref = fem.Constant(domain, default_scalar_type(0.0))

# SUPG/GLS stabilisation parameter delta
# The (h/dt)^2 term prevents blow-up at t=0 when u_old=0 everywhere
res_m = (u - u_old)/dt + ufl.dot(u_old, ufl.grad(u)) + ufl.grad(p)
delta = h / (2.0 * ufl.sqrt(ufl.inner(u_old, u_old) + (h/dt)**2))

# Galerkin terms + GLS stabilisation
F  = (ufl.inner((u - u_old)/dt + ufl.dot(u_old, ufl.grad(u)), v)
      - p * ufl.div(v) + q * ufl.div(u)) * ufl.dx
F += delta * ufl.inner(res_m, ufl.dot(u_old, ufl.grad(v)) + ufl.grad(q)) * ufl.dx

# Slip condition on tunnel walls (tag 3): u.n = 0 weakly
F += (gamma / h) * ufl.dot(u, n) * ufl.dot(v, n) * ds(3)

# Slip condition on F-35 aircraft skin (tag 4): u.n = 0 weakly (Euler = no penetration)
F += (gamma / h) * ufl.dot(u, n) * ufl.dot(v, n) * ds(4)

# Outlet pressure reference (tag 2): natural outflow, p pinned to 0
F += p_ref * ufl.dot(v, n) * ds(2)

a = ufl.lhs(F)
L = ufl.rhs(F)

# =====================
# 5 - SOLVER
# =====================

print("Setting up solver...")
a_form = fem.form(a)
L_form = fem.form(L)
w_h    = fem.Function(W)

# Direct solver: MUMPS LU factorisation
# Best choice for moderate-size 3D problems; scales well up to ~32 MPI ranks
solver = PETSc.KSP().create(domain.comm)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)
solver.getPC().setFactorSolverType("mumps")

# =====================
# 6 - HELPER FUNCTIONS
# =====================

def save_results(w_h, domain, suffix=""):
    """Save final velocity and pressure fields to XDMF."""
    u_h, p_h = w_h.sub(0).collapse(), w_h.sub(1).collapse()
    u_h.name, p_h.name = "VELOCITY", "PRESSURE"
    fname = f"results_f35{suffix}.xdmf"
    with io.XDMFFile(domain.comm, fname, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(u_h)
        xdmf.write_function(p_h)
    if domain.comm.rank == 0:
        print(f"  -> Saved: {fname}")


def save_checkpoint(w_old, domain, t, step):
    """
    Save a mid-run checkpoint so we can resume or inspect intermediate results.
    Writes both an XDMF field file and a small .npy metadata file with t and step.
    """
    u_h, p_h = w_old.sub(0).collapse(), w_old.sub(1).collapse()
    u_h.name, p_h.name = "VELOCITY", "PRESSURE"
    fname = f"checkpoint_t{t:.3f}_step{step:06d}.xdmf"
    with io.XDMFFile(domain.comm, fname, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(u_h)
        xdmf.write_function(p_h)
    if domain.comm.rank == 0:
        np.save("checkpoint_meta.npy", np.array([t, step]))
        print(f"  -> Checkpoint saved: {fname}  (t={t:.3f}s, step={step})")

# =====================
# 7 - TIME LOOP
# =====================

print("Starting time loop...")
t                 = 0.0
step              = 0
wall_start        = time.time()
last_checkpoint   = wall_start
diverged          = False

if domain.comm.rank == 0:
    print(f"  Total timesteps planned: {int(T_final / dt_val)}")
    print(f"  dt = {dt_val}s  |  T_final = {T_final}s  |  U_inf = {U_inf} m/s")
    print("SIMULATION STARTING...")

while t < T_final:
    t    += dt.value
    step += 1

    # Assemble and solve linear system
    A = assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    solver.setOperators(A)

    b = assemble_vector(L_form)
    fem.petsc.apply_lifting(b, [a_form], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, bcs)

    solver.solve(b, w_h.x.petsc_vec)

    # Scatter to ghost nodes (critical for parallel correctness)
    w_h.x.scatter_forward()

    # Update solution for next timestep
    w_old.x.array[:] = w_h.x.array[:]

    # ---- DIAGNOSTICS ----
    # Print key physical quantities every SAVE_EVERY steps to catch divergence early
    if step % SAVE_EVERY == 0 and domain.comm.rank == 0:
        u_h_diag = w_h.sub(0).collapse()
        p_h_diag = w_h.sub(1).collapse()
        u_vals   = u_h_diag.x.array.reshape(-1, 3)
        u_mag    = np.linalg.norm(u_vals, axis=1)

        has_nan  = np.any(np.isnan(u_mag))
        has_inf  = np.any(np.isinf(u_mag))
        u_max    = float(np.max(u_mag))   if not (has_nan or has_inf) else float('nan')
        u_mean   = float(np.mean(u_mag))  if not (has_nan or has_inf) else float('nan')
        p_max    = float(np.max(np.abs(p_h_diag.x.array)))
        status   = "DIVERGED (NaN)!" if has_nan else ("DIVERGED (Inf)!" if has_inf else "OK")
        elapsed  = (time.time() - wall_start) / 60.0

        print(f"  Step {step:4d} | t={t:.3f}s | {status} | "
              f"u_max={u_max:.4f} m/s | u_mean={u_mean:.4f} m/s | "
              f"p_max={p_max:.4f} Pa | wall={elapsed:.1f}min")

        # Physical sanity check: u_max should be O(10) m/s, not O(1e8)
        if u_max > 1e6:
            print("  WARNING: u_max > 1e6 m/s — simulation likely diverging!")

        if has_nan or has_inf:
            print("  SOLVER DIVERGED — saving emergency checkpoint and aborting.")
            save_checkpoint(w_old, domain, t, step)
            diverged = True
            break
    # ----------------------

    # ---- WALL-CLOCK CHECKPOINT ----
    # Save progress every CHECKPOINT_SEC seconds of real time
    # so we don't lose work if the job is killed or times out
    if CHECKPOINT_SEC is not None:
        now = time.time()
        if now - last_checkpoint >= CHECKPOINT_SEC:
            save_checkpoint(w_old, domain, t, step)
            last_checkpoint = now
    # -------------------------------

# =====================
# 8 - SAVE FINAL RESULTS
# =====================

if not diverged:
    save_results(w_h, domain)
else:
    save_results(w_h, domain, suffix="_diverged")

wall_total = (time.time() - wall_start) / 60.0
if domain.comm.rank == 0:
    if diverged:
        print(f"SIMULATION ABORTED AT t={t:.3f}s after {wall_total:.1f} minutes.")
    else:
        print(f"SIMULATION COMPLETE. Physical time: {t:.3f}s | Wall time: {wall_total:.1f}min")
