# ======================
# 0 - IMPORT LIBRARIES
# ======================

import time
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, io, default_scalar_type
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from petsc4py import PETSc
import basix.ufl

# ==============================================
# 1 - LOAD MESHES
# ==============================================

mesh_comm = MPI.COMM_WORLD

# READ VOLUME
with io.XDMFFile(mesh_comm, "f35_domain_lvl2.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")
domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

# READ SURFACE
with io.XDMFFile(mesh_comm, "f35_facets_lvl2.xdmf", "r") as xdmf:
    facet_tags = xdmf.read_meshtags(domain, name="Grid")

# ================================
# 2 - FUNCTION SPACES (cG(1)cG(1))
# ================================

v_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1, shape=(3,))
p_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
W = fem.functionspace(domain, basix.ufl.mixed_element([v_el, p_el]))

# ========================
# 3 - BOUNDARY CONDITIONS
# ========================

V, _ = W.sub(0).collapse()

# INLET WIND VELOCITY
U_inf = 10.0
u_inlet_val = fem.Function(V)
u_inlet_val.interpolate(lambda x: np.vstack((
    np.full(x.shape[1], U_inf),
    np.zeros(x.shape[1]),
    np.zeros(x.shape[1])
)))

# LOCATE INLET (x-min face = upstream inflow boundary)
x_min_all = domain.comm.allreduce(domain.geometry.x[:, 0].min(), op=MPI.MIN)
def inlet_marker(x):
    return np.isclose(x[0], x_min_all, atol=1e-2)

inlet_dofs = fem.locate_dofs_geometrical((W.sub(0), V), inlet_marker)
bc_inlet = fem.dirichletbc(u_inlet_val, inlet_dofs, W.sub(0))

bcs = [bc_inlet]

# ==========================================
# 4 - VARIATIONAL FORMULATION G2 (EULER)
# ==========================================

(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)

w_old = fem.Function(W)
u_old, p_old = ufl.split(w_old)

dt = fem.Constant(domain, default_scalar_type(0.005))
h  = ufl.CellDiameter(domain)

# STREAMLINE UPWIND PETROV-GALERKIN (SUPG/GLS) STABILISATION PARAMETER
# Safeguard: avoid division by zero if u_old = 0 at t=0
eps  = fem.Constant(domain, default_scalar_type(1e-10))
u_norm = ufl.sqrt(ufl.inner(u_old, u_old) + eps + (h / dt) ** 2)
delta  = h / (2.0 * u_norm)

# MOMENTUM RESIDUAL (strong form, used in stabilisation)
res_m = (u - u_old) / dt + ufl.dot(u_old, ufl.grad(u)) + ufl.grad(p)

# G2 GALERKIN-LEAST-SQUARES FORM
F  = (ufl.inner((u - u_old) / dt + ufl.dot(u_old, ufl.grad(u)), v)
      - p * ufl.div(v)
      + q * ufl.div(u)) * ufl.dx
F += delta * ufl.inner(res_m,
                        ufl.dot(u_old, ufl.grad(v)) + ufl.grad(q)) * ufl.dx

# SLIP CONDITION ON AIRCRAFT SURFACE (facet tag 2)
# Penalises normal velocity to enforce u·n = 0 weakly
n     = ufl.FacetNormal(domain)
ds    = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
gamma = fem.Constant(domain, default_scalar_type(10.0))

F += (gamma / h) * ufl.dot(u, n) * ufl.dot(v, n) * ds(2)

a = ufl.lhs(F)
L = ufl.rhs(F)

# =====================
# 5 - SOLVER SETUP
# =====================

a_form = fem.form(a)
L_form = fem.form(L)
w_h    = fem.Function(W)

solver = PETSc.KSP().create(domain.comm)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)
solver.getPC().setFactorSolverType("mumps")


# =====================
# 6 - CHECKPOINT HELPER
# =====================

CHECKPOINT_INTERVAL_S = 90 * 60  # save every 90 minutes of wall-clock time

def save_checkpoint(w_h, domain, step, t, label="checkpoint"):
    """Write velocity + pressure + step metadata to an XDMF checkpoint file."""
    u_h, p_h = w_h.sub(0).collapse(), w_h.sub(1).collapse()
    u_h.name = "VELOCITY"
    p_h.name = "PRESSURE"
    fname = f"checkpoint_f35_{label}.xdmf"
    with io.XDMFFile(domain.comm, fname, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(u_h)
        xdmf.write_function(p_h)
    if domain.comm.rank == 0:
        # Write a tiny metadata file alongside so we can resume later
        with open(f"checkpoint_f35_{label}.txt", "w") as f:
            f.write(f"step={step}\n")
            f.write(f"t={t:.6f}\n")
        print(f"[CHECKPOINT] Saved '{fname}' at t={t:.4f}s  (step {step})")


# =====================
# 7 - TIME LOOP
# =====================

t       = 0.0
T_final = 2.0          # physical seconds to simulate
step    = 0

wall_start         = time.time()
last_checkpoint_wt = wall_start

print("INITIALIZING SIMULATION...")

try:
    while t < T_final:
        t    += dt.value
        step += 1

        # ---- assemble & solve ----
        A = assemble_matrix(a_form, bcs=bcs)
        A.assemble()
        solver.setOperators(A)

        b = assemble_vector(L_form)
        fem.petsc.apply_lifting(b, [a_form], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, bcs)

        solver.solve(b, w_h.x.petsc_vec)

        # Scatter to ghost nodes (important for parallel correctness)
        w_h.x.scatter_forward()

        w_old.x.array[:] = w_h.x.array[:]

        # ---- progress log ----
        if step % 10 == 0:
            wall_elapsed = time.time() - wall_start
            print(f"  t={t:.3f}s  step={step}"
                  f"  wall={wall_elapsed/60:.1f}min")

        # ---- wall-clock checkpoint ----
        wall_now = time.time()
        if wall_now - last_checkpoint_wt >= CHECKPOINT_INTERVAL_S:
            save_checkpoint(w_h, domain, step, t,
                            label=f"step{step:06d}")
            last_checkpoint_wt = wall_now

except KeyboardInterrupt:
    # Graceful exit: dump whatever we have if the job is killed
    if domain.comm.rank == 0:
        print("\n[INTERRUPTED] Saving emergency checkpoint...")
    save_checkpoint(w_h, domain, step, t, label="interrupted")
    raise


# ============================================
# 8 - SAVE FINAL RESULTS (results_f35.xdmf)
# ============================================

u_h, p_h = w_h.sub(0).collapse(), w_h.sub(1).collapse()
u_h.name = "VELOCITY"
p_h.name = "PRESSURE"

with io.XDMFFile(domain.comm, "results_f35.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u_h)
    xdmf.write_function(p_h)

wall_total = (time.time() - wall_start) / 60
if domain.comm.rank == 0:
    print(f"\nSIMULATION FINISHED  |  physical time simulated: {t:.3f}s"
          f"  |  wall time: {wall_total:.1f}min")