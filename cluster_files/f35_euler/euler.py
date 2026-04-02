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

TEST_MODE = False       # <- True local, False no cluster

if TEST_MODE:
    T_final        = 0.05
    dt_val         = 0.005
    SAVE_EVERY     = 5
    CHECKPOINT_SEC = None
    print(">>> RODANDO EM TEST_MODE — 10 passos apenas <<<")
else:
    T_final        = 2.0
    dt_val         = 0.005
    SAVE_EVERY     = 50
    CHECKPOINT_SEC = 90 * 60   # checkpoint a cada 1h30 (segundos)

# ==============================================
# 1 - LOAD MESH
# ==============================================
print('carregando mesh')
mesh_comm = MPI.COMM_WORLD

with io.XDMFFile(mesh_comm, "f35_domain_lvl2.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")
domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
print('mesh carregada')
# ================================
# 2 - FUNCTION SPACES (cG1cG1)
# ================================

v_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1, shape=(3,))
p_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
W    = fem.functionspace(domain, basix.ufl.mixed_element([v_el, p_el]))

# ========================
# 3 - MARCADORES GEOMÉTRICOS
# ========================
print('iniciando marcadores geometricos')
x_min = domain.comm.allreduce(domain.geometry.x[:, 0].min(), op=MPI.MIN)
x_max = domain.comm.allreduce(domain.geometry.x[:, 0].max(), op=MPI.MAX)
y_min = domain.comm.allreduce(domain.geometry.x[:, 1].min(), op=MPI.MIN)
y_max = domain.comm.allreduce(domain.geometry.x[:, 1].max(), op=MPI.MAX)
z_min = domain.comm.allreduce(domain.geometry.x[:, 2].min(), op=MPI.MIN)
z_max = domain.comm.allreduce(domain.geometry.x[:, 2].max(), op=MPI.MAX)
# Tolerância relativa — funciona em qualquer escala (mm, m, etc.)
extent_x = x_max - x_min
tol = extent_x * 0.001   # 0.1% do comprimento total

fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)

facets_inlet  = locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], x_min, atol=tol))
facets_outlet = locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], x_max, atol=tol))
facets_walls  = locate_entities_boundary(domain, fdim, lambda x: (
    np.isclose(x[1], y_min, atol=tol) | np.isclose(x[1], y_max, atol=tol) |
    np.isclose(x[2], z_min, atol=tol) | np.isclose(x[2], z_max, atol=tol)
))

all_boundary = locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
facets_f35   = np.setdiff1d(all_boundary, np.concatenate([facets_inlet, facets_outlet, facets_walls]))

if domain.comm.rank == 0:
    print(f"Inlet  : {len(facets_inlet)} facetas")
    print(f"Outlet : {len(facets_outlet)} facetas")
    print(f"Paredes: {len(facets_walls)} facetas")
    print(f"F-35   : {len(facets_f35)} facetas")
    if len(facets_inlet)  == 0: print("🚨 INLET não encontrado!")
    if len(facets_outlet) == 0: print("🚨 OUTLET não encontrado!")
    if len(facets_f35)    == 0: print("🚨 F-35 não encontrado!")

all_facets = np.concatenate([facets_inlet, facets_outlet, facets_walls, facets_f35])
all_tags   = np.concatenate([
    np.full(len(facets_inlet),  1, dtype=np.int32),
    np.full(len(facets_outlet), 2, dtype=np.int32),
    np.full(len(facets_walls),  3, dtype=np.int32),
    np.full(len(facets_f35),    4, dtype=np.int32),
])
ordem          = np.argsort(all_facets)
facet_tags_geo = meshtags(domain, fdim, all_facets[ordem], all_tags[ordem])
ds             = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags_geo)
print('marcadores geometricos finalizados')

# ========================
# 3b - BOUNDARY CONDITIONS
# ========================

V, _ = W.sub(0).collapse()

U_inf       = 10.0
u_inlet_val = fem.Function(V)
u_inlet_val.interpolate(lambda x: np.vstack((
    np.full(x.shape[1], U_inf),
    np.zeros(x.shape[1]),
    np.zeros(x.shape[1])
)))

inlet_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_inlet)
bc_inlet   = fem.dirichletbc(u_inlet_val, inlet_dofs, W.sub(0))
bcs        = [bc_inlet]

# ==========================================
# 4 - VARIATIONAL FORMULATION G2 (EULER)
# ==========================================

(u, p)   = ufl.TrialFunctions(W)
(v, q)   = ufl.TestFunctions(W)
w_old    = fem.Function(W)
u_old, _ = ufl.split(w_old)

dt    = fem.Constant(domain, default_scalar_type(dt_val))
h     = ufl.CellDiameter(domain)
n     = ufl.FacetNormal(domain)
gamma = fem.Constant(domain, default_scalar_type(10.0))
p_ref = fem.Constant(domain, default_scalar_type(0.0))

res_m = (u - u_old)/dt + ufl.dot(u_old, ufl.grad(u)) + ufl.grad(p)
delta = h / (2.0 * ufl.sqrt(ufl.inner(u_old, u_old) + (h/dt)**2))

F  = (ufl.inner((u - u_old)/dt + ufl.dot(u_old, ufl.grad(u)), v)
      - p * ufl.div(v) + q * ufl.div(u)) * ufl.dx
F += delta * ufl.inner(res_m, ufl.dot(u_old, ufl.grad(v)) + ufl.grad(q)) * ufl.dx
F += (gamma / h) * ufl.dot(u, n) * ufl.dot(v, n) * ds(3)   # paredes túnel
F += (gamma / h) * ufl.dot(u, n) * ufl.dot(v, n) * ds(4)   # skin F-35
F += p_ref * ufl.dot(v, n) * ds(2)                          # outlet

a = ufl.lhs(F)
L = ufl.rhs(F)

# =====================
# 5 - SOLVER
# =====================
print('iniciando solver')
a_form = fem.form(a)
L_form = fem.form(L)
w_h    = fem.Function(W)

solver = PETSc.KSP().create(domain.comm)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)
solver.getPC().setFactorSolverType("mumps")

# =====================
# 6 - FUNÇÕES AUXILIARES
# =====================

def save_results(w_h, domain, suffix=""):
    u_h, p_h = w_h.sub(0).collapse(), w_h.sub(1).collapse()
    u_h.name, p_h.name = "VELOCITY", "PRESSURE"
    fname = f"results_f35{suffix}.xdmf"
    with io.XDMFFile(domain.comm, fname, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(u_h)
        xdmf.write_function(p_h)
    if domain.comm.rank == 0:
        print(f"  -> Salvo: {fname}")

def save_checkpoint(w_old, domain, t, passo):
    u_h, p_h = w_old.sub(0).collapse(), w_old.sub(1).collapse()
    u_h.name, p_h.name = "VELOCITY", "PRESSURE"
    fname = f"checkpoint_t{t:.3f}_passo{passo}.xdmf"
    with io.XDMFFile(domain.comm, fname, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(u_h)
        xdmf.write_function(p_h)
    if domain.comm.rank == 0:
        np.save("checkpoint_meta.npy", np.array([t, passo]))
        print(f"  -> Checkpoint salvo: {fname} (t={t:.3f}s, passo={passo})")

# =====================
# 7 - TIME LOOP
# =====================
print('iniciando time loop')
t                 = 0.0
passo             = 0
tempo_inicio      = time.time()
ultimo_checkpoint = tempo_inicio

if domain.comm.rank == 0:
    print("INICIANDO SIMULAÇÃO...")

while t < T_final:
    t     += dt.value
    passo += 1

    A = assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    solver.setOperators(A)

    b = assemble_vector(L_form)
    fem.petsc.apply_lifting(b, [a_form], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, bcs)

    solver.solve(b, w_h.x.petsc_vec)
    w_old.x.array[:] = w_h.x.array[:]

    # ---- DIAGNÓSTICO ----
    if passo % SAVE_EVERY == 0 and domain.comm.rank == 0:
        u_h_diag, p_h_diag = w_h.sub(0).collapse(), w_h.sub(1).collapse()
        u_vals = u_h_diag.x.array.reshape(-1, 3)
        u_mag  = np.linalg.norm(u_vals, axis=1)

        tem_nan = np.any(np.isnan(u_mag))
        tem_inf = np.any(np.isinf(u_mag))
        u_max   = np.max(u_mag)  if not (tem_nan or tem_inf) else float('nan')
        u_mean  = np.mean(u_mag) if not (tem_nan or tem_inf) else float('nan')
        p_max   = np.max(np.abs(p_h_diag.x.array))
        status  = "💀 NaN!" if tem_nan else ("💀 Inf!" if tem_inf else "✅ OK")
        elapsed = (time.time() - tempo_inicio) / 60.0

        print(f"Passo {passo:4d} | t={t:.3f}s | {status} | "
              f"u_max={u_max:.4f} | u_mean={u_mean:.4f} | "
              f"p_max={p_max:.4f} | wall={elapsed:.1f}min")

        if tem_nan or tem_inf:
            print("SOLVER DIVERGIU — abortando.")
            break
    # ----------------------

    # ---- CHECKPOINT ----
    if CHECKPOINT_SEC is not None:
        agora = time.time()
        if agora - ultimo_checkpoint >= CHECKPOINT_SEC:
            save_checkpoint(w_old, domain, t, passo)
            ultimo_checkpoint = agora
    # --------------------

# =====================
# 8 - SALVAR RESULTADO FINAL
# =====================

save_results(w_h, domain)
if domain.comm.rank == 0:
    print("SIMULAÇÃO FINALIZADA.")