import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, io, default_scalar_type
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from petsc4py import PETSc
import basix.ufl  # Essencial para o cG(1)cG(1) moderno

# ==========================================
# 1. Carregamento da Malha
# ==========================================
mesh_comm = MPI.COMM_WORLD
with io.XDMFFile(mesh_comm, "wing_domain.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")
domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

with io.XDMFFile(mesh_comm, "wing_facets.xdmf", "r") as xdmf:
    facet_tags = xdmf.read_meshtags(domain, name="Grid")

# ==========================================
# 2. Espaço de Função G2 (cG(1)cG(1))
# ==========================================
# Jansson's method: P1 para velocidade e P1 para pressão
v_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1, shape=(3,))
p_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
W = fem.functionspace(domain, basix.ufl.mixed_element([v_el, p_el]))

# ==========================================
# 3. Condições de Contorno (BCs) Fortes
# ==========================================
V, _ = W.sub(0).collapse()

# Velocidade da corrente livre (Freestream)
U_inf = 10.0
u_inlet = fem.Function(V)
u_inlet.interpolate(lambda x: np.vstack((np.full(x.shape[1], U_inf), np.zeros(x.shape[1]), np.zeros(x.shape[1]))))

# No método de Euler (Nitsche), aplicamos BC Forte APENAS na entrada (Inlet)
inlet_dofs = fem.locate_dofs_topological((W.sub(0), V), 2, facet_tags.find(3))
bc_inlet = fem.dirichletbc(u_inlet, inlet_dofs, W.sub(0))
bcs = [bc_inlet] 

# ==========================================
# 4. Formulação Variacional G2 (Euler via GLS)
# ==========================================
(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)

# Variáveis do passo de tempo anterior
w_old = fem.Function(W)
u_old, p_old = ufl.split(w_old)

# Inicializamos u_old com a velocidade do inlet para evitar instabilidade em t=0
# w_old.sub(0).interpolate(u_inlet) COMENTADO 

dt = fem.Constant(domain, default_scalar_type(0.005))  # PASSO DE TEMPO
h = ufl.CellDiameter(domain)

# Momentum Residual (Linearizado com u_old)
res_m = (u - u_old)/dt + ufl.dot(u_old, ufl.grad(u)) + ufl.grad(p)

# Parâmetro de Estabilização GLS (Delta)
delta = h / (2.0 * ufl.sqrt(ufl.inner(u_old, u_old) + (h/dt)**2))

# Forma Fraca Padrão G2
F = (ufl.inner((u - u_old)/dt + ufl.dot(u_old, ufl.grad(u)), v) - p * ufl.div(v) + q * ufl.div(u)) * ufl.dx

# Termo de Estabilização GLS
F += delta * ufl.inner(res_m, ufl.dot(u_old, ufl.grad(v)) + ufl.grad(q)) * ufl.dx

# Condição de Deslizamento de Nitsche (Slip BC: u . n = 0) na Asa (2) e Túnel (5)
n = ufl.FacetNormal(domain)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
gamma = fem.Constant(domain, default_scalar_type(10.0)) # Parâmetro de penalidade Nitsche

F += (gamma / h) * ufl.dot(u, n) * ufl.dot(v, n) * ds(2) # Asa
F += (gamma / h) * ufl.dot(u, n) * ufl.dot(v, n) * ds(5) # Túnel

# Separando equações bilineares (LHS) e lineares (RHS)
a = ufl.lhs(F)
L = ufl.rhs(F)

# ==========================================
# 5. Solver no Tempo (Pseudo-Transiente)
# ==========================================
a_form = fem.form(a)
L_form = fem.form(L)

w_h = fem.Function(W)

# Configuração do MUMPS (Direto)
solver = PETSc.KSP().create(domain.comm)
solver.setType(PETSc.KSP.Type.PREONLY)
pc = solver.getPC()
pc.setType(PETSc.PC.Type.LU)
pc.setFactorSolverType("mumps")

t = 0.0
T_final = 1.0 # Simular por 1 segundo físico (20 iterações)

print("Iniciando loop de tempo do solver G2 Euler...")
while t < T_final:
    t += dt.value
    print(f"Resolvendo t = {t:.3f} s")
    
    # Montamos a matriz A em cada passo porque u_old muda a matriz!
    A = assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    solver.setOperators(A)
    
    # Montamos o vetor L
    b = assemble_vector(L_form)
    fem.petsc.apply_lifting(b, [a_form], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, bcs)
    
    # Resolve e atualiza
    solver.solve(b, w_h.x.petsc_vec)
    
    # Atualiza o vetor do passado com a solução atual para o próximo loop
    w_old.x.array[:] = w_h.x.array[:]

# ==========================================
# 6. Extraindo e Salvando Resultados
# ==========================================
u_h = w_h.sub(0).collapse()
p_h = w_h.sub(1).collapse()
p_h.name = "Pressao"
u_h.name = "Velocidade"

print("Salvando arquivo XDMF...")
with io.XDMFFile(domain.comm, "results_euler.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u_h)
    xdmf.write_function(p_h)
    
# ==========================================
# 7. Cálculo das Forças (Arrasto Invísicido)
# ==========================================
print("\nCalculando forças aerodinâmicas (Regime Invísicido)...")

# Em Euler não tem viscosidade (mu = 0). A força na superfície é APENAS o campo de pressão.
T_x = (p_h * n[0]) * ds(2)
T_y = (p_h * n[1]) * ds(2)
T_z = (p_h * n[2]) * ds(2)

drag_val = domain.comm.allreduce(fem.assemble_scalar(fem.form(T_x)), op=MPI.SUM)
lift_val = domain.comm.allreduce(fem.assemble_scalar(fem.form(T_y)), op=MPI.SUM) 
lat_val  = domain.comm.allreduce(fem.assemble_scalar(fem.form(T_z)), op=MPI.SUM) 

print("="*40)
print(f"Arrasto (Drag) Invísicido [X]: {drag_val:.4f} N")
print(f"Sustentação (Lift) [Y]:        {lift_val:.4f} N")
print(f"Força Lateral (Side) [Z]:      {lat_val:.4f} N")
print("="*40)
print("Solver G2 Finalizado! O avião já pode decolar.")