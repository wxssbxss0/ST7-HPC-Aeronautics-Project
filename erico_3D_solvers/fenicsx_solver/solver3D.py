import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, io, default_scalar_type
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from petsc4py import PETSc
from basix.ufl import element, mixed_element

# 1. Carregamento da Malha
with io.XDMFFile(MPI.COMM_WORLD, "wing_domain.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="Grid")
domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
with io.XDMFFile(MPI.COMM_WORLD, "wing_facets.xdmf", "r") as xdmf:
    facet_tags = xdmf.read_meshtags(domain, name="Grid")

# 2. Espaço de Função
v_el = element("Lagrange", domain.topology.cell_name(), 2, shape=(3,)) 
p_el = element("Lagrange", domain.topology.cell_name(), 1)            
W = fem.functionspace(domain, mixed_element([v_el, p_el]))

# 3. BCs
V, _ = W.sub(0).collapse()
u_inlet = fem.Function(V)
u_inlet.interpolate(lambda x: np.vstack((np.full(x.shape[1], 10.0), np.zeros(x.shape[1]), np.zeros(x.shape[1]))))
u_zero = fem.Function(V)

wing_dofs = fem.locate_dofs_topological((W.sub(0), V), 2, facet_tags.find(2))
inlet_dofs = fem.locate_dofs_topological((W.sub(0), V), 2, facet_tags.find(3))
tunnel_dofs = fem.locate_dofs_topological((W.sub(0), V), 2, facet_tags.find(5))

bc_wing = fem.dirichletbc(u_zero, wing_dofs, W.sub(0))
bc_inlet = fem.dirichletbc(u_inlet, inlet_dofs, W.sub(0))
bc_tunnel = fem.dirichletbc(u_zero, tunnel_dofs, W.sub(0))

bcs = [bc_inlet, bc_tunnel, bc_wing]

# 4. Formulação Variacional
(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)
f = fem.Constant(domain, default_scalar_type((0, 0, 0)))
mu = fem.Constant(domain, default_scalar_type(1.0))

a = (mu * ufl.inner(ufl.grad(u), ufl.grad(v)) - p * ufl.div(v) + q * ufl.div(u)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

# 5. Montagem e Solver
print("Montando as matrizes do sistema...")
a_form = fem.form(a)
L_form = fem.form(L)

A = assemble_matrix(a_form, bcs=bcs)
A.assemble()

b = assemble_vector(L_form)
fem.petsc.apply_lifting(b, [a_form], [bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
fem.petsc.set_bc(b, bcs)

print("Configurando o Solver MUMPS...")
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
pc = solver.getPC()
pc.setType(PETSc.PC.Type.LU)
pc.setFactorSolverType("mumps") 

w_h = fem.Function(W)
print("Resolvendo a física...")
solver.solve(b, w_h.x.petsc_vec)

# 6. Extraindo e Salvando Resultados
u_h = w_h.sub(0).collapse()
p_h = w_h.sub(1).collapse()
p_h.name = "Pressao"

v_el_p1 = element("Lagrange", domain.topology.cell_name(), 1, shape=(3,))
V_p1 = fem.functionspace(domain, v_el_p1)
u_out = fem.Function(V_p1)
u_out.name = "Velocidade"
u_out.interpolate(u_h) 

print("Salvando arquivo XDMF...")
with io.XDMFFile(domain.comm, "results.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u_out)
    xdmf.write_function(p_h)
    
# 7. Cálculo das Forças
print("\nCalculando forças aerodinâmicas...")
n = ufl.FacetNormal(domain)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

# FÓRMULA CORRETA (Força DO fluido SOBRE a asa)
T_x = (p_h * n[0] - mu * ufl.dot(ufl.grad(u_h), n)[0]) * ds(2)
T_y = (p_h * n[1] - mu * ufl.dot(ufl.grad(u_h), n)[1]) * ds(2)
T_z = (p_h * n[2] - mu * ufl.dot(ufl.grad(u_h), n)[2]) * ds(2)

# EIXOS CORRIGIDOS DO CADQUERY
drag_val = domain.comm.allreduce(fem.assemble_scalar(fem.form(T_x)), op=MPI.SUM)
lift_val = domain.comm.allreduce(fem.assemble_scalar(fem.form(T_y)), op=MPI.SUM) # Y é o UP/DOWN!
lat_val  = domain.comm.allreduce(fem.assemble_scalar(fem.form(T_z)), op=MPI.SUM) # Z é a envergadura!

print("="*40)
print(f"Arrasto (Drag) [Eixo X]:       {drag_val:.4f} N")
print(f"Sustentação (Lift) [Eixo Y]:   {lift_val:.4f} N")
print(f"Força Lateral (Side) [Eixo Z]: {lat_val:.4f} N")
print("="*40)
print("Sucesso! Pode abrir no ParaView.")