import numpy as np
from dolfinx.io import gmsh as dolfinx_gmsh
import gmsh
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import basix.ufl
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem


def create_mesh_in_memory():
    """Generates the channel with a circle directly in memory using Gmsh."""
    gmsh.initialize()
    gmsh.model.add("channel")
    
    # Define geometry
    rect = gmsh.model.occ.addRectangle(0, 0, 0, 4.0, 1.0)
    circle = gmsh.model.occ.addDisk(1.0, 0.5, 0, 0.2, 0.2)
    fluid, _ = gmsh.model.occ.cut([(2, rect)], [(2, circle)])
    gmsh.model.occ.synchronize()

    gmsh.model.occ.synchronize()
    
    # --- ADD THESE TWO LINES ---
    # fluid[0][1] extracts the integer tag of the 2D surface we just created
    gmsh.model.addPhysicalGroup(2, [fluid[0][1]], tag=1)
    gmsh.model.setPhysicalName(2, 1, "FluidDomain")
    # ---------------------------

    # Mesh generation
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.05)
    
    # Mesh generation
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.05)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.05)
    gmsh.model.mesh.generate(2)
    
    # Convert directly to FEniCSx mesh
    # Convert directly to FEniCSx mesh (dolfinx 0.10+ syntax)
    mesh_data = dolfinx_gmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    
    # Extract the actual mesh from the new MeshData object
    msh = mesh_data.mesh
    
    gmsh.finalize()
    return msh
# 1. Load Mesh
msh = create_mesh_in_memory()

# 2. Define Taylor-Hood Elements (P2-P1) using modern basix API
v_el = basix.ufl.element("Lagrange", msh.topology.cell_name(), 2, shape=(msh.geometry.dim,))
p_el = basix.ufl.element("Lagrange", msh.topology.cell_name(), 1) # <--- This line was missing!
def circle_loc(x):  return np.isclose(np.sqrt((x[0] - 1.0)**2 + (x[1] - 0.5)**2), 0.2, atol=1e-3)
W_el = basix.ufl.mixed_element([v_el, p_el])
W = fem.functionspace(msh, W_el)

# 3. Locate boundaries mathematically (Avoids Gmsh tag errors)
def inlet_loc(x):   return np.isclose(x[0], 0.0)
def outlet_loc(x):  return np.isclose(x[0], 4.0)
def wall_loc(x):    return np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))
# Added absolute tolerance (atol) because Gmsh circles are approximations
def circle_loc(x):  return np.isclose(np.sqrt((x[0] - 1.0)**2 + (x[1] - 0.5)**2), 0.2, atol=1e-3)

# Subspaces for BCs
V, _ = W.sub(0).collapse()
Q, _ = W.sub(1).collapse()

# Parabolic Inlet Profile: u(y) = 4 * U_max * y * (H - y) / H^2
inlet_velocity = fem.Function(V)
inlet_velocity.interpolate(lambda x: np.vstack((4.0 * 1.0 * x[1] * (1.0 - x[1]), np.zeros_like(x[0]))))

# Create proper fem.Functions for the zero conditions
zero_velocity = fem.Function(V)
zero_velocity.x.array[:] = 0.0  # No-slip condition (u=0)

zero_pressure = fem.Function(Q)
zero_pressure.x.array[:] = 0.0  # Zero pressure at outlet (p=0)

# Find degrees of freedom (DOFs)
inlet_dofs = fem.locate_dofs_geometrical((W.sub(0), V), inlet_loc)
wall_dofs = fem.locate_dofs_geometrical((W.sub(0), V), wall_loc)
circle_dofs = fem.locate_dofs_geometrical((W.sub(0), V), circle_loc)
outlet_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), outlet_loc)

# Create Dirichlet Boundary Conditions
bcs = [
    fem.dirichletbc(inlet_velocity, inlet_dofs, W.sub(0)),
    fem.dirichletbc(zero_velocity, wall_dofs, W.sub(0)),
    fem.dirichletbc(zero_velocity, circle_dofs, W.sub(0)),
    fem.dirichletbc(zero_pressure, outlet_dofs, W.sub(1))
]
# 4. Variational Formulation (Stokes)
u, p = ufl.TrialFunctions(W)
v, q = ufl.TestFunctions(W)
mu = fem.Constant(msh, PETSc.ScalarType(1.0)) # Viscosity

# Weak form: mu * grad(u):grad(v) - p * div(v) + q * div(u) = 0
a = (mu * ufl.inner(ufl.grad(u), ufl.grad(v)) - p * ufl.div(v) + q * ufl.div(u)) * ufl.dx
L = ufl.inner(fem.Constant(msh, PETSc.ScalarType((0.0, 0.0))), v) * ufl.dx

# 5. Solve the Linear System
# 5. Solve the Linear System
problem = LinearProblem(
    a, L, bcs=bcs, 
    petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
    petsc_options_prefix="stokes_"  # <--- THIS IS THE FIX
)
w_h = problem.solve()

# 6. Extract Solutions and Export to XDMF
u_h = w_h.sub(0).collapse()
p_h = w_h.sub(1).collapse()
p_h.name = "Pressure"

# --- THE FIX: Interpolate P2 velocity to P1 for XDMF export ---
# Create a P1 vector element (same degree as the mesh)
v_p1 = basix.ufl.element("Lagrange", msh.topology.cell_name(), 1, shape=(msh.geometry.dim,))
V_save = fem.functionspace(msh, v_p1)

u_save = fem.Function(V_save)
u_save.name = "Velocity"
u_save.interpolate(u_h) # Downsample P2 to P1 for visualization only
# --------------------------------------------------------------

with io.XDMFFile(msh.comm, "stokes_results.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)
    xdmf.write_function(u_save) # Save the P1 velocity
    xdmf.write_function(p_h)    # Save the natively P1 pressure

print("Simulation complete. Results saved to stokes_results.xdmf")
