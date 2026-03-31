import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import basix.ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem

def solve_and_compute_error(h):
    # 1. Generate empty rectangular channel
    nx, ny = int(4.0 / h), int(1.0 / h)
    msh = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([4.0, 1.0])],
        [nx, ny],
        cell_type=mesh.CellType.triangle
    )

    # 2. Function Spaces (Taylor-Hood P2-P1)
    v_el = basix.ufl.element("Lagrange", msh.topology.cell_name(), 2, shape=(msh.geometry.dim,))
    p_el = basix.ufl.element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix.ufl.mixed_element([v_el, p_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    # 3. Method of Manufactured Solutions (MMS)
    # We use a Sine wave instead of a parabola so P2 elements actually produce an error
    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.as_vector((ufl.sin(ufl.pi * x[1]), 0.0))
    p_exact = 0.0

    # The forcing term 'f' required to make the momentum equation balance this sine wave
    # f = -mu * Laplace(u) + grad(p)
    f = ufl.as_vector((ufl.pi**2 * ufl.sin(ufl.pi * x[1]), 0.0))

    # 4. Boundaries
    # The PROPER way to find all outer boundaries without manual math
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(msh.topology)
    boundary_dofs = fem.locate_dofs_topological((W.sub(0), V), msh.topology.dim - 1, boundary_facets)

    # Apply the exact sine wave condition to ONLY the boundary
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x_coords: np.vstack((np.sin(np.pi * x_coords[1]), np.zeros_like(x_coords[0]))))

    zero_pressure = fem.Function(Q)
    zero_pressure.x.array[:] = 0.0
    
    # Fix pressure at the origin to anchor the system (requires absolute tolerance)
    def origin_loc(x_coords):
        return np.logical_and(np.isclose(x_coords[0], 0.0, atol=1e-6), 
                              np.isclose(x_coords[1], 0.0, atol=1e-6))
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), origin_loc)

    bcs = [
        fem.dirichletbc(u_bc, boundary_dofs, W.sub(0)),
        fem.dirichletbc(zero_pressure, p_dofs, W.sub(1))
    ]

    # 5. Variational Form
    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)
    mu = fem.Constant(msh, PETSc.ScalarType(1.0))

    # Weak form WITH the manufactured forcing term L
    a = (mu * ufl.inner(ufl.grad(u), ufl.grad(v)) - p * ufl.div(v) + q * ufl.div(u)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    # 6. Solve
    problem = LinearProblem(
        a, L, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix=f"stokes_{h}_"
    )
    w_h = problem.solve()
    u_h = w_h.sub(0)

    # 7. Calculate exact L2 Error using UFL integral
    error_form = fem.form(ufl.inner(u_h - u_exact, u_h - u_exact) * ufl.dx)
    error_sq = fem.assemble_scalar(error_form)
    L2_error = np.sqrt(msh.comm.allreduce(error_sq, op=MPI.SUM))

    return L2_error

# ==========================================
# Run the Analysis Loop
# ==========================================
mesh_sizes = [0.2, 0.1, 0.05, 0.025]
errors = []

print("Running MMS Convergence Analysis...")
for h in mesh_sizes:
    err = solve_and_compute_error(h)
    errors.append(err)
    print(f"h = {h:.3f}, L2 Error = {err:.2e}")

# Calculate slope
log_h = np.log(mesh_sizes)
log_e = np.log(errors)
slope, intercept = np.polyfit(log_h, log_e, 1)

print(f"\nCalculated Convergence Slope: {slope:.2f}")

# Plotting
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.plot(log_h, log_e, 'o-', color='#b00000', linewidth=2, markersize=8)
plt.xlabel(r"$\log(h)$", fontsize=12)
plt.ylabel(r"$\log(E_u)$", fontsize=12)
plt.title(f"Convergence Analysis: Stokes MMS\nSlope: {slope:.2f}", fontsize=14)
plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.savefig("convergence_plot_mms.png", dpi=300)
print("Plot saved as convergence_plot_mms.png")