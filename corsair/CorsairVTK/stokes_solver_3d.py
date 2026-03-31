#!/usr/bin/env python3
"""
stokes_solver_3d.py

Minimal 3D Stokes solver for a Gmsh .msh fluid-domain mesh (box minus aircraft),
without relying on physical groups. Boundary conditions are recovered geometrically:

- inlet  : x = xmin
- outlet : x = xmax
- walls  : y = ymin/ymax and z = zmin/zmax
- body   : all remaining boundary facets

Uses Taylor-Hood elements:
- velocity: P2 vector
- pressure: P1 scalar

Outputs:
- velocity_3d.xdmf
- pressure_3d.xdmf

Usage:
    python3 stokes3d_test.py corsair.msh
"""

import sys
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh as dmesh, io
from dolfinx.io.gmshio import read_from_msh
import ufl
from basix.ufl import element, mixed_element


def main():
    if len(sys.argv) < 2:
        if MPI.COMM_WORLD.rank == 0:
            print("Usage: python3 stokes_solver_3d.py <corsair.msh>")
        sys.exit(1)

    msh_file = sys.argv[1]

    # --------------------------------------------------
    # 1. Read mesh
    # --------------------------------------------------
    mesh, cell_tags, facet_tags = read_from_msh(
        "corsair.msh", MPI.COMM_WORLD, gdim=3
    )

    tdim = mesh.topology.dim
    fdim = tdim - 1
    gdim = mesh.geometry.dim

    assert gdim == 3, f"Expected gdim=3, got {gdim}"

    if mesh.comm.rank == 0:
        print(f"Loaded mesh: {msh_file}")
        print(f"Topological dimension = {tdim}")
        print(f"Geometric dimension   = {gdim}")

    # --------------------------------------------------
    # 2. Compute bounding box
    # --------------------------------------------------
    x = mesh.geometry.x
    xmin = np.min(x[:, 0])
    xmax = np.max(x[:, 0])
    ymin = np.min(x[:, 1])
    ymax = np.max(x[:, 1])
    zmin = np.min(x[:, 2])
    zmax = np.max(x[:, 2])

    Lx = xmax - xmin
    Ly = ymax - ymin
    Lz = zmax - zmin
    Lref = max(Lx, Ly, Lz)

    # Tolerance for identifying outer-box boundaries
    tol = 1e-6 * Lref

    if mesh.comm.rank == 0:
        print("Bounding box:")
        print(f"  x in [{xmin:.6g}, {xmax:.6g}]")
        print(f"  y in [{ymin:.6g}, {ymax:.6g}]")
        print(f"  z in [{zmin:.6g}, {zmax:.6g}]")
        print(f"Tolerance = {tol:.6g}")

    # --------------------------------------------------
    # 3. Locate boundary facets geometrically
    # --------------------------------------------------
    mesh.topology.create_connectivity(fdim, tdim)

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
        print("Facet counts (local on rank 0 only if serial):")
        print(f"  inlet facets   = {len(inlet_facets)}")
        print(f"  outlet facets  = {len(outlet_facets)}")
        print(f"  wall facets    = {len(wall_facets)}")
        print(f"  body facets    = {len(body_facets)}")
        print(f"  all boundary   = {len(all_boundary_facets)}")

    # --------------------------------------------------
    # 4. Taylor-Hood function space
    # --------------------------------------------------
    cell = mesh.basix_cell()
    P2 = element("Lagrange", cell, 2, shape=(gdim,))
    P1 = element("Lagrange", cell, 1)
    TH = mixed_element([P2, P1])
    W = fem.functionspace(mesh, TH)

    W0 = W.sub(0)  # velocity
    W1 = W.sub(1)  # pressure

    V, _ = W0.collapse()
    Q, _ = W1.collapse()

    # --------------------------------------------------
    # 5. Boundary conditions
    # --------------------------------------------------
    U0 = 1.0
    mu = fem.Constant(mesh, PETSc.ScalarType(1.0))
    f = fem.Constant(mesh, PETSc.ScalarType((0.0, 0.0, 0.0)))

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

    p_zero = fem.Function(Q)
    p_zero.x.array[:] = 0.0

    inlet_dofs = fem.locate_dofs_topological((W0, V), fdim, inlet_facets)
    wall_dofs = fem.locate_dofs_topological((W0, V), fdim, wall_facets)
    body_dofs = fem.locate_dofs_topological((W0, V), fdim, body_facets)
    outlet_dofs = fem.locate_dofs_topological((W1, Q), fdim, outlet_facets)

    bc_inlet = fem.dirichletbc(u_in, inlet_dofs, W0)

    # Simple farfield approximation for first test:
    # impose freestream velocity on outer box walls too
    bc_walls = fem.dirichletbc(u_in, wall_dofs, W0)

    # No-slip on aircraft
    bc_body = fem.dirichletbc(u_zero, body_dofs, W0)

    # Pressure reference at outlet
    bc_outlet = fem.dirichletbc(p_zero, outlet_dofs, W1)

    bcs = [bc_inlet, bc_walls, bc_body, bc_outlet]

    if mesh.comm.rank == 0:
        print("DOF counts:")
        print(f"  inlet velocity dofs  = {len(inlet_dofs)}")
        print(f"  wall velocity dofs   = {len(wall_dofs)}")
        print(f"  body velocity dofs   = {len(body_dofs)}")
        print(f"  outlet pressure dofs = {len(outlet_dofs)}")

    # --------------------------------------------------
    # 6. Variational problem
    # --------------------------------------------------
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    a = (
        mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )

    L = ufl.inner(f, v) * ufl.dx

    # --------------------------------------------------
    # 7. Solve
    # --------------------------------------------------
    if mesh.comm.rank == 0:
        print("Solving linear Stokes system...")

    problem = fem.petsc.LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )

    wh = problem.solve()

    uh, ph = wh.split()
    uh.name = "velocity"
    ph.name = "pressure"

    if mesh.comm.rank == 0:
        print("Solve complete.")

    # --------------------------------------------------
    # 8. Export solution
    # --------------------------------------------------
    with io.XDMFFile(mesh.comm, "velocity_3d.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(uh)

    with io.XDMFFile(mesh.comm, "pressure_3d.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(ph)

    if mesh.comm.rank == 0:
        print("Wrote:")
        print("  velocity_3d.xdmf")
        print("  pressure_3d.xdmf")
        print("\nOpen in ParaView and inspect pressure near the nose / leading edges.")


if __name__ == "__main__":
    main()
