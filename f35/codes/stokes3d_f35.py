import os
import tempfile
import numpy as np
import meshio
import h5py
import ufl
import dolfinx
from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from basix.ufl import element, mixed_element
import gmsh


def _read_msh_without_gmshio(msh_file: str):
    try:
        import meshio
    except ImportError as exc:
        raise RuntimeError(
            "meshio is required. Install with: python -m pip install meshio"
        ) from exc

    msh = meshio.read(msh_file)
    available_cell_types = list(msh.cells_dict.keys())

    if "tetra" in msh.cells_dict:
        cells = msh.cells_dict["tetra"].copy()
    elif "tetra10" in msh.cells_dict:
        cells = msh.cells_dict["tetra10"][:, :4].copy()
    else:
        raise RuntimeError(
            "No tetrahedral 3D cells found. "
            f"Available cell types: {available_cell_types}"
        )

    points = msh.points[:, :3].copy()
    mesh3d = meshio.Mesh(points=points, cells=[("tetra", cells)])

    tmp_xdmf = tempfile.NamedTemporaryFile(delete=False, suffix=".xdmf")
    tmp_xdmf.close()
    meshio.write(tmp_xdmf.name, mesh3d)
    h5_file = tmp_xdmf.name.replace(".xdmf", ".h5")

    with XDMFFile(MPI.COMM_WORLD, tmp_xdmf.name, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")

    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

    os.unlink(tmp_xdmf.name)
    if os.path.exists(h5_file):
        os.unlink(h5_file)

    return domain


def _principal_axis_from_surface_mesh(entities):
    gmsh.model.mesh.clear()
    gmsh.model.mesh.generate(2)
    _, coords, _ = gmsh.model.mesh.getNodes()
    gmsh.model.mesh.clear()

    points = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] < 8:
        return None, None

    center = points.mean(axis=0)
    centered = points - center
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    axis = axis / np.linalg.norm(axis)
    if axis[0] < 0:
        axis = -axis
    return center, axis


def _align_entities_with_x_axis(entities):
    center, axis = _principal_axis_from_surface_mesh(entities)
    if center is None:
        return

    target = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    dot = float(np.clip(np.dot(axis, target), -1.0, 1.0))
    if abs(dot - 1.0) < 1e-10:
        return

    if abs(dot + 1.0) < 1e-10:
        rot_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        angle = np.pi
    else:
        rot_axis = np.cross(axis, target)
        rot_axis = rot_axis / np.linalg.norm(rot_axis)
        angle = float(np.arccos(dot))

    gmsh.model.occ.rotate(
        entities,
        float(center[0]), float(center[1]), float(center[2]),
        float(rot_axis[0]), float(rot_axis[1]), float(rot_axis[2]),
        angle,
    )
    gmsh.model.occ.synchronize()


def _write_boundary_vtu(domain, box_facets, vehicle_facets, file_name="f35_boundaries.vtu"):
    import meshio

    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, 0)
    facet_to_vertex = domain.topology.connectivity(fdim, 0)

    facets = np.hstack((box_facets, vehicle_facets)).astype(np.int32)
    markers = np.hstack((
        np.full(len(box_facets), 1, dtype=np.int32),
        np.full(len(vehicle_facets), 2, dtype=np.int32),
    ))

    triangles = []
    tri_markers = []
    for facet, marker in zip(facets, markers):
        verts = facet_to_vertex.links(int(facet))
        if len(verts) == 3:
            triangles.append(verts)
            tri_markers.append(marker)

    if len(triangles) == 0:
        return

    surf_mesh = meshio.Mesh(
        points=domain.geometry.x[:, :3],
        cells=[("triangle", np.asarray(triangles, dtype=np.int64))],
        cell_data={"boundary_id": [np.asarray(tri_markers, dtype=np.int32)]},
    )
    meshio.write(file_name, surf_mesh)


def _write_stream_seed_csv(bounds, file_name="f35_stream_seeds.csv", ny=12, nz=12):
    x_seed = bounds["x0"] + 0.02 * (bounds["x1"] - bounds["x0"])
    y_vals = np.linspace(bounds["y0"] + 0.1 * (bounds["y1"] - bounds["y0"]),
                         bounds["y1"] - 0.1 * (bounds["y1"] - bounds["y0"]), ny)
    z_vals = np.linspace(bounds["z0"] + 0.1 * (bounds["z1"] - bounds["z0"]),
                         bounds["z1"] - 0.1 * (bounds["z1"] - bounds["z0"]), nz)

    with open(file_name, "w", encoding="utf-8") as csv_file:
        csv_file.write("x,y,z\n")
        for y in y_vals:
            for z in z_vals:
                csv_file.write(f"{x_seed},{y},{z}\n")


def _build_f35_fluid_mesh(step_file: str):
    if not os.path.exists(step_file):
        raise FileNotFoundError(f"STEP file not found: {step_file}")

    gmsh.initialize()
    gmsh.model.add("f35_flow")
    gmsh.option.setNumber("General.Terminal", 1)

    imported = gmsh.model.occ.importShapes(step_file)
    gmsh.model.occ.synchronize()

    _align_entities_with_x_axis(imported)

    vehicle_volumes = [dt for dt in imported if dt[0] == 3]

    if len(vehicle_volumes) == 0:
        vehicle_surfaces = [dt for dt in imported if dt[0] == 2]
        if len(vehicle_surfaces) == 0:
            raise RuntimeError("STEP import contains neither volumes nor surfaces.")

        if MPI.COMM_WORLD.rank == 0:
            print("STEP imported as surfaces only; attempting shell -> volume conversion...")

        surf_loop = gmsh.model.occ.addSurfaceLoop([s[1] for s in vehicle_surfaces], sewing=True)
        vol_tag = gmsh.model.occ.addVolume([surf_loop])
        gmsh.model.occ.synchronize()
        vehicle_volumes = [(3, vol_tag)]

    xmin = ymin = zmin = 1.0e30
    xmax = ymax = zmax = -1.0e30

    for dim, tag in vehicle_volumes:
        bx0, by0, bz0, bx1, by1, bz1 = gmsh.model.occ.getBoundingBox(dim, tag)
        xmin = min(xmin, bx0)
        ymin = min(ymin, by0)
        zmin = min(zmin, bz0)
        xmax = max(xmax, bx1)
        ymax = max(ymax, by1)
        zmax = max(zmax, bz1)

    lx = xmax - xmin
    ly = ymax - ymin
    lz = zmax - zmin
    lref = max(lx, ly, lz)

    x0 = xmin - 0.6 * lref
    x1 = xmax + 1.2 * lref
    y0 = ymin - 0.8 * lref
    y1 = ymax + 0.8 * lref
    z0 = zmin - 0.8 * lref
    z1 = zmax + 0.8 * lref

    box = gmsh.model.occ.addBox(x0, y0, z0, x1 - x0, y1 - y0, z1 - z0)
    gmsh.model.occ.synchronize()

    cut_result, _ = gmsh.model.occ.cut([(3, box)], vehicle_volumes, removeObject=True, removeTool=True) #can change back to false l8r
    gmsh.model.occ.synchronize()
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()

    fluid_volumes = [dt for dt in cut_result if dt[0] == 3]
    if len(fluid_volumes) == 0:
        gmsh.finalize()
        raise RuntimeError("Boolean cut failed: no fluid volume created.")

    fluid_tag = fluid_volumes[0][1]
    gmsh.model.addPhysicalGroup(3, [fluid_tag], 5)
    gmsh.model.setPhysicalName(3, 5, "fluid")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.15 * lref)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.60 * lref)
    
    gmsh.option.setNumber("Mesh.Optimize", 0) # reduces compute time by a factor of 3x or 4x easily
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)

    def _count_3d_elements():
        _, elem_tags_by_type, _ = gmsh.model.mesh.getElements(3)
        return int(sum(len(arr) for arr in elem_tags_by_type))

    n3 = 0
    for alg in (1, 4, 10):
        gmsh.model.mesh.clear()
        gmsh.option.setNumber("Mesh.Algorithm3D", alg)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(1)
        n3 = _count_3d_elements()
        if MPI.COMM_WORLD.rank == 0:
            print(f"3D meshing attempt (Algorithm3D={alg}) -> elements: {n3}")
        if n3 > 0:
            break

    if n3 == 0:
        gmsh.finalize()
        raise RuntimeError("Gmsh could not generate tetrahedra.")

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".msh")
    tmp.close()
    gmsh.write(tmp.name)
    gmsh.finalize()

    domain = _read_msh_without_gmshio(tmp.name)
    os.unlink(tmp.name)

    return domain, {
        "x0": x0, "x1": x1,
        "y0": y0, "y1": y1,
        "z0": z0, "z1": z1,
        "tol": max(1e-5, 5e-3 * lref),
    }


def main():
    step_path = "f35.step"
    domain, bounds = _build_f35_fluid_mesh(step_path)

    v_el = element("Lagrange", domain.topology.cell_name(), 2, shape=(domain.geometry.dim,))
    p_el = element("Lagrange", domain.topology.cell_name(), 1)
    W = fem.functionspace(domain, mixed_element([v_el, p_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    fdim = domain.topology.dim - 1
    x0, x1 = bounds["x0"], bounds["x1"]
    y0, y1 = bounds["y0"], bounds["y1"]
    z0, z1 = bounds["z0"], bounds["z1"]
    tol = bounds["tol"]

    inlet_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], x0, atol=tol))
    outlet_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], x1, atol=tol))
    wall_y0_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], y0, atol=tol))
    wall_y1_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], y1, atol=tol))
    wall_z0_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[2], z0, atol=tol))
    wall_z1_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[2], z1, atol=tol))
    wall_facets = np.unique(np.hstack((wall_y0_facets, wall_y1_facets, wall_z0_facets, wall_z1_facets))).astype(np.int32)

    all_boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    outer_facets = np.unique(np.hstack((inlet_facets, outlet_facets, wall_facets))).astype(np.int32)
    vehicle_facets = np.setdiff1d(all_boundary_facets.astype(np.int32), outer_facets, assume_unique=False)

    if MPI.COMM_WORLD.rank == 0:
        print(f"Detected boundary facets: inlet={len(inlet_facets)}, outlet={len(outlet_facets)}, walls={len(wall_facets)}, vehicle={len(vehicle_facets)}")

    if len(inlet_facets) == 0 or len(outlet_facets) == 0 or len(vehicle_facets) == 0:
        raise RuntimeError(
            f"Boundary detection failed: inlet={len(inlet_facets)}, outlet={len(outlet_facets)}, vehicle={len(vehicle_facets)}"
        )

    _write_boundary_vtu(domain, outer_facets, vehicle_facets, file_name="f35_boundaries.vtu")
    _write_stream_seed_csv(bounds, file_name="f35_stream_seeds.csv")

    inlet_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, inlet_facets)
    wall_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets)
    vehicle_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, vehicle_facets)
    outlet_dofs = fem.locate_dofs_topological((W.sub(1), Q), fdim, outlet_facets)

    if isinstance(outlet_dofs, (tuple, list)):
        outlet_dofs = outlet_dofs[0]

    u_inlet = fem.Function(V)
    u_inlet.interpolate(lambda x: np.vstack((
        np.full(x.shape[1], 1.0, dtype=dolfinx.default_scalar_type),
        np.zeros(x.shape[1], dtype=dolfinx.default_scalar_type),
        np.zeros(x.shape[1], dtype=dolfinx.default_scalar_type),
    )))

    u_zero = fem.Function(V)
    u_zero.x.array[:] = 0.0

    bc_inlet = fem.dirichletbc(u_inlet, inlet_dofs, W.sub(0))
    bc_walls = fem.dirichletbc(u_zero, wall_dofs, W.sub(0))
    bc_vehicle = fem.dirichletbc(u_zero, vehicle_dofs, W.sub(0))
    bc_outlet = fem.dirichletbc(dolfinx.default_scalar_type(0), outlet_dofs, W.sub(1))
    bcs = [bc_inlet, bc_walls, bc_vehicle, bc_outlet]

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    mu = fem.Constant(domain, dolfinx.default_scalar_type(1.0))
    f = fem.Constant(domain, dolfinx.default_scalar_type((0.0, 0.0, 0.0)))

    a = (mu * ufl.inner(ufl.grad(u), ufl.grad(v)) - ufl.div(v) * p + q * ufl.div(u)) * ufl.dx
    L = ufl.dot(f, v) * ufl.dx

    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )

    wh = problem.solve()
    u_h = wh.sub(0).collapse()
    p_h = wh.sub(1).collapse()

    V_viz = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    Q_viz = fem.functionspace(domain, ("Lagrange", 1))
    u_viz = fem.Function(V_viz)
    p_viz = fem.Function(Q_viz)
    u_viz.interpolate(u_h)
    p_viz.interpolate(p_h)

    with XDMFFile(domain.comm, "f35_velocity.xdmf", "w") as xdmf_u:
        xdmf_u.write_mesh(domain)
        xdmf_u.write_function(u_viz)

    with XDMFFile(domain.comm, "f35_pressure.xdmf", "w") as xdmf_p:
        xdmf_p.write_mesh(domain)
        xdmf_p.write_function(p_viz)

    if MPI.COMM_WORLD.rank == 0:
        print("F35 Stokes simulation completed.")
        print("Wrote: f35_velocity.xdmf, f35_pressure.xdmf, f35_boundaries.vtu, f35_stream_seeds.csv")


if __name__ == "__main__":
    main()
