================================================================================
3D Aerodynamic Wing Simulation Pipeline
================================================================================

This repository contains a complete, automated Python pipeline for generating a 
3D swept wing, meshing the surrounding fluid domain, and running a Computational 
Fluid Dynamics (CFD) simulation to calculate aerodynamic forces (Lift, Drag, 
and Side Force). 

The pipeline leverages powerful open-source libraries including CadQuery 
(CAD modeling), Gmsh (meshing), Meshio (format conversion), and 
FEniCSx / DOLFINx (Finite Element Method solver).

--------------------------------------------------------------------------------
PREREQUISITES & DEPENDENCIES
--------------------------------------------------------------------------------
To run this pipeline, you need a Python environment configured with the 
following libraries:
- numpy
- cadquery
- gmsh
- meshio
- fenics-dolfinx
- ufl
- mpi4py
- petsc4py

Note: Because FEniCSx and its dependencies (MPI, PETSc) can be complex to 
install natively, it is highly recommended to run this environment using Conda 
or a dedicated Docker container (like the official dolfinx/dolfinx image).

--------------------------------------------------------------------------------
FILE OVERVIEW
--------------------------------------------------------------------------------
Here is a breakdown of what each script does, in the order they should be 
executed:

1. generate_wing.py (CAD Generation)
   - Purpose: Uses CadQuery to loft a 3D swept wing with a washout angle. 
     It creates a realistic blunt trailing edge using a root profile 
     (NACA 2412) and a tip profile (NACA 0012).
   - Output: swept_wing_washout.step

2. refined_wing.py (Domain & Meshing)
   - Purpose: Uses Gmsh to import the STEP file, create a bounding box for 
     the fluid tunnel, subtract the wing from the box, and generate a 3D 
     volumetric mesh adapted to the wing's curvature.
   - Output: refined_wing.msh

3. convert_mesh.py (Mesh Conversion)
   - Purpose: Uses Meshio to split the .msh file into distinct volume 
     (tetrahedrons) and surface (triangles) components formatted for 
     FEniCSx compatibility.
   - Outputs: wing_domain.xdmf, wing_facets.xdmf

4. solver3D.py (CFD Solver)
   - Purpose: Uses FEniCSx to solve the steady Stokes flow equations. 
     Applies boundary conditions, solves using the MUMPS direct solver, 
     exports results for ParaView, and integrates pressures/shear to 
     calculate forces.
   - Outputs: results.xdmf, Terminal Output

--------------------------------------------------------------------------------
USAGE INSTRUCTIONS
--------------------------------------------------------------------------------
Execute the scripts in your terminal in the following sequential order:

Step 1: Generate the Geometry
Command: python generate_wing.py
What happens: Generates custom NACA profiles, lofts them together to create a 
solid 3D wing, verifies the volume, and exports the geometry.

Step 2: Mesh the Domain
Command: python refined_wing.py
What happens: Performs a boolean cut to hollow out the wing shape from the 
fluid box. Identifies the physical boundaries and generates a curvature-adapted 
3D tetrahedral mesh to save RAM while maintaining accuracy.

Step 3: Convert the Mesh
Command: python convert_mesh.py
What happens: Extracts the 3D volume data (Fluid) into one XDMF file and the 
2D surface data (boundary tags) into another for FEniCSx.

Step 4: Run the Simulation
Command: python solver3D.py
What happens: Loads the mesh, sets up an inlet velocity of 10.0 units, applies 
no-slip conditions, solves the physics matrices, and integrates the fluid 
stresses to print out the final Drag (X-axis), Lift (Y-axis), and 
Lateral force (Z-axis).

--------------------------------------------------------------------------------
VISUALIZING RESULTS
--------------------------------------------------------------------------------
Once solver3D.py finishes successfully, it will generate a results.xdmf file. 

To visualize the flow field:
1. Open ParaView.
2. Load results.xdmf.
3. Apply filters like Glyph (for velocity vectors) or StreamTracer 
   (for streamlines).
4. Inspect the Pressao (Pressure) and Velocidade (Velocity) fields.