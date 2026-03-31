# ST7 Project – 3D Stokes Solver Work Log (F35 Pipeline)

## Overview

During this work session I focused on building a **self-contained 3D Stokes pipeline** for the F35 geometry, with the goal of demonstrating a realistic **CAD → mesh → solve → postprocess** workflow for the midterm presentation.

The final approach starts from an **F35 STEP file**, generates the surrounding fluid domain **inside the solver script itself**, creates a tetrahedral mesh, solves the **3D steady Stokes equations** with **Taylor–Hood elements**, and exports postprocessing files for ParaView.

A key part of this effort was adapting **Erico’s script/logic** for the F35 STEP generation and then simplifying the geometry-handling side so the downstream meshing and boolean operations became more robust.

---

## Main contribution

My work tonight can be summarized as:

1. **Adapted Erico’s STEP-generation logic for the F35**
   - Reused/adapted the teammate workflow for constructing the aircraft STEP file.
   - Used that as the geometric starting point for the 3D CFD pipeline.

2. **Created / tested a simplified STEP-based route**
   - The original CAD-to-mesh route was fragile.
   - I explored simplified geometry-generation strategies to make the body more robust for boolean subtraction and volume meshing.
   - I also tested a fallback wing-generation path and used CAD validity checks (`Solid`, `validity`, `volume`) to understand why some STEP geometries failed.

3. **Built a self-contained 3D solver script**
   - Instead of relying on a separately preprocessed mesh with physical boundary groups, the solver script now:
     - imports the STEP geometry,
     - constructs the outer fluid box,
     - performs the boolean subtraction `fluid = box − vehicle`,
     - generates the 3D tetrahedral mesh,
     - detects boundaries geometrically,
     - solves the 3D Stokes system,
     - writes postprocessing files.

4. **Reduced runtime by changing meshing strategy**
   - Earlier runs were extremely long due to overly fine meshing, larger outer boxes, and mesh optimization.
   - I introduced a faster/coarser route for debugging and first-pass validation.

---

## Final workflow used

The final workflow is:

**F35 STEP**  
→ import geometry  
→ build fluid box in-script  
→ boolean cut: `fluid = outer box − aircraft`  
→ generate 2D/3D mesh in Gmsh  
→ read mesh into DOLFINx using a `meshio`/XDMF workaround  
→ detect inlet/outlet/walls/vehicle geometrically  
→ solve 3D Stokes with Taylor–Hood elements  
→ export pressure / velocity / boundaries / stream seed files

This made the workflow much more self-contained than the original approach.

---

## Difference from the earlier approach

### Earlier approach
The original route was more fragmented:

- STEP / manually boxed geometry
- external meshing
- `.msh` conversion to `.xdmf`
- separate solver expecting physical tags / facet labels

This caused repeated problems because:
- mesh metadata was incomplete,
- physical groups were often missing,
- some scripts assumed `facet_tags` existed when they did not,
- some imported domains behaved visually like “just a box”.

### New approach
The improved route avoids depending on pre-labeled mesh entities:

- the **solver script itself generates the fluid mesh**,
- boundaries are recovered **geometrically**,
- the script is more reproducible end-to-end.

This is a major improvement for a CAD-to-CFD workflow.

---

## Numerical model

The solver uses the **3D steady Stokes equations** with a mixed finite-element formulation:

- velocity: **P2**
- pressure: **P1**

This is the standard **Taylor–Hood** pair, chosen because it is stable for incompressible mixed problems.

Boundary conditions are detected geometrically from the final fluid box:

- **Inlet**: one face of the box (uniform flow direction)
- **Outlet**: opposite face
- **Outer box walls**: the remaining box faces
- **Vehicle**: all remaining internal boundary facets

This was necessary because I did not rely on preexisting physical groups in the imported mesh.

---

## Main files

### Geometry / setup
- `f35.step`  
  Main STEP geometry used for the F35 run.

- `f35_generate_wing.py`  
  Geometry-generation / adaptation work related to the F35 / wing branch.

### Solver
- `stokes3d_f35.py`  
  Main self-contained 3D Stokes solver script.

### Output directory
Inside `output/`:

- `f35_velocity.xdmf`
- `f35_pressure.xdmf`
- `f35_velocity.h5`
- `f35_pressure.h5`
- `f35_boundaries.vtu`
- `f35_stream_seeds.csv`
---

## Meaning of the output files

### 1. `f35_velocity.xdmf`
This contains the **3D velocity field** solved on the fluid domain.

It should be used in ParaView to visualize:
- the flow field,
- velocity magnitude,
- slices / clips through the domain,
- streamlines (combined with the stream seed file).

### 2. `f35_pressure.xdmf`
This contains the **3D pressure field** solved on the fluid domain.

Physically, this is the key quantity for checking the expected 3D Stokes behavior:
- higher pressure near leading/front stagnation regions,
- pressure drop through the domain,
- surface pressure behavior once the internal vehicle boundary is isolated.

### 3. `f35_velocity.h5`
HDF5 data file associated with `f35_velocity.xdmf`.

This stores the actual numerical data for the velocity solution, typically including:

- mesh-related field storage,

- nodal / degree-of-freedom values for the velocity field,

- topology/geometry information needed by the XDMF wrapper.

Conceptually, this is where the actual heavy data lives, while the .xdmf acts more like an XML descriptor/index that tells ParaView how to interpret it.

### 4. `f35_pressure.h5`

HDF5 data file associated with `f35_pressure.xdmf`.

This contains the actual numerical pressure solution values and associated mesh-based field storage.

### 5. `f35_boundaries.vtu`
This is a **boundary-only surface file** used for postprocessing.

Its purpose is to distinguish:
- the **outer box boundary**
- the **internal vehicle boundary**

This is especially useful because the solution is computed on the **fluid volume**, and in ParaView the outer box often hides the internal aircraft cavity.  
So this file is meant to help isolate the internal surface for cleaner visualization and later force analysis.

### 6. `f35_stream_seeds.csv`
This file stores **seed points** for streamline generation.

It is intended to help ParaView generate stream tracers more quickly by providing a set of input points near the inlet region of the flow.

This is useful for visualizing the overall flow pattern without manually selecting seed points every time.

---

## Difficulties encountered

A major part of tonight’s work was debugging the **CAD / meshing / environment pipeline**, not just the PDE itself.

### 1. STEP validity / CAD robustness
One of the biggest issues was that not every STEP-like solid was actually usable for CFD.

Some generated geometries appeared as:
- `Shape type: Solid`
- positive volume

but still failed CAD validity checks.

That matters because a body can look like a valid solid while still being **topologically invalid**, which breaks:
- boolean subtraction,
- cavity preservation,
- volume meshing.

This is why simplifying geometry generation and validating solids was important.

### 2. Mesh metadata / physical groups
The earlier mesh route often lacked reliable physical groups / facet tags.

As a result, scripts that assumed labeled boundaries (`facet_tags.find(...)`) failed.

I therefore switched to **geometric boundary detection**, which was a major conceptual change in the workflow.

### 3. Runtime / parameter selection
Another major difficulty was **very long meshing times**.

The longest runtimes were caused by:
- a larger outer box,
- finer characteristic lengths,
- Gmsh optimization / Netgen optimization,
- repeated 3D refinement and cleanup of ill-shaped tetrahedra.

To reduce runtime, I:
- made the box more compact,
- used coarser mesh lengths,
- disabled optimization,
- limited the number of meshing algorithm attempts.

### 4. Environment / dependency problems
There were also repeated issues with:
- missing `gmsh` Python bindings,
- missing `gmshio`,
- missing `h5py`,
- DOLFINx API path mismatches across environments.

These cost a lot of time because some failures occurred only **after** long meshing stages had already completed.

### 5. Postprocessing / visualization problems
Even after the solver completed, ParaView postprocessing turned out to be difficult.

The main issue was that the solution is stored on the **fluid volume**, so ParaView initially shows the outer fluid box. This makes it look like “only a box” unless the internal vehicle boundary is isolated correctly.

As a result, although the solver completed, I had trouble quickly validating the pressure distribution visually inside ParaView before the end of the session.

---

## What worked

The good news is that the final F35 solver run **did complete** and exported all expected files.

Most importantly, the script detected a **nonzero internal vehicle boundary**, which strongly suggests that the final fluid-domain construction was meaningful and that the solve was not merely on a plain empty box.

This is encouraging because it indicates:
- the boolean cut likely succeeded,
- the fluid domain likely retained the aircraft cavity,
- the Stokes solve ran on the intended geometry.

Although the ParaView postprocessing remained frustrating, the solver-side output is still strong evidence that the 3D pipeline is now functioning.

---

## Why the current result is still promising

Even though the visualization stage was incomplete, the run produced:

- a solved velocity field,
- a solved pressure field,
- a boundary-only file for postprocessing,
- stream seed points,
- a completed 3D Stokes log with nonzero vehicle boundary detection.

So the main unresolved issue is **validation / postprocessing**, not the existence of a solved CFD result.

That is important for the midterm: the work did not fail at the solver stage; the bottleneck became how to visualize and communicate the result cleanly.

---

## Suggested next steps

1. Validate the ParaView workflow with teammates
   - isolate the internal vehicle boundary from `f35_boundaries.vtu`
   - map pressure onto that surface
   - confirm whether the expected leading-edge pressure behavior is visible

2. Refine the postprocessing pipeline
   - automate extraction of the vehicle surface
   - automate streamline generation
   - produce cleaner figures for the report / presentation

3. Improve mesh quality gradually
   - once visualization is validated, revisit mesh resolution and optimization
   - compare coarse vs finer 3D runs

4. Standardize the pipeline for the group
   - keep the self-contained solver structure
   - document the geometry assumptions
   - make the run settings easier to tune

---

## Summary

Tonight’s session produced a working **self-contained 3D Stokes solver pipeline** for the F35 geometry.

The major contribution was moving from a fragile, tag-dependent mesh workflow to a more robust:

**STEP → auto fluid box → boolean cut → 3D mesh → geometric BC detection → Stokes solve → output export**

Although ParaView postprocessing remained difficult, the solver completed successfully and exported the key files needed for later validation and presentation support.
