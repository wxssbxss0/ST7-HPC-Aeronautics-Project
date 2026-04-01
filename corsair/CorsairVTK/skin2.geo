SetFactory("OpenCASCADE");

Geometry.OCCFixDegenerated = 1;
Geometry.OCCFixSmallEdges = 1;
Geometry.OCCFixSmallFaces = 1;
Geometry.OCCSewFaces = 1;
Geometry.OCCMakeSolids = 1;

Merge "updated_3.stp";

// 1) aircraft
v_air() = Volume "*";
Dilate {{0, 0, 0}, 0.383} { Volume{v_air()}; }

// 2) box
Box(1000) = {-60, -30, -30, 130, 90, 60};

// 3) fluid domain
fluid() = BooleanDifference{ Volume{1000}; Delete; }{ Volume{v_air()}; Delete; };

// 4) physicals
all_surfaces() = Surface "*";

xmin_wall() = Surface In BoundingBox{-60.1, -30.1, -30.1, -59.9,  60.1,  30.1};
xmax_wall() = Surface In BoundingBox{ 69.9, -30.1, -30.1,  70.1,  60.1,  30.1};
ymin_wall() = Surface In BoundingBox{-60.1, -30.1, -30.1,  70.1, -29.9,  30.1};
ymax_wall() = Surface In BoundingBox{-60.1,  59.9, -30.1,  70.1,  60.1,  30.1};
zmin_wall() = Surface In BoundingBox{-60.1, -30.1, -30.1,  70.1,  60.1, -29.9};
zmax_wall() = Surface In BoundingBox{-60.1, -30.1,  29.9,  70.1,  60.1,  30.1};

box_walls() = xmin_wall();
box_walls() += xmax_wall();
box_walls() += ymin_wall();
box_walls() += ymax_wall();
box_walls() += zmin_wall();
box_walls() += zmax_wall();

aircraft_skin() = all_surfaces();
aircraft_skin() -= box_walls();

Physical Volume("Fluid_Domain", 1) = {Volume "*"};
Physical Surface("Aircraft_Skin", 10) = {aircraft_skin()};
Physical Surface("Box_Walls", 20) = {box_walls()};

// 5) brutally coarse mesh controls
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromParametricPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;

Mesh.MeshSizeMin = 40;
Mesh.MeshSizeMax = 220;
Mesh.MeshSizeFactor = 8;

Mesh.ElementOrder = 1;
Mesh.Optimize = 0;
Mesh.OptimizeNetgen = 0;
Mesh.Smoothing = 0;

Mesh.Algorithm = 6;
Mesh.Algorithm3D = 1;

// 6) generate and save ONLY vtk
Mesh 3;
Save "fluid_only.vtk";
