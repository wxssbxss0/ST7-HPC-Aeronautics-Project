SetFactory("OpenCASCADE");
Merge "base.stp"; 

v() = Volume "*";

// 1. RE-SCALE (DILATE)
// Reduz o avião em 61.7% (fator de 0.383) tendo a origem (0,0,0) como centro
Dilate { {0, 0, 0}, 0.383 } { Volume{v()}; }

// 2. TÚNEL DE VENTO AJUSTADO
// Como o avião agora tem ~15.7m, não precisamos mais de uma caixa gigante.
// Uma caixa de 60m x 40m x 40m é suficiente.
Box(1000) = {-30, -20, -20, 60, 40, 40};

// 3. SUBTRAÇÃO (Cria o domínio do fluido)
BooleanDifference{ Volume{1000}; Delete; }{ Volume{v()}; Delete; }

// 4. GRUPOS FÍSICOS
Physical Surface("Aircraft_Skin", 1) = {Surface "*"}; 
Physical Volume("Fluid_Domain", 2) = {Volume "*"};

// 5. MESH SIZE - MODO "ESTUDO DE CONVERGÊNCIA"
Mesh.MeshSizeExtendFromBoundary = 1; // Deixa ele sentir o avião
Mesh.MeshSizeFromCurvature = 1;      // Deixa ele gastar elementos nas curvas

// Reduzimos o tamanho das "peças" do Lego:
// No avião: elementos de 15cm (0.15)
// No ar: elementos de no máximo 80cm (0.8)
Mesh.MeshSizeMin = 0.15;
Mesh.MeshSizeMax = 0.8;

Mesh.MeshSizeFactor = 1.0;