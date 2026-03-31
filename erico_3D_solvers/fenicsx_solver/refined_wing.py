import gmsh
import numpy as np
import sys

gmsh.initialize()
gmsh.model.add("wing_domain")

print("Lendo STEP...")
gmsh.merge("swept_wing_washout.step")
gmsh.model.occ.synchronize()

# Pega os volumes importados diretamente do CAD
vols = gmsh.model.getEntities(3)

if len(vols) == 0:
    print("ERRO: Seu STEP não contêm volumes. Volte no CAD e garanta que exportou um Sólido e não apenas Superfícies.")
    sys.exit()

# Assumindo que a asa é o primeiro (e único) volume do arquivo
wing_vol = vols[0][1] 

w_xmin, w_ymin, w_zmin, w_xmax, w_ymax, w_zmax = gmsh.model.occ.getBoundingBox(3, wing_vol)

# Daqui para baixo, seu código original continua igualzinho...
margin = 5.0
box_tag = gmsh.model.occ.addBox(w_xmin - margin, w_ymin - margin, w_zmin - margin, 
                               (w_xmax - w_xmin) + 2*margin + 10, 
                               (w_ymax - w_ymin) + 2*margin, 
                               (w_zmax - w_zmin) + 2*margin)
gmsh.model.occ.synchronize()

print("Cortando a asa...")
try:
    out, _ = gmsh.model.occ.cut([(3, box_tag)], [(3, wing_vol)], removeObject=True, removeTool=True)
    gmsh.model.occ.synchronize()
    fluid_vol = out[0][1]
except Exception as e:
    print(f"Erro fatal no corte: {e}")
    sys.exit()

print("Limpando lixos do CAD...")
gmsh.model.occ.removeAllDuplicates()
gmsh.model.occ.synchronize()

# Busca todos os volumes que ainda existem no modelo
todos_volumes = gmsh.model.getEntities(3)

# Se houver algum volume sobrando que não seja o fluido, DELETA!
volumes_para_apagar = [v for v in todos_volumes if v[1] != fluid_vol]
if volumes_para_apagar:
    print(f"Deletando {len(volumes_para_apagar)} volume(s) intruso(s)...")
    gmsh.model.removeEntities(volumes_para_apagar, recursive=True)

gmsh.model.occ.synchronize()

print("Identificando superfícies...")
f_xmin, f_ymin, f_zmin, f_xmax, f_ymax, f_zmax = gmsh.model.occ.getBoundingBox(3, fluid_vol)

all_surfaces = gmsh.model.getEntities(2)

wing_surfaces = []
inlet_surfaces = []
outlet_surfaces = []
tunnel_surfaces = []

for _, tag in all_surfaces:
    b_xmin, b_ymin, b_zmin, b_xmax, b_ymax, b_zmax = gmsh.model.occ.getBoundingBox(2, tag)
    
    tol = 0.1
    if np.isclose(b_xmin, f_xmin, atol=tol) and np.isclose(b_xmax, f_xmin, atol=tol):
        inlet_surfaces.append(tag)
    elif np.isclose(b_xmin, f_xmax, atol=tol) and np.isclose(b_xmax, f_xmax, atol=tol):
        outlet_surfaces.append(tag)
    elif (np.isclose(b_ymin, f_ymin, atol=tol) and np.isclose(b_ymax, f_ymin, atol=tol)) or \
         (np.isclose(b_ymin, f_ymax, atol=tol) and np.isclose(b_ymax, f_ymax, atol=tol)) or \
         (np.isclose(b_zmin, f_zmin, atol=tol) and np.isclose(b_zmax, f_zmin, atol=tol)) or \
         (np.isclose(b_zmin, f_zmax, atol=tol) and np.isclose(b_zmax, f_zmax, atol=tol)):
        tunnel_surfaces.append(tag)
    else:
        wing_surfaces.append(tag)

# Atribui Grupos Físicos
gmsh.model.addPhysicalGroup(3, [fluid_vol], 1, name="Fluid")
gmsh.model.addPhysicalGroup(2, wing_surfaces, 2, name="Wing")
gmsh.model.addPhysicalGroup(2, inlet_surfaces, 3, name="Inlet")
gmsh.model.addPhysicalGroup(2, outlet_surfaces, 4, name="Outlet")
gmsh.model.addPhysicalGroup(2, tunnel_surfaces, 5, name="Tunnel")

# Malha leve, direta e reta para não matar a RAM
# --- CONFIGURAÇÕES INTELIGENTES DE MALHA ---
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2) 

# Longe da asa, os elementos podem ser gigantes para poupar RAM
gmsh.option.setNumber("Mesh.MeshSizeMax", 3.0)

# Tamanho mínimo permitido (pequeno o suficiente para capturar o bordo de ataque)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.02)

# O PULO DO GATO: Adaptar a malha à curvatura do CAD!
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
# Define quantos elementos formam um círculo imaginário (20 a 30 é um ótimo balanço de qualidade/RAM)
gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 25)
print("Gerando a malha 3D leve...")
gmsh.model.mesh.generate(3)
gmsh.write("refined_wing.msh")
gmsh.finalize()