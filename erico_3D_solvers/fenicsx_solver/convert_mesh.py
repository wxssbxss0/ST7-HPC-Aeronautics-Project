import meshio
import numpy as np

def convert():
    # Lê a malha MSH 2.2 que o seu refined_wing.py gerou
    msh = meshio.read("refined_wing.msh")

    # --- PROCESSANDO O VOLUME (Tetras) ---
    # Pegamos todas as células tetraédricas
    cells = msh.get_cells_type("tetra")
    # Pegamos os dados físicos (Tag 1 = Fluid)
    cell_data = msh.get_cell_data("gmsh:physical", "tetra")
    
    out_mesh = meshio.Mesh(points=msh.points, 
                           cells={"tetra": cells},
                           cell_data={"Grid": [cell_data]})
    meshio.write("wing_domain.xdmf", out_mesh)

    # --- PROCESSANDO AS SUPERFÍCIES (Triângulos) ---
    # Pegamos todas as células triangulares (Asa, Inlet, Tunnel)
    facet_cells = msh.get_cells_type("triangle")
    facet_data = msh.get_cell_data("gmsh:physical", "triangle")
    
    facet_mesh = meshio.Mesh(points=msh.points,
                             cells={"triangle": facet_cells},
                             cell_data={"Grid": [facet_data]})
    meshio.write("wing_facets.xdmf", facet_mesh)
    
    print("--- DEBUG DE CONVERSÃO ---")
    print(f"Total de pontos: {len(msh.points)}")
    print(f"Células de volume (Tetra): {len(cells)}")
    print(f"Células de superfície (Triangle): {len(facet_cells)}")
    print(f"Tags encontradas nas superfícies: {np.unique(facet_data)}")

convert()