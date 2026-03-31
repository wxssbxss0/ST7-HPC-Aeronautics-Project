import numpy as np
import cadquery as cq

def get_naca_blunt(naca_str, chord=1.0, num_pts=40):
    """Generates an airtight profile with a realistic Blunt Trailing Edge."""
    m = int(naca_str[0]) / 100.0
    p = int(naca_str[1]) / 10.0
    t = int(naca_str[2:]) / 100.0

    beta = np.linspace(0, np.pi, num_pts)
    x = chord * (0.5 * (1 - np.cos(beta)))
    
    yt = 5 * t * chord * (0.2969 * np.sqrt(x/chord) - 0.1260 * (x/chord) - 
                          0.3516 * (x/chord)**2 + 0.2843 * (x/chord)**3 - 0.1015 * (x/chord)**4)
    
    blunt_thickness = 0.002 * chord
    yt[-1] = blunt_thickness 
    
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    
    if p > 0:
        front = x <= p * chord
        back = x > p * chord
        yc[front] = (m / p**2) * (2 * p * (x[front]/chord) - (x[front]/chord)**2)
        yc[back] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * (x[back]/chord) - (x[back]/chord)**2)
        dyc_dx[front] = (2 * m / p**2) * (p - x[front]/chord)
        dyc_dx[back] = (2 * m / (1 - p)**2) * (p - x[back]/chord) 

    theta = np.arctan(dyc_dx)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    xu[0] = 0.0; yu[0] = 0.0
    xl[0] = 0.0; yl[0] = 0.0
    xu[-1] = chord; yu[-1] = blunt_thickness
    xl[-1] = chord; yl[-1] = -blunt_thickness

    # --- A MÁGICA ACONTECE AQUI ---
    # Criamos uma ÚNICA lista contínua dando a volta no bordo de ataque
    pts = [(float(xu[i]), float(yu[i])) for i in range(len(xu)-2, 0, -1)]
    pts.append((0.0, 0.0))
    pts.extend([(float(xl[i]), float(yl[i])) for i in range(1, len(xl))])
    
    return pts

# ==========================================
# 3D Wing Parameters
# ==========================================
span = 5.0
root_chord = 1.0
tip_chord = 0.5
sweep_back = 1.5      
washout_angle = -5.0  

print("Generating blunt-edge profiles...")
root_pts = get_naca_blunt("2412", chord=root_chord)
tip_pts = get_naca_blunt("0012", chord=tip_chord)

# ==========================================
# CADQuery Fluent API Lofting
# ==========================================


print("Lofting watertight 3D CAD geometry...")
wing = (
    cq.Workplane("XY")
    
    # ROOT PROFILE
    .moveTo(root_chord, 0.002 * root_chord)  
    .spline(root_pts)   # Uma curva só!       
    .close()                  
    
    # SHIFT TO TIP
    .workplane(offset=span)
    .center(sweep_back, 0)
    .transformed(rotate=cq.Vector(0, 0, washout_angle))
    
    # TIP PROFILE
    .moveTo(tip_chord, 0.002 * tip_chord) 
    .spline(tip_pts)    # Uma curva só! 
    .close()
    
    # LOFT 
    .loft(ruled=True)
)

# O TESTE DE FOGO
vol = wing.val().Volume()
print(f"Volume físico do sólido calculado pelo CAD: {vol:.4f}")

if vol == 0.0:
    print("ERRO: O objeto ainda é uma casca (Shell) oca!")
else:
    export_filename = "swept_wing_washout.step"
    cq.exporters.export(wing, export_filename)
    print(f"Sucesso! Sólido 3D exportado perfeitamente para: {export_filename}")