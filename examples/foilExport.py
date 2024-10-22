#%%
# Import modules
import numpy as np
import AXIS_wing_definitions as AX_wings
from foilpy.foildef import FoilAssembly
# from foilpy.LL_functions import steady_LL_solve
from foilpy.utils import knts2ms, ms2knts, reinterp_arc_length
from foilpy.splines.curve import BSplineCurve, curve_approx, constrained_approx, parameterise_curve
import matplotlib.pyplot as plt
import foilpy.export as ex
from scipy.interpolate import PchipInterpolator, pchip_interpolate
import json
# %matplotlib widget

# Seawater properties: https://www.engineeringtoolbox.com/sea-water-properties-d_840.html
# Assumes 15 deg temperature seawater
speed_knts = 15
speed_ms = knts2ms(speed_knts)
chord = 0.2 # meters
rho = 1026
re = speed_ms * chord * rho / 0.00126
print("Reynolds number = ", str(re), "\n")

# Define simple motion vector, moving forward in y
u_motion = np.array([0, speed_ms, 0]).reshape(1, 3)
print("Input velocity vector = ", str(u_motion), " m/s (= ", str(ms2knts(u_motion)), " knots)\n ")


# Define front wing
front_wing = AX_wings.bsc_810(re, nsegs=40, plot_flag=False)
# front_wing.plot2D()

(x, ref_axis, 
    chord, t2c, 
    washout, method, qc) = ex.spanwise_geom(front_wing, 100, 1, 
                                    span_spacing='linspace', 
                                    add_socket=False, 
                                    half=True)
ncs_pts1 = 301
(afoil_coords,
    afoil_coords_interpolator) = ex.prep_afoils(front_wing.afoil_table, 
                                                ncs_pts1)

# Generate foil file
p = 3
u = parameterise_curve(qc[:,[0,2]], method='centripetal')
# Get QC rondure, sweep, chord, twist, sectionID distributions
rond = curve_approx(qc[:,[0,2]], 7, p, u_bar=u, U=None, plot_flag=True,
                        knot_spacing='adaptive')
ch = curve_approx(chord.reshape(-1,1), 6, p, u_bar=u, plot_flag=True,
                        knot_spacing='adaptive')
tw = curve_approx(washout.reshape(-1,1), 6, p, u_bar=u, U=ch.U, plot_flag=True,
                        knot_spacing='adaptive')
swp = curve_approx(qc[:,1:2], 6, p, u_bar=u, U=ch.U, plot_flag=True,
                        knot_spacing='adaptive')

# Convert t2c to secID
afoil = np.array(front_wing.afoil)[:,0]
for i, af in enumerate(front_wing.afoil_table):
    afoil[afoil == af] = i
secIDraw = np.hstack((np.array(front_wing.afoil)[:,1:], afoil.reshape(-1,1)))
secInterp = PchipInterpolator(secIDraw[:,0], secIDraw[:,1])
secID = secInterp(x / x[-1]).reshape(-1,1)

Wq = np.ones((secID.shape[0],1))
Wq[0] = -1
Wq[-1] = -1
D = np.array([0, 0, 0, 0]). reshape(-1,1)
I = [0, 1, secID.shape[0]-2, secID.shape[0]-1]
secIDspl = constrained_approx(secID, Wq, 15, p, 
            D=D, s=len(I)-1, I=I, Wd=[-1, -1, -1, -1],
            u_bar=u, plot_flag=True, 
            knot_spacing='adaptive')

# Get section shapes and polar data
sections = []
for i, af in enumerate(front_wing.afoil_table):
    sec = dict(SectionName = af, Alpha0Cl='From0Alpha', iDefaultShape=0)
    # shape = dict(MorphValue = 0.0, Name = af, xyPoints = afoil_coords[:,1:,i].tolist(), polar=dict())

    coords = front_wing.afoil_table[af]['coords']
    coords = np.delete(coords, 1000, axis=0)

    shape = dict(MorphValue = 0.0, Name = af, xyPoints = reinterp_arc_length(coords, 300).tolist(), polar=dict())
    # polar = dict()
    sec['Shapes'] = [shape]
    sections.append(sec)


# Populate data in foil file format
element = dict(
    Name = 'FrontWingStbd',
    IsSectionFlipped = False,
    DuplicatedAndReflected = True,
    DuplicatedAndReflectedName = "FrontWingPort",
    RotationAboutRondure = 0.0,
)
tStations = np.unique(np.concatenate((ch.U, tw.U, swp.U, secIDspl.U)))
rond.contrl_pts[:,0] = -rond.contrl_pts[:,0]
element['Rondure'] = dict(
    Uniform = False,
    tKU = rond.knotsU.tolist(),
    B = rond.contrl_pts.tolist(),
    tStations = tStations.tolist()
)
element['Chord'] = dict(B = ch.contrl_pts.reshape(-1).tolist(), stationFlags = np.isin(tStations, ch.knotsU).tolist())
element['Twist'] = dict(B = tw.contrl_pts.reshape(-1).tolist(), stationFlags = np.isin(tStations, tw.knotsU).tolist())
element['Sweep'] = dict(B = swp.contrl_pts.reshape(-1).tolist(), stationFlags = np.isin(tStations, swp.knotsU).tolist())
secIDspl.contrl_pts[secIDspl.contrl_pts < 1e-8] = 0
secIDspl.contrl_pts[secIDspl.contrl_pts > 1] = 1
element['SectionID'] = dict(B = secIDspl.contrl_pts.reshape(-1).tolist(), stationFlags = np.isin(tStations, secIDspl.knotsU).tolist())
element['Structure'] = dict(t = [0,1], EI=[100,100], GK=[100,100], xnSC=[0,0], ynSC=[0,0], LELimitEstimate=0.0, TELimitEstimate=0.0, EforEstimate=145, IscaleForEstimate=1.0, GforEstimate=15.66, KscaleForEstimate=1.0)
element['Sections'] = sections

foilDef = dict(
    Version = 3,
    Name = 'T-Foil',
    sExtRef = 0.0,
    iBearingElement = 0,
    Elements = [element]
)

path = 'frontWing.foil'
json_object = json.dumps(foilDef, indent=2)
with open(path, "w") as outfile:
    outfile.write(json_object)