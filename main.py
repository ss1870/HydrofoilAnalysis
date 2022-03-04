from source.classes import LiftingSurface, FoilAssembly, ms2knts, knts2ms
from source.LL_functions import eval_biot_savart, LL_residual
import numpy as np
import matplotlib.pyplot as plt
from numpy import matlib
import jax.numpy as jnp
from jax import grad, jit, vmap

# rho*u*L/u
# Seawater properties: https://www.engineeringtoolbox.com/sea-water-properties-d_840.html
# Assumes 15 deg temperature seawater
u = 5  # flow speed in m/s
chord = 0.2  # characteristic length
re = u * chord * 1026 / 0.00126
print("Reynolds number = ", str(re), "\n")

# Define simple motion vector, moving forward in y at 10 knots
u_motion = np.array([0, knts2ms(10), 0]).reshape(1, 3)
print("Input velocity vector = ", str(u_motion), " m/s (= ", str(ms2knts(u_motion)), " knots)\n ")

# Instantiate a front wing
front_wing = LiftingSurface(rt_chord=250,
                            tip_chord=80,
                            span=800,
                            Re=re,
                            sweep_tip=-200,
                            sweep_curv=3,
                            dih_tip=-75,
                            dih_curve=2,
                            afoil_name='naca2412')

# Instantiate stabiliser
stabiliser = LiftingSurface(rt_chord=90,
                            tip_chord=50,
                            span=500,
                            Re=re,
                            sweep_tip=-30,
                            sweep_curv=2,
                            dih_tip=30,
                            dih_curve=8,
                            afoil_name='naca0012')

# Instantiate a mast
mast = LiftingSurface(rt_chord=130,
                      tip_chord=130,
                      span=600,
                      Re=re,
                      type='mast',
                      afoil_name='naca0015')  # Axis mast is 19 mm thick, and around 130 mm chord = ~15% thickness

# Assemble the foil
foil = FoilAssembly(front_wing,
                    stabiliser,
                    mast,
                    fuselage_length=699 - 45 - 45,  # assumes AXIS short black fuselage
                    mast_attachment_ratio=267 - 45,  # assumes AXIS short black fuselage
                    wing_angle=1,
                    stabiliser_angle=-2)

foil.rotate_foil_assembly([1, 0, 0])
print(np.sum(foil.compute_foil_loads(u_motion, 1025), axis=0))
foil.rotate_foil_assembly([-1, 0, 0])

angle = np.linspace(-5,10,16)
mom = np.zeros(angle.shape)
for i in range(len(angle)):
    foil.rotate_foil_assembly([angle[i], 0, 0])
    loads = np.sum(foil.compute_foil_loads(u_motion, 1025), axis=0)
    mom[i] = loads[3]
    foil.rotate_foil_assembly([-angle[i], 0, 0])

# plt.plot(angle, mom, 'k-')
# plt.grid(True)
# plt.show()
# foil.plot_foil_assembly()




xnode1 = jnp.concatenate([obj.node1.reshape(1, 1, -1) for obj in front_wing.BVs], axis=1)
xnode2 = jnp.concatenate([obj.node2.reshape(1, 1, -1) for obj in front_wing.BVs], axis=1)
# gamma = np.array([obj.circ for obj in front_wing.BVs]).reshape(1, -1, 1)
l0 = jnp.array([obj.length0 for obj in front_wing.BVs]).reshape(1, -1, 1)
gamma = jnp.ones(l0.shape)

xcp = front_wing.xcp

u_BV = eval_biot_savart(xcp, xnode1, xnode2, gamma, l0)
fast_BS = jit(eval_biot_savart)
u_BV1 = fast_BS(xcp, xnode1, xnode2, gamma, l0)
print(np.max(u_BV-u_BV1))
# print(u_BV.shape)


gamma = jnp.random.randn(u_BV.shape[0])
rho = 1025
u_FV = jnp.zeros((1,3))
R = LL_residual(gamma, rho, u_BV, u_FV, u_motion, front_wing.dl, front_wing.a1, front_wing.a3, front_wing.cl_spline, front_wing.dA)
print(R)

# to-do:
# - in LiftingSurface class: designate whether a BV is on the LL or not (vtype)
# - modify Cl interpolation in residual so can use jax - need to check whether jax works for interp functins
# - work out auto diff jacobian of residual function and check against finite-diff
# - implement circulation solver using gradient

