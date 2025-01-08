import numpy as np
import sisopy31 as siso
import atm_std as atm
from aeronef import *
import control

aero = Aeronef()
aero.cog(aero.c)
M = 0.91

z0 = 10085*0.3048
m  = 8400           

gamma0 = 0
alpha0 = 0
theta0 = 0

hgeo, rho, a = atm.get_cte_atm(z0)
Veq = M*a
Q   = (rho*Veq**2) /2
X = np.array([Veq, gamma0, alpha0, Q, theta0, z0]).T

A, B, C, D = aero.state_space(X)
state_space = c.ss(A,B,C,D)
ri, a, b, xi, w, st = siso.damp(state_space)

for i in st : 
    print(i)

"""
--------------------Step responses in short-period mode : alpha and q ------------------------
"""
# System
sys_a, tf_a, sys_q, tf_q = aero.short_period(A, B)
tf_q_OL = tf_q # tf_q in open loop

# Curves
Ya , Ta = c.matlab.step(sys_a , 10 )
Yq , Tq = c.matlab.step(sys_q , 10 )
plt.plot(Ta ,Ya , 'b' , Tq ,Yq , 'r', lw=2)

# Alpha info
step_info1 = c.step_info(Ya,Ta)
Tsa = step_info1['SettlingTime']
Tra = step_info1['RiseTime']
Osa = step_info1['Overshoot']
yya = interp1d(Ta,Ya)
plt.plot(Tsa, yya(Tsa),'bs')
plt.text(Tsa, yya(Tsa)+0.2, round(Tsa,2))

# q info
step_info2 = c.step_info(Yq,Tq)
Tsq = step_info2['SettlingTime']
Trq = step_info2['RiseTime']
Osq = step_info2['Overshoot']
yyq = interp1d(Tq,Yq)
plt.plot(Tsq, yyq(Tsq),'rs')
plt.text(Tsq, yyq(Tsq)-0.3, Tsq)

# Settling times
print('alpha Setlling time 5%% = %f s'%Tsa)
print('q Setlling time 5%%     = %f s'%Tsq)

plt.title(r' SHORT PERIOD MODE: step response $\alpha/\delta_m$ and $q/\delta_m$')
plt.legend((r'$\alpha/\delta_m$ ', r'$q/\delta_m$'))
plt.xlabel ( 'Time (s)' )
plt.ylabel ( r'$\alpha$ (rad) & $q$ ( rad/s )' )
plt.show()

"""
------------- Step responses for phugoid mode : v and gamma -------------------------------
"""
# System
ph_v_ss, ph_v_tf, ph_gammma_ss, ph_gammma_tf= aero.phugoid_mode(A, B)

# Curves
Yv , Tv = c.matlab.step(ph_v_ss, 700)
Yg , Tg = c.matlab.step(ph_gammma_ss, 700 )
plt.plot(Tv ,Yv , 'b' , Tg ,Yg , 'r', lw=2)

# V info
step_info1 = c.step_info(Yv,Tv)
Tsv = step_info1['SettlingTime']
Trv = step_info1['RiseTime']
Osv = step_info1['Overshoot']
yyv = interp1d(Tv,Yv)
plt.plot(Tsv, yyv(Tsv),'bs')
plt.text(Tsv, yyv(Tsv)-0.2, Tsv)

# Gamma info
step_info2 = c.step_info(Yg,Tg)
Tsg = step_info2['SettlingTime']
Trg = step_info2['RiseTime']
Osg = step_info2['Overshoot']
yyg=interp1d(Tg,Yg)
plt.plot(Tsg, yyg(Tsg),'rs')
plt.text(Tsg, yyg(Tsg)-0.2, Tsg)

# Settling times
print('alpha Setlling time 5%% = %f s'%Tsv)
print('q Setlling time 5%%     = %f s'%Tsg)

plt.title(r' PHUGOID MODE: step response $V/\delta_m$ and $\gamma/\delta_m$')
plt.legend((r'$V/\delta_m$ ', r'$\gamma/\delta_m$'))
plt.xlabel ( 'Time (s)' )
plt.ylabel ( r'$V$ (rad) & $\gamma$ ( rad/s )' )
plt.show()

# -----------------------------------------------------------------------------
# Perfect Auto-throttle

A_new = A[1:, 1:] # size 5 without V
B_new = B[1:]     # size 5 without V

C_gamma= np.array([[1,0,0,0,0]])
C_alpha=np.array([[0,1,0,0,0]])
C_q= np.array([[0,0,1,0,0]])
C_z= np.array([[0,0,0,0,1]])

aero.open_loop(A_new, B_new, C_q, D, mode = "", title = " q")
plt.show()

"""
-------------------- q feedback loop : --------------------------------
"""
# Value determined with the rootlocus graph
Kr=-0.0784

Aq, Bq, sys_q, tf_q= aero.correction_open_loop(A_new, B_new, C_q, D, k=Kr)
tf_q_CL = tf_q # tf_q in closed loop
print("Aq : ", Aq, "Bq : ", Bq)


Y, t = c.matlab.step(sys_q, 10) # 10 seconds
plt.plot(t, Y)
plt.title("Step response of the closed loop (q feedback loop)")
plt.xlabel("Time (s)")
plt.ylabel("Pitch Rotation Speed (rad/s)")
plt.show()
sys_calpha = c.ss(Aq, Bq, C_alpha, D)

""" 
--------Washout filter : ------------
"""

tau = 0.7
washout= c.tf([tau, 0], [tau, 1])
washout_closed = c.feedback(Kr, tf_q_OL * washout)

sys_FTBO = c.tf(c.ss(A_new, B_new, C_alpha, D))
sys_FTBF = c.series(1/Kr, c.feedback(Kr, tf_q_OL), tf_a) 
sys_FTBF_filter = c.series(1 / Kr, washout_closed, tf_a)

T = 10
step_1 , t_1 = c.matlab.step(sys_FTBO,T = T)
step_2 , t_2 = c.matlab.step(sys_FTBF,T = T)
step_3 , t_3 = c.matlab.step(sys_FTBF_filter,T = T)

plt.plot(t_1, step_1, label = "Open Loop")
plt.plot(t_2, step_2, label = "Closed Loop")
plt.plot(t_3, step_3, label = "Closed Loop and Washout Filter")

plt.legend()
plt.title("Step response")
plt.xlabel("Time (in s)")
plt.ylabel(r'$\alpha$')
plt.show()

""" 
--------------------- Gamma feedback loop : ---------------------------
"""
syst = c.minreal(control.tf(c.ss(Aq, Bq, C_gamma , D)))
siso.sisotool(syst)

# Value determined from the conditions of the second tuning
kgamma =  7.506 
Agamma, Bgamma, sys_gamma, tf_gamma = aero.correction_open_loop(Aq, Bq, C_gamma, D, k=kgamma)
print("Agamma : ", Agamma, "Bgamma : ", Bgamma)

Y, t = c.matlab.step(tf_gamma, 8) # 8 seconds
plt.plot(t, Y)
plt.title("Step response of the closed loop (gamma feedback loop)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

""" 
------------------------------ z feedback loop : ------------------------------------
"""
# Value determined from the conditions of the tuning
kz = 0.00372
Az, Bz, sys_z, tf_z = aero.correction_open_loop(Agamma, Bgamma, C_z, D, k=kz)
print("Az : ", Az, "Bz : ", Bz)

Y, t = c.matlab.step(sys_z, 10) # 10 seconds
plt.plot(t, Y)
plt.title("Step response of the closed loop (z feedback loop)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

""" 
------------------------ New C.O.G -------------------------
"""
aero.cog(aero.c_new)
A_new_cog, B_new_cog, C_new_cog, D_new_cog = aero.state_space(X)

sys_new_cog =  control.ss(A_new_cog, B_new_cog, C_new_cog, D_new_cog)
sys_tf_new_cog = control.tf(sys_new_cog)
print("Anew = ", A_new_cog, "Bnew = ", B_new_cog)
print(sys_tf_new_cog)
control.matlab.damp(sys_new_cog)

q_new , t_new = control.matlab.step(sys_tf_new_cog)
plt.plot(t_new,q_new)
plt.title("Step response of the new system model")
plt.xlabel("Time (s)")
plt.ylabel('Response')
plt.show()
