import numpy as np
import control as c
import atm_std as atm
import sisopy31 as siso
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class Aeronef():
    def __init__(self, M = 0.91, z=10085):
        self.z0 = z*0.3048            # altitude : m
        self.m = 8400                 # mass     : kg
        self.Cx0 = 0.022              # drag coeff for null incidence 
        self.Cz_alpha = 3.2           # lift gradient coeff
        self.Cz_deltam = 1.25         # lift gradient coeff
        self.delta_m0 = 0.02          # equilibrium fin deflection for null lift
        self.alpha0 = 0.01            # incidence for null lift and null deflection
        self.f = 0.56                 # aerodynamic center for body and wings
        self.f_delta = 0.81           # aerodynamic center of fins
        self.k = 0.26                 # polar coeff
        self.Cm_q = -0.68             # damping coeff
        self.l_ref = 5.24             # reference length : m  
        self.l_t = self.l_ref*3/2     # total length     : m
        self.g0 = 9.81                # gravitational force
        self.c = 0.52                 # position of the COG (52% of the fuselage)
        self.c_new = self.f*1.1       # new COG position  
        self.S = 34                   # reference surface  : m²
        self.rg = 2.65                # radius of gyration : m
        self.tau = 0.75               # time constant      : s
        # dynamic pressure Pa
        self.gamma0 = 0              
        self.alpha0 = 0               
        self.theta0 = 0               
        self.F = self.f*self.l_t
        self.F_delta = self.f_delta*self.l_t
        self.Iyy = self.m*self.rg**2


    def cog(self, c_val):
        """ Returns the center of gravity from its position"""
        self.G  = c_val*self.l_t
        self.dx = c_val*self.l_t - self.f*self.l_t
        self.dy = c_val*self.l_t - self.f_delta*self.l_t


    def compute_alpha_eq(self, X):
        """Compute state space matrices of the model.

        Args:
            X (np.array([V, gamma, aplha, Q, tetha, z]))

        Returns:
            C_xeq, C_zeq, alpha_eq, delta_meq, F_pxeq : flight parameters
        """
        alpha0 = X[2]
        Q0 = X[3]
        print(Q0)
        alpha_eq = 0
        F_pxeq = 0

        err = -1
        epislon = 1e-5
        while err > epislon or err == -1:

            C_zeq     = (self.m * self.g0 - F_pxeq * np.sin(alpha_eq)) / (Q0 * self.S)
            C_xeq     = self.Cx0 + self.k * C_zeq**2

            C_xdeltam = 2 * self.k * C_zeq * self.Cz_deltam

            num       = C_xeq * np.sin(alpha_eq) + C_zeq * np.cos(alpha_eq)
            den       = C_xdeltam * np.sin(alpha_eq) + self.Cz_deltam * np.cos(alpha_eq)
            delta_meq = self.delta_m0 - num * self.dx / (den * (self.dy - self.dx))

            correction   = C_zeq / self.Cz_alpha - self.Cz_deltam * delta_meq / self.Cz_alpha
            alpha_eq_new = alpha0 + correction
            F_pxeq       = Q0 * self.S * C_xeq / np.cos(alpha_eq)

            err = abs(alpha_eq_new - alpha_eq)
            alpha_eq = alpha_eq_new

        return C_xeq, C_zeq, alpha_eq, delta_meq, F_pxeq


    def state_space(self, X):
        """Compute state space matrices of the model.

        Args:
            X (np.array([V, gamma, aplha, Q, tetha, z]))

        Returns:
            tuple(A, B, C, D): State space matrices.
        """
        
        gamma_eq = 0
        Ft=0
        Q = X[3]
        self.V_eq = X[0]
        # Calculus simplification variables.
        Cx_eq, Cz_eq, self.alpha_eq, delta_eq, F_eq = self.compute_alpha_eq(X)
        print(self.alpha_eq)
        QSV = Q * self.S / (self.m*self.V_eq)
        QSI = Q * self.S * self.l_ref / self.Iyy
        Cx_alpha  = 2 * self.k * Cz_eq * self.Cz_alpha
        Cx_deltam = 2 * self.k * Cz_eq * self.Cz_deltam
        Cm_alpha  = self.dx * (Cx_alpha * np.sin(self.alpha_eq) + self.Cz_alpha * np.cos(self.alpha_eq)) / self.l_ref
        Cm_deltam = self.dy * (Cx_deltam * np.sin(self.alpha_eq) + self.Cz_deltam * np.cos(self.alpha_eq)) / self.l_ref
        
        Xv       = 2 * QSV * Cx_eq
        X_alpha  = F_eq * np.sin(self.alpha_eq) / (self.m*self.V_eq) + QSV * Cx_alpha
        X_gamma  = self.g0 * np.cos(gamma_eq) / self.V_eq

        self.m_alpha  = QSI * Cm_alpha
        self.m_q      = QSI * self.l_ref * self.Cm_q / self.V_eq
        self.m_deltam = QSI * Cm_deltam

        Zv            = 2 * QSV * Cz_eq
        self.Z_alpha  = F_eq * np.cos(self.alpha_eq) / (self.m*self.V_eq) + QSV * self.Cz_alpha
        self.Z_deltam = QSV * self.Cz_deltam

        A = np.array([[-Xv,  -X_gamma,     -X_alpha,         0, 0, 0],
                      [ Zv,         0,  self.Z_alpha,        0, 0, 0],
                      [-Zv,         0, -self.Z_alpha,        1, 0, 0],
                      [  0,         0,  self.m_alpha, self.m_q, 0, 0],
                      [  0,         0,             0,        1, 0, 0],
                      [  0, self.V_eq,             0,        0, 0, 0]])
        
        B       = np.array([[0, self.Z_deltam, -self.Z_deltam, self.m_deltam, 0, 0]]).T
        C       = np.zeros((1,6))
        C[0, 2] = 1
        D       = np.zeros((1,1))

        print("\n========================")
        print(A)
        print(B)
        print(C)
        print(D)
        print("========================\n")

        return A, B, C, D
    

    def short_period(self, A, B):
        """Compute state space for short period mode.

        Args:
            A, B: State space matrices.

        Returns:
            sp_alpha_ss, sp_alpha_tf, sp_q_ss, sp_q_tf: state spaces and transfer functions
        """
        Asp = A[2:4 , 2:4]
        Bsp = B[2:4 , 0:1]
        Csp_alpha = np.matrix ( [[1 , 0]] )
        Csp_q = np.matrix ( [[0 , 1]] )
        Dsp = np.matrix ( [[0]])

        sp_alpha_ss = c.ss(Asp,Bsp,Csp_alpha,Dsp)
        c.matlab.damp(sp_alpha_ss)
        print( "\n\nTransfer function alpha / delta_m =" )
        sp_alpha_tf = c.matlab.ss2tf(sp_alpha_ss)
        print(sp_alpha_tf)
        print ( "Static gain of alpha / delta_m = %f "%(c.dcgain(sp_alpha_ss)))

        sp_q_ss = c.ss(Asp,Bsp,Csp_q,Dsp)
        print( "\n\nTransfer function q / delta_m =" )
        sp_q_tf = c.matlab.ss2tf(sp_q_ss)
        print(sp_q_tf)
        print ( "Static gain of q / delta_m = %f "%(c.dcgain(sp_q_ss)))
        return sp_alpha_ss, sp_alpha_tf, sp_q_ss, sp_q_tf
    

    def phugoid_mode(self, A, B):
        """Compute state space for phugoid mode.

        Args:
            A, B: State space matrices.

        Returns:
            ph_v_ss, ph_v_tf, ph_gammma_ss, ph_gammma_tf: state spaces and transfer functions
        """
        Ap=A[ 0 : 2 , 0 : 2 ]
        Bp=B[ 0 : 2 , 0 : 1 ]
        Cpv = np.matrix ( [[1 , 0]] )
        Cpg = np.matrix ( [[0 , 1]] )
        Dp = np.matrix ( [[0]])
        ph_v_ss =  c.ss(Ap ,Bp ,Cpv ,Dp)
        c.matlab.damp(ph_v_ss)
        print( "\n\nTransfer function V / delta_m =" )
        ph_v_tf = c.tf ( ph_v_ss )
        print( ph_v_tf )
        print ( "Static gain of V / delta_m = %f "%(c.dcgain(ph_v_tf)))

        ph_gammma_ss = c.ss(Ap ,Bp ,Cpg ,Dp)
        print( "\n\nTransfer function gamma / delta_m =" )
        ph_gammma_tf = c.matlab.ss2tf(ph_gammma_ss)
        print(ph_gammma_tf)
        print ( "Static gain of q / delta_m = %f "%(c.dcgain(ph_gammma_ss)))
        return ph_v_ss, ph_v_tf, ph_gammma_ss, ph_gammma_tf
    
    
    def open_loop(self, A, B, C, D, mode, title):
        """Plot system in open loop.

        Args:
            A, B, C, D: State space matrices.
            mode, title : info on the system

        Returns:
            plot of the system
        """
        ss = c.ss(A, B, C, D)
        ri, a, b, xi, w, st = siso.damp(ss)
        print(f"\n----------------- {mode} Mode :    --------------------------------------\n ")
        for i in st :
            print(i)
        tf = siso.ss2tf(ss)
        print(f"Transfer function : \n ", tf)
        Y, t=c.matlab.step(tf, 10)
        plt.plot(t,Y, label= title)
        plt.title("(Open Loop) Step Response "+title)
        plt.xlabel("Time (s)")
        plt.ylabel("(rad)")

    
    def correction_open_loop(self, A, B, C, D, k):
        """Correct the state space in open loop.

        Args:
            A, B, C, D: State space matrices.
            mode, title : info on the system

        Returns:
            Ak, Bk: new state space matrices
            closed_loop_ss, closed_loop_tf : state space and transfer function
        """
        print("k = ", k)
        Ak = A - k * np.dot(B,C)
        Bk = k * B
        Dk = k * D
        closed_loop_ss = c.ss(Ak, Bk, C, Dk)
        closed_loop_tf = siso.ss2tf(closed_loop_ss)
        print(f"Transfer function : \n ", closed_loop_tf)
        ri, a, b, xi, w, st = siso.damp(closed_loop_ss)
        for i in st : 
            print(i)
        return Ak, Bk, closed_loop_ss, closed_loop_tf

