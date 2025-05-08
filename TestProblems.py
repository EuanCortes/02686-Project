import numpy as np
import matplotlib.pyplot as plt

# Function for the prey-predator model
def prey_predator_model(alpha, beta, esdirk = False):
    def f(t, z):
        x, y = z
        dxdt = alpha * x - beta * x * y
        dydt = beta * x * y - alpha * y
        if esdirk:
            return np.array([dxdt, dydt]), z
        else:
            return np.array([dxdt, dydt])
    
    def jacobian(t, z):
        x, y = z
        df_dx = alpha - beta * y
        df_dy = -beta * x
        dg_dx = beta * y
        dg_dy = beta * x - alpha
        if esdirk:
            return np.array([[df_dx, df_dy],
                             [dg_dx, dg_dy]]), np.eye(2)
        else:
            return np.array([[df_dx, df_dy],
                         [dg_dx, dg_dy]])
    
    return f, jacobian

#Function for the Van der Pol model
def van_der_pol_model(mu, esdirk = False):
    def f(t, z):
        x, y = z
        dxdt = y
        dydt = mu * (1 - x**2) * y - x
        if esdirk:
            return np.array([dxdt, dydt]), z
        else:
            return np.array([dxdt, dydt])
    
    def jacobian(t, z):
        x, y = z
        df_dx = 0
        df_dy = 1
        dg_dx = -2 * mu * x * y - 1
        dg_dy = mu * (1 - x**2)
        if esdirk:
            return np.array([[df_dx, df_dy],
                             [dg_dx, dg_dy]]), np.eye(2)
        else:
            return np.array([[df_dx, df_dy],
                         [dg_dx, dg_dy]])
    
    return f, jacobian




# Functions for Chemical Reaction in adiabatic reactors
# Model of the CSTR (3 state Model)
# Parameters:
p = 1.0             # Density
cp = 4.186          # Specific heat capacity
k0 = np.exp(24.6)   # Arrhenius constant
Ea_R = 8500.0       # Activation energy   
deltaHr = -56000.0  # Reaction enthalphy
L = 10              # Length of the reactor
A = 0.1             # Cross-sectional area Reactor
D = [0.1, 0.1, 0.1] # Diffusion coefficients
# F is some value between 200 and 800
beta = - deltaHr / (p * cp) 


def CSTR_3state_model(params, esdirk = False):
    # Functions for Chemical Reaction in adiabatic reactors
    # Model of the CSTR (3 state Model)
    # Parameters:
    p = 1.0             # Density
    cp = 4.186          # Specific heat capacity
    k0 = np.exp(24.6)   # Arrhenius constant
    Ea_R = 8500.0       # Activation energy   
    deltaHr = -56000.0  # Reaction enthalphy
    L = 10              # Length of the reactor
    A = 0.1             # Cross-sectional area Reactor
    D = [0.1, 0.1, 0.1] # Diffusion coefficients
    # F is some value between 200 and 800
    beta = - deltaHr / (p * cp) 

    # Unpack parameters
    F, V, CAin, CBin, Tin = params

    def k(T):
        return k0 * np.exp(-Ea_R / T)

    def r(CA, CB, T):
        return k(T) * CA * CB

    def RA(T, CA, CB):
        return -r(CA, CB, T)

    def RB(T, CA, CB):
        return -2*r(CA, CB, T)

    def RT(T, CA, CB):
        return beta * r(CA, CB, T)
    

    def f(t, x):
        CA, CB, T = x

        # Calculate derivatives
        dCA_dt = (F/V) * (CAin-CA) + RA(T, CA, CB)
        dCB_dt = (F/V) * (CBin - CB ) + RB(T, CA, CB)
        dT_dt = (F/V) * (Tin - T) + RT(T, CA, CB)
        if esdirk:
            return np.array([dCA_dt, dCB_dt, dT_dt]), x
        else:

            return np.array([dCA_dt, dCB_dt, dT_dt])
    
    def jacobian(t, x):
        CA, CB, T = x
        beta = - deltaHr / (p * cp) 
        # Jacobian matrix
        J = np.array([[ -F/V - k0 * np.exp(-Ea_R / T)*CB, - k0 * np.exp(-Ea_R / T) * CA, - (Ea_R / T**2) * (k0 * np.exp(-Ea_R / T) * CA * CB)],
                        [ - 2*k0 * np.exp(-Ea_R / T)*CB, - F/V - 2*k0 * np.exp(-Ea_R / T) * CA, - (Ea_R / T**2) *(2*k0 * np.exp(-Ea_R / T) * CA * CB)],
                        [ beta * k0 * np.exp(-Ea_R / T) * CB, beta * k0 * np.exp(-Ea_R / T) * CA, -F/V - (Ea_R / T**2) *(beta * k0 * np.exp(-Ea_R / T) * CA * CB)]])
        if esdirk:
            return J, np.eye(3)
        else:
            return J
    
    return f, jacobian

# Function for the CSTR (1 state Model)

def CSTR_1state_model(params, esdirk = False):
    F, V, CA_in, CB_in, Tin = params

    def f(t, T):
        def k(T):
            return k0 * np.exp(-Ea_R / T)
        # Unpack parameters
        CA = CA_in + (1 / beta) * (Tin - T)
        CB = CB_in + (2 / beta) * (Tin - T)
        r = k(T) * CA * CB
        if esdirk:
            return F / V * (Tin - T) + beta * r, T
        else:
            return F / V * (Tin - T) + beta * r

    # Jacobian for the CSTR (1 state Model)
    def jacobian(t, T):
        def k(T):
            return k0 * np.exp(-Ea_R / T)
        # Unpack parameters
        CA = CA_in + (1 / beta) * (Tin - T)
        CB = CB_in + (2 / beta) * (Tin - T)
        # Jacobian matrix
        J = np.array([-F/V + beta(k(T)*(Ea_R / T**2) * CA * CB + k(T) * CB * (-1/beta) + k(t) * CA * (-2/beta))])
        if esdirk:
            return J, np.eye(1)
        else:
            return J

    return f, jacobian

# Models for PFR 1 state and 3 state, Pipe Advection Diffusion Reaction.
# For the 3 state model:

def PFR_3state_model(u, p, esdirk = False):
    """
    PFR Advection-Diffusion-Reaction model with 3 states: CA, CB, T.

    PDE System:
        dCA/dt = -v * dCA/dz + DA * d2CA/dz2 - k(T) * CA * CB
        dCB/dt = -v * dCB/dz + DB * d2CB/dz2 - 2k(T) * CA * CB
        dT/dt  = -v * dT/dz  + DT * d2T/dz2  + beta * k(T) * CA * CB

    Boundary conditions:
        - Inlet (z=0): Dirichlet (Cin values)
        - Outlet (z=L): Neumann (no diffusion, ∂C/∂z = 0)

    Parameters:
    -----------
    t : float
        Time
    x : array-like
        State vector: [CA_0,...,CA_n-1, CB_0,...,CB_n-1, T_0,...,T_n-1]
    u : array-like
        Inlet conditions: [CAin, CBin, Tin]
    p : dict
        Parameters:
            - L : int        (Length of the reactor)
            - dz : float     (spatial step)
            - v : float      (velocity)
            - D : float      (diffusivity in shape [DA, DB, DT])
            - beta : float   (heat of reaction) = -DeltaHr / (p * cp)
            - k : function   (reaction rate constant)
            - k0 : float     (pre-exponential factor)

    Returns:
    --------
    xdot : array-like   [CA_0',...,CA_n-1', CB_0',...,CB_n-1', T_0',...,T_n-1']
        Time derivatives of state variables
    """
    # Unpack parameters

    
    CAin, CBin, Tin = u
    
    dz = p["dz"]
    v = p["v"]
    D = p["D"]
    k = p["k"]
    DA, DB, DT = D
    beta = p["beta"]
    v = p["v"]          #F/A



    def f(t,x):
        CA, CB, T = x
        n = len(CA)
        # Initialize derivatives
        dCA_dt = np.zeros(n)
        dCB_dt = np.zeros(n)
        dT_dt = np.zeros(n)
        # Reaction term
        r = k(T) * CA * CB

        ### Convection and diffusion terms for CA
        #Convection term
        NconvA = np.zeros(n+1)
        NconvA[0] = v * CAin        # Inlet Boundary (Dirichlet)
        NconvA[1:n+1] = v * CA[0:n]

        #Diffusion term
        JA = np.zeros(n+1)
        JA[1:n] = (-DA / dz) * (CA[1:n] - CA[0:n-1])
        JA[0] = 0                   # Inlet Boundary (Dirichlet)
        JA[-1] = 0                  # Outlet Boundary (Neumann)

        # Flux = Convection + Diffusion
        NA = NconvA + JA

        # Reaction term
        rA = -r

        # Differential Equation for CA
        dCA_dt = (NA[1:n+1] - NA[0:n]) / (-dz) + rA

        ### Convection and diffusion terms for CB
        #Convection term
        NconvB = np.zeros(n+1)
        NconvB[1:n+1] = v * CB[0:n]
        NconvB[0] = v * CBin        # Inlet Boundary (Dirichlet)

        #Diffusion term
        JB = np.zeros(n+1)
        JB[1:n] = (-DB / dz) * (CB[1:n] - CB[0:n-1])
        JB[0] = 0                   # Inlet Boundary (Dirichlet)
        JB[-1] = 0                  # Outlet Boundary (Neumann)

        # Flux = Convection + Diffusion
        NB = NconvB + JB
        # Reaction term
        rB = -2 * r
        # Differential Equation for CB
        dCB_dt = (NB[1:n+1] - NB[0:n]) / (-dz) + rB

        ### Convection and diffusion terms for T
        #Convection term
        NconvT = np.zeros(n+1)
        NconvT[1:n+1] = v * T[0:n]
        NconvT[0] = v * Tin         # Inlet Boundary (Dirichlet)

        #Diffusion term
        JT = np.zeros(n+1)
        JT[1:n] = (-DT / dz) * (T[1:n] - T[0:n-1])
        JT[0] = 0                   # Inlet Boundary (Dirichlet)
        JT[-1] = 0                  # Outlet Boundary (Neumann)
        # Flux = Convection + Diffusion
        NT = NconvT + JT
        # Reaction term
        rT = beta * r
        # Differential Equation for T
        dT_dt = (NT[1:n+1] - NT[0:n]) / (-dz) + rT

        # Combine derivatives into a single vector
        xdot = np.concatenate([dCA_dt, dCB_dt, dT_dt])
        if esdirk:
            return xdot, x
        else:
            return xdot

    def Jacobian(t, x):
        """
        Jacobian of the PFR Advection-Diffusion-Reaction model with 3 states: CA, CB, T.

        Parameters:
        -----------
        t : float
            Time
        x : array-like
            State vector: [CA_0,...,CA_n-1, CB_0,...,CB_n-1, T_0,...,T_n-1]
        u : array-like
            Inlet conditions: [CAin, CBin, Tin]
        p : dict
            Parameters:
                - L : int        (Length of the reactor)
                - dz : float     (spatial step)
                - v : float      (velocity)
                - D : float      (diffusivity in shape [DA, DB, DT])
                - beta : float   (heat of reaction) = -DeltaHr / (p * cp)
                - k : function   (reaction rate constant)
                - k0 : float     (pre-exponential factor)

        Returns:
        --------
        J : array-like
            Jacobian matrix of the system

        """
        CA, CB, T = x
        n = len(CA)
        # Initialize derivatives
        dCA_dt = np.zeros(n)
        dCB_dt = np.zeros(n)
        dT_dt = np.zeros(n)
        # Reaction term
        r = k(T) * CA * CB

        # Initialize Jacobian matrix
        J = np.zeros((n,3,3))

        # Fill in the Jacobian matrix based on the model equations

        # We start with the equation of CAdot

        J[:,0,0] = (v - DA / dz) / (-dz) + k(T) * CB
        J[:,0,1] = - k(T) * CA
        J[:,0,2] = - k(T) * CA * CB * (Ea_R / T**2)
        
        # Next, we fill in the equation of CBdot
        J[:,1,0] = -2 * k(T) * CB
        J[:,1,1] = (v - DB / dz) / (-dz) - 2 * k(T) * CA
        J[:,1,2] = -2 * k(T) * CA * CB * (Ea_R / T**2)

        # Finally, we fill in the equation of Tdot
        J[:,2,0] = beta * k(T) * CB
        J[:,2,1] = beta * k(T) * CA
        J[:,2,2] = (v - DT / dz) / (-dz) + (Ea_R / T**2) * beta * k(T) * CA * CB

        if esdirk:
            return J, np.eye(3)
        return J
    
    return f, Jacobian



def PFR_1state_model(t, x, u, p):
    """
    PFR Advection-Diffusion-Reaction model with 1 states: T.

    PDE System:
        dT/dt = -v * dT/dz + DT * d2T/dz2  + beta * k(T) * CA * CB
    
    Boundary conditions:
        - Inlet (z=0): Dirichlet (Cin values)
        - Outlet (z=L): Neumann (no diffusion, ∂C/∂z = 0)
    
    Parameters:
    -----------
    t : float
        Time
    x : array-like
        State vector: [T_0,...,T_n-1]
    u : array-like
        Inlet conditions: [CAin, CBin, Tin]
    p : dict
        Parameters:
            - L : int        (Length of the reactor)
            - dz : float     (spatial step)
            - v : float      (velocity)
            - D : float      (diffusivity in shape [DT])
            - beta : float   (heat of reaction) = -DeltaHr / (p * cp)
            - k : function   (reaction rate constant)
            - k0 : float     (pre-exponential factor)
    Returns:
    --------
    xdot : array-like   [T_0',...,T_n-1']
        Time derivatives of state variables

    """

    # Unpack parameters
    CAin, CBin, Tin = u
    
    dz = p["dz"]
    v = p["v"]
    D = p["D"]
    k = p["k"]
    DT = D
    beta = p["beta"]
    v = p["v"]          #F/A

    def f(t,x, esdirk = False):

        n = len(x)
        # Initialize derivatives
        dT_dt = np.zeros(n)

        CA = CAin + (1 / beta) * (Tin - x)
        CB = CBin + (2 / beta) * (Tin - x)

        # Reaction term
        r = k(x) * CA * CB
        # Initialize convection and diffusion terms
        NconvT = np.zeros(n+1)
        JT = np.zeros(n+1)

        # Convection term
        NconvT[0] = v * Tin        # Inlet Boundary (Dirichlet)
        NconvT[1:n+1] = v * x[0:n]

        # Diffusion term
        JT[1:n] = (-DT / dz) * (x[1:n] - x[0:n-1])
        JT[0] = 0                   # Inlet Boundary (Dirichlet)
        JT[-1] = 0                  # Outlet Boundary (Neumann)
        # Flux = Convection + Diffusion
        NT = NconvT + JT
        # Reaction term
        rT = beta * r
        # Differential Equation for T
        dT_dt = (NT[1:n+1] - NT[0:n]) / (-dz) + rT
        # Combine derivatives into a single vector
        xdot = dT_dt
        if esdirk:
            return xdot, x
        else:
            return xdot



    def Jacobian(t, x, esdirk = False):
        """
        Jacobian of the PFR Advection-Diffusion-Reaction model with 1 state: T.

        Parameters:
        -----------
        t : float
            Time
        x : array-like
            State vector: [T_0,...,T_n-1]
        u : array-like
            Inlet conditions: [CAin, CBin, Tin]
        p : dict
            Parameters:
                - L : int        (Length of the reactor)
                - dz : float     (spatial step)
                - v : float      (velocity)
                - D : float      (diffusivity in shape [DT])
                - beta : float   (heat of reaction) = -DeltaHr / (p * cp)
                - k : function   (reaction rate constant)
                - k0 : float     (pre-exponential factor)

        Returns:
        --------
        J : array-like
            Jacobian matrix of the system
        """
        # Unpack parameters

        n = len(x)

        # Initialize Jacobian matrix
        J = np.zeros((n))

        # Compute CA and CB based on T
        CA = CAin + (1 / beta) * (Tin - x)
        CB = CBin + (2 / beta) * (Tin - x)


        J = np.array([(v - DT / dz) / (-dz) + (Ea_R / x**2) * beta * k(x) * CA * CB])

        if esdirk:
            return J, np.eye(1)
        else:
            return J

    return f, Jacobian