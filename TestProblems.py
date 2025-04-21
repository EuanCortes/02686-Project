import numpy as np
import matplotlib.pyplot as plt

# Function for the prey-predator model
def prey_predator_model(alpha, beta):
    def f(t, z):
        x, y = z
        dxdt = alpha * x - beta * x * y
        dydt = beta * x * y - alpha * y
        return np.array([dxdt, dydt])
    
    def jacobian(t, z):
        x, y = z
        df_dx = alpha - beta * y
        df_dy = -beta * x
        dg_dx = beta * y
        dg_dy = beta * x - alpha
        return np.array([[df_dx, df_dy],
                         [dg_dx, dg_dy]])
    
    return f, jacobian

#Function for the Van der Pol model
def van_der_pol_model(mu):
    def f(t, z):
        x, y = z
        dxdt = y
        dydt = mu * (1 - x**2) * y - x
        return np.array([dxdt, dydt])
    
    def jacobian(t, z):
        x, y = z
        df_dx = 0
        df_dy = 1
        dg_dx = -2 * mu * x * y - 1
        dg_dy = mu * (1 - x**2)
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


def CSTR_3state_model(params):
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

        return np.array([dCA_dt, dCB_dt, dT_dt])
    
    def jacobian(t, x):
        CA, CB, T = x
        beta = - deltaHr / (p * cp) 
        # Jacobian matrix
        J = np.array([[ -F/V - k0 * np.exp(-Ea_R / T)*CB, - k0 * np.exp(-Ea_R / T) * CA, - (Ea_R / T**2) *(k0 * np.exp(-Ea_R / T) * CA * CB)],
                        [ - 2*k0 * np.exp(-Ea_R / T)*CB, - F/V - 2*k0 * np.exp(-Ea_R / T) * CA, - (Ea_R / T**2) *(2*k0 * np.exp(-Ea_R / T) * CA * CB)],
                        [ beta * k0 * np.exp(-Ea_R / T) * CB, beta * k0 * np.exp(-Ea_R / T) * CA, -F/V - (Ea_R / T**2) *(beta * k0 * np.exp(-Ea_R / T) * CA * CB)]])
        return J
    
    return f, jacobian

# Function for the CSTR (1 state Model)

def CSTR_1state_model(params):
    F, V, CA_in, CB_in, Tin = params

    def f(t, T):
        def k(T):
            return k0 * np.exp(-Ea_R / T)
        # Unpack parameters
        CA = CA_in + (1 / beta) * (Tin - T)
        CB = CB_in + (2 / beta) * (Tin - T)
        r = k(T) * CA * CB
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
        
        return J  
    return f, jacobian

"""""
def CSTR(t, CA, CB, T, params):
    
    Implementation of the CSTR model

    Parameters:
    t : float
        Time variable
    CA : float
        Concentration of A
    CB : float
        Concentration of B
    T : float
        Temperature
    params : array-like
        Model parameters [F, V, CAin, CBin, Tin]
        F : Flow rate
        V : Volume of the reactor
        CAin : Inlet concentration of A
        CBin : Inlet concentration of B
        Tin : Inlet temperature
    Returns:
    dCA_dt : float
        Rate of change of concentration of A
    dCB_dt : float
        Rate of change of concentration of B
    dT_dt : float
        Rate of change of temperature
 
    beta = - deltaHr / (p * cp) 

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
    # Unpack parameters
    F, V, CAin, CBin, Tin = params

    # Calculate derivatives
    dCA_dt = (F/V) * (CAin-CA) + RA(T, CA, CB)
    dCB_dt = (F/V) * (CBin - CB ) + RB(T, CA, CB)
    dT_dt = (F/V) * (Tin - T) + RT(T, CA, CB)

    return np.array([dCA_dt, dCB_dt, dT_dt])

# Function for the CSTR (3 state Model) Jacobian

def CSTR_jacobian(t, x, params):

    Jacobian of the CSTR model

    Parameters:
    t : float
        Time variable
    x : array-like
        State variables [CA, CB, T]
    params : array-like
        Model parameters [F, V, CAin, CBin, Tin]

    Returns:
    J : array-like
        Jacobian matrix
    # Unpack parameters
    F, V, CAin, CBin, Tin = params

    # Unpack state variables
    CA, CB, T = x
    beta = - deltaHr / (p * cp) 
    # Jacobian matrix
    J = np.array([[ -F/V - k0 * np.exp(-Ea_R / T)*CB, - k0 * np.exp(-Ea_R / T) * CA, - (Ea_R / T**2) *(k0 * np.exp(-Ea_R / T) * CA * CB)],
                  [ - 2*k0 * np.exp(-Ea_R / T)*CB, - F/V - 2*k0 * np.exp(-Ea_R / T) * CA, - (Ea_R / T**2) *(2*k0 * np.exp(-Ea_R / T) * CA * CB)],
                  [ beta * k0 * np.exp(-Ea_R / T) * CB, beta * k0 * np.exp(-Ea_R / T) * CA, -F/V - (Ea_R / T**2) *(beta * k0 * np.exp(-Ea_R / T) * CA * CB)]])
    
    return J

"""

# Models for PFR 1 state and 3 state, Pipe Advection Diffusion Reaction.
# For the 3 state model:

def PFR_3state_model(t, x, u, p):
    """
    PFR Advection-Diffusion-Reaction model with 3 states: CA, CB, T.

    PDE System:
        dCA/dt = -v * dCA/dz + D * d2CA/dz2 - k(T) * CA * CB
        dCB/dt = -v * dCB/dz + D * d2CB/dz2 - 2k(T) * CA * CB
        dT/dt  = -v * dT/dz  + D * d2T/dz2  + beta * k(T) * CA * CB

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
            - Nz : int       (number of spatial points)
            - dz : float     (spatial step)
            - v : float      (velocity)
            - D : float      (diffusivity)
            - beta : float   (heat of reaction)
            - Ea_R : float   (activation energy / gas constant)
            - k0 : float     (pre-exponential factor)

    Returns:
    --------
    xdot : array-like
        Time derivatives of state variables
    """
    # Unpack parameters
    n = p["Nz"]
    dz = p["dz"]
    v = p["v"]
    D = p["D"]
    beta = p["beta"]
    Ea_R = p["Ea_R"]
    k0 = p["k0"]

    # Unpack inputs
    cAin, cBin, Tin = u

    # Unpack states
    cA = x[0:n]
    cB = x[n:2*n]
    T  = x[2*n:3*n]

    # Reaction rate function (Arrhenius)
    def k(T_local):
        return k0 * np.exp(-Ea_R / T_local)

    # Allocate derivatives
    dcA_dt = np.zeros(n)
    dcB_dt = np.zeros(n)
    dT_dt  = np.zeros(n)

    # Finite difference loop
    for i in range(n):
        # Local reaction rate
        r = k(T[i]) * cA[i] * cB[i]

        if i == 0:
            # Inlet boundary (Dirichlet for CA, CB, T)
            dCA_dz = (cA[i] - cAin) / dz
            dCB_dz = (cB[i] - cBin) / dz
            dT_dz  = (T[i] - Tin) / dz

            d2CA_dz2 = 0
            d2CB_dz2 = 0
            d2T_dz2  = 0

        elif i == n - 1:
            # Outlet boundary (Neumann ∂C/∂z = 0, so ∂²C/∂z² ≈ 0)
            dCA_dz = (cA[i] - cA[i-1]) / dz
            dCB_dz = (cB[i] - cB[i-1]) / dz
            dT_dz  = (T[i] - T[i-1]) / dz

            d2CA_dz2 = 0
            d2CB_dz2 = 0
            d2T_dz2  = 0

        else:
            # Interior nodes: upwind (1st) and central (2nd)
            dCA_dz = (cA[i] - cA[i-1]) / dz
            dCB_dz = (cB[i] - cB[i-1]) / dz
            dT_dz  = (T[i]  - T[i-1])  / dz

            d2CA_dz2 = (cA[i+1] - 2*cA[i] + cA[i-1]) / dz**2
            d2CB_dz2 = (cB[i+1] - 2*cB[i] + cB[i-1]) / dz**2
            d2T_dz2  = (T[i+1]  - 2*T[i]  + T[i-1])  / dz**2

        # Advection + diffusion + reaction
        dcA_dt[i] = -v * dCA_dz + D * d2CA_dz2 - r
        dcB_dt[i] = -v * dCB_dz + D * d2CB_dz2 - 2 * r
        dT_dt[i]  = -v * dT_dz  + D * d2T_dz2  + beta * r

    # Stack derivatives
    xdot = np.concatenate([dcA_dt, dcB_dt, dT_dt])
    return xdot
