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




    CA, CB, T = x
    CAin, CBin, Tin = u
    n = len(CA)
    dz = p["dz"]
    v = p["v"]
    D = p["D"]
    k = p["k"]
    DA, DB, DT = D
    beta = p["beta"]
    v = p["v"]          #F/A

    # Initialize derivatives
    dCA_dt = np.zeros(n)
    dCB_dt = np.zeros(n)
    dT_dt = np.zeros(n)

    # Reaction term
    r = k(T) * CA * CB

    # Loop over spatial points
    for i in range(n):
        # Convection and diffusion terms for CA
        if i == 0:  # Inlet boundary (Dirichlet)
            NconvA = v * CAin
            JA = 0
        else:
            NconvA = v * CA[i-1]
            JA = -DA * (CA[i] - CA[i-1]) / dz

        if i == n-1:  # Outlet boundary (Neumann)
            NconvA_next = 0
            JA_next = 0
        else:
            NconvA_next = v * CA[i]
            JA_next = -DA * (CA[i+1] - CA[i]) / dz

        NA = NconvA + JA
        NA_next = NconvA_next + JA_next
        dCA_dt[i] = (NA_next - NA) / dz - r[i]

        # Convection and diffusion terms for CB
        if i == 0:  # Inlet boundary (Dirichlet)
            NconvB = v * CBin
            JB = 0
        else:
            NconvB = v * CB[i-1]
            JB = -DB * (CB[i] - CB[i-1]) / dz

        if i == n-1:  # Outlet boundary (Neumann)
            NconvB_next = 0
            JB_next = 0
        else:
            NconvB_next = v * CB[i]
            JB_next = -DB * (CB[i+1] - CB[i]) / dz

        NB = NconvB + JB
        NB_next = NconvB_next + JB_next
        dCB_dt[i] = (NB_next - NB) / dz - 2 * r[i]

        # Convection and diffusion terms for T
        if i == 0:  # Inlet boundary (Dirichlet)
            NconvT = v * Tin
            JT = 0
        else:
            NconvT = v * T[i-1]
            JT = -DT * (T[i] - T[i-1]) / dz

        if i == n-1:  # Outlet boundary (Neumann)
            NconvT_next = 0
            JT_next = 0
        else:
            NconvT_next = v * T[i]
            JT_next = -DT * (T[i+1] - T[i]) / dz

        NT = NconvT + JT
        NT_next = NconvT_next + JT_next
        dT_dt[i] = (NT_next - NT) / dz + beta * r[i]

    # Combine derivatives into a single vector
    xdot = np.concatenate([dCA_dt, dCB_dt, dT_dt])
    return xdot

def jacobian_PFR3(t, x, u, p):
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
    # Unpack parameters
    CAin, CBin, Tin = u
    n = len(x) // 3
    dz = p["dz"]
    v = p["v"]
    D = p["D"]
    k = p["k"]
    DA, DB, DT = D
    beta = p["beta"]

    # Initialize Jacobian matrix
    J = np.zeros((3*n, 3*n))

    # Fill in the Jacobian matrix based on the model equations

    # We start with the derivatives of CA
    for i in range(n):
        # Diagonal elements
        J[i, i] = -v / dz - 2 * k(x[i]) * x[n + i] - (DA / dz**2)
        if i > 0:
            J[i, i-1] = DA / dz**2
        if i < n - 1:
            J[i, i+1] = DA / dz**2

        # Off-diagonal elements
        J[i, n + i] = -k(x[i]) * x[i]
        J[i, 2*n + i] = beta * k(x[i]) * x[n + i]
        J[i, 3*n + i] = -2 * k(x[i]) * x[n + i]
        


    return J



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
    n = len(x)
    dz = p["dz"]
    v = p["v"]
    D = p["D"]
    k = p["k"]
    DT = D
    beta = p["beta"]
    v = p["v"]          #F/A

    # Initialize derivatives
    dT_dt = np.zeros(n)

    CA = CAin + (1 / beta) * (Tin - x)
    CB = CBin + (2 / beta) * (Tin - x)


    # Loop over spatial points
    for i in range(n):
        # Convection and diffusion terms for T
        if i == 0:  # Inlet boundary (Dirichlet)
            NconvT = v * Tin
            JT = 0
        else:
            NconvT = v * x[i-1]
            JT = -DT * (x[i] - x[i-1]) / dz

        if i == n-1:  # Outlet boundary (Neumann)
            NconvT_next = 0
            JT_next = 0
        else:
            NconvT_next = v * x[i]
            JT_next = -DT * (x[i+1] - x[i]) / dz

        NT = NconvT + JT
        NT_next = NconvT_next + JT_next
        dT_dt[i] = (NT_next - NT) / dz + beta * k(x[i]) * CA[i] * CB[i]

    return dT_dt