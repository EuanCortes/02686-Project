import numpy as np

# Function for the Van der Pol model
# Function for the Van der Pol model
def van_der_pol_model(mu, esdirk=False, SDE=False, SDE2 = False, sigma = 0.05):
    def f(t, z):
        x, y = z
        dxdt = y
        dydt = mu * (1 - x**2) * y - x
        if esdirk:
            return np.array([dxdt, dydt]), z
        else:
            return np.array([dxdt, dydt])
    
    def g(t, z):
        x, y = z
        return np.array([0, sigma * (1.0 + x**2)])
    
    def g2(t, z):
        x, y = z
        return np.array([0, sigma])

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
    
    if SDE:
        return f, g, jacobian
    elif SDE2:
        return f, g2, jacobian
    else:
        return f, jacobian

def CSTR_1state_model(params, sde = False, esdirk = False, sigma_T = 0.1):
    p = 1.0             # Density
    cp = 4.186          # Specific heat capacity
    k0 = np.exp(24.6)   # Arrhenius constant
    Ea_R = 8500.0       # Activation energy   
    deltaHr = -560.0  # Reaction enthalphy
    L = 10              # Length of the reactor
    A = 0.1             # Cross-sectional area Reactor
    D = [0.1, 0.1, 0.1] # Diffusion coefficients
    # F is some value between 200 and 800
    beta = - deltaHr / (p * cp) 

    F, V, CA_in, CB_in, Tin = params
    V = 0.105
    def f(t, T):
        def k(T):
            if T == 0:# Prevent division by zero
                return k0 * np.exp(-Ea_R / 1e-10)
            else:
                return k0 * np.exp(-Ea_R / T)
        # Unpack parameters
        CA = CA_in + (1 / beta) * (Tin - T)
        CB = CB_in + (2 / beta) * (Tin - T)
        r = k(T) * CA * CB
        if esdirk:
            return F / V * (Tin - T) + beta * r, T
        else:
            return F / V * (Tin - T) + beta * r
        
    def g(t, T):
        return np.array([(F/V)*sigma_T])

    # Jacobian for the CSTR (1 state Model)
    def jacobian(t, T):
        def k(T):
            if T == 0: # Prevent division by zero
                return k0 * np.exp(-Ea_R / 1e-10)
            else:
                return k0 * np.exp(-Ea_R / T)
        # Unpack parameters
        CA = CA_in + (1 / beta) * (Tin - T)
        CB = CB_in + (2 / beta) * (Tin - T)

        # Prevent division by zero
        e2 = (Ea_R / T**2) if T != 0 else (Ea_R / 1e-10**2)
        # Jacobian matrix
        J = np.array([-F/V + beta*(k(T)*e2 * CA * CB + k(T) * CB * (-1/beta) + k(T) * CA * (-2/beta))])
        if esdirk:
            return np.array([J]), np.eye(1)
        else:
            return np.array(J)

    return f, g, jacobian

def PFR_1state_model(u, p, sde = True, sigma_T = 5):

    # Unpack parameters
    CAin, CBin, Tin = u
    
    dz = p["dz"]
    v = p["v"]
    D = p["D"]
    
    DT = D
    beta = p["beta"]
    beta = 560.0 / (1.0 * 4.186) # Heat of reaction
    Ea_R = 8500.0 
    k0 = np.exp(24.6)   # Arrhenius constant 
    def k(T):
        return k0 * np.exp(-Ea_R / T)
    
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

    def g(t, x):
        return np.array([Tin*sigma_T])




    def Jacobian(t, x, esdirk = False):
        n = len(x)
        J = np.zeros((n, n))

        CA = CAin + (1 / beta) * (Tin - x)
        CB = CBin + (2 / beta) * (Tin - x)
        kT = k(x)
        dk_dT = kT * Ea_R / x**2

        # Reaction term derivative (diagonal)
        dR_dT = beta * (
            dk_dT * CA * CB - kT * CB - 2 * kT * CA
        )

        # Fill diagonal with reaction + central spatial terms
        for i in range(n):
            J[i, i] = dR_dT[i] - 2 * DT / dz**2 - v / dz

        # Fill spatial coupling terms
        for i in range(1, n):
            J[i, i - 1] = DT / dz**2 + v / dz
        for i in range(n - 1):
            J[i, i + 1] = DT / dz**2

        if esdirk:
            return J, np.eye(1)
        else:
            return J

    return f, g, Jacobian