import numpy as np

# Function for the prey-predator model
def prey_predator_model(alpha, beta, esdirk=False, SDE=False, sigma_x=0.1, sigma_y=0.1):
    def f(t, z):
        x, y = z
        dxdt = alpha * x - beta * x * y
        dydt = beta * x * y - alpha * y
        if esdirk:
            return np.array([dxdt, dydt]), z
        else:
            return np.array([dxdt, dydt])
    
    def g(t, z):
        x, y = z
        return np.diag([sigma_x * x, sigma_y * y])  # Multiplicative noise
    
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
    
    if SDE:
        return f, g, jacobian
    else:
        return f, jacobian

# Function for the Van der Pol model
def van_der_pol_model(mu, esdirk=False, SDE=False, sigma_x=0.1, sigma_y=0.1):
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
        return np.diag([sigma_x * (1 + x**2), sigma_y * y])  # State-dependent noise
    
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
    else:
        return f, jacobian

# Functions for Chemical Reaction in adiabatic reactors
def CSTR_3state_model(params, esdirk=False, SDE=False, sigma_CA=0.01, sigma_CB=0.01, sigma_T=0.1):
    # Parameters
    p = 1.0             # Density
    cp = 4.186          # Specific heat capacity
    k0 = np.exp(24.6)   # Arrhenius constant
    Ea_R = 8500.0       # Activation energy   
    deltaHr = -56000.0  # Reaction enthalphy
    beta = - deltaHr / (p * cp) 

    # Unpack parameters
    F, V, CAin, CBin, Tin = params

    def k(T):
        return k0 * np.exp(-Ea_R / T)

    def f(t, x):
        CA, CB, T = x
        r = k(T) * CA * CB
        dCA_dt = (F/V) * (CAin-CA) - r
        dCB_dt = (F/V) * (CBin - CB) - 2*r
        dT_dt = (F/V) * (Tin - T) + beta * r
        if esdirk:
            return np.array([dCA_dt, dCB_dt, dT_dt]), x
        else:
            return np.array([dCA_dt, dCB_dt, dT_dt])
    
    def g(t, x):
        CA, CB, T = x
        return np.diag([sigma_CA * CA, sigma_CB * CB, sigma_T * T])
    
    def jacobian(t, x):
        CA, CB, T = x
        J = np.array([
            [-F/V - k0*np.exp(-Ea_R/T)*CB, -k0*np.exp(-Ea_R/T)*CA, -(Ea_R/T**2)*(k0*np.exp(-Ea_R/T)*CA*CB)],
            [-2*k0*np.exp(-Ea_R/T)*CB, -F/V-2*k0*np.exp(-Ea_R/T)*CA, -(Ea_R/T**2)*(2*k0*np.exp(-Ea_R/T)*CA*CB)],
            [beta*k0*np.exp(-Ea_R/T)*CB, beta*k0*np.exp(-Ea_R/T)*CA, -F/V-(Ea_R/T**2)*(beta*k0*np.exp(-Ea_R/T)*CA*CB)]
        ])
        if esdirk:
            return J, np.eye(3)
        else:
            return J
    
    if SDE:
        return f, g, jacobian
    else:
        return f, jacobian

# Function for the CSTR (1 state Model)
def CSTR_1state_model(params, esdirk=False, SDE=False, sigma_T=0.1):
    # Parameters
    p = 1.0
    cp = 4.186
    k0 = np.exp(24.6)
    Ea_R = 8500.0
    deltaHr = -56000.0
    beta = - deltaHr / (p * cp)
    
    F, V, CA_in, CB_in, Tin = params

    def f(t, T):
        CA = CA_in + (1 / beta) * (Tin - T)
        CB = CB_in + (2 / beta) * (Tin - T)
        r = k0 * np.exp(-Ea_R / T) * CA * CB
        if esdirk:
            return F / V * (Tin - T) + beta * r, T
        else:
            return F / V * (Tin - T) + beta * r
    
    def g(t, T):
        return np.array([sigma_T * T])
    
    def jacobian(t, T):
        CA = CA_in + (1 / beta) * (Tin - T)
        CB = CB_in + (2 / beta) * (Tin - T)
        J = np.array([-F/V + beta*(k0*np.exp(-Ea_R/T)*(Ea_R/T**2)*CA*CB + 
                       k0*np.exp(-Ea_R/T)*CB*(-1/beta) + 
                       k0*np.exp(-Ea_R/T)*CA*(-2/beta))])
        if esdirk:
            return J, np.eye(1)
        else:
            return J
    
    if SDE:
        return f, g, jacobian
    else:
        return f, jacobian

# Wiener process (unchanged)
def StdWeinerProcess(T, N, nW, Ns, seed=False):
    if seed:
        np.random.seed(seed)
    dt = T/N
    dW = np.sqrt(dt) * np.random.randn(nW, N, Ns)
    W = np.concatenate((np.zeros((nW, 1, Ns)), np.cumsum(dW, axis=1)), axis=1)
    Tw = np.linspace(0, T, N+1)
    return W, Tw, dW