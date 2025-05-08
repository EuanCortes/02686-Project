import numpy as np

# Function for the prey-predator model
def prey_predator_model(alpha, beta, esdirk=False, SDE=False, SDE2 = False, sigma = 0.05):
    def f(t, z):
        x, y = z
        dxdt = alpha * (1-y)*x
        dydt = - beta * (1-x)*y
        if esdirk:
            return np.array([dxdt, dydt]), z
        else:
            return np.array([dxdt, dydt])
    
    def g(t, z):
        x, y = z
        return np.diag([0.0, sigma])
    
    def g2(t, z):
        x, y = z
        return np.diag([sigma * (1.0 + y**2), 0.0])
    
    def jacobian(t, z):
        x, y = z
        df_dx = alpha * (1-y)
        df_dy = - alpha * x
        dg_dx = beta * y
        dg_dy = -beta * (1 - x)
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


# Functions for Chemical Reaction in adiabatic reactors
def CSTR_3state_model(params, esdirk=False, SDE=False, sigma_T=0.5):
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
        return np.diag([0, 0, sigma_T * (F/V)])
    
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
def CSTR_1state_model(params, esdirk=False, SDE=False, sigma_T=0.5):
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
        return np.array([sigma_T * (F/V)])
    
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


import numpy as np

def PFR_3state_model(params, esdirk=False, SDE = False, sigma_T = 0.1):
    # Unpack parameters
    dz = params["dz"]
    v = params["v"]
    DA, DB, DT = params["D"]
    beta = params["beta"]
    k = params["k"]
    CAin, CBin, Tin = params["u"]
    Ea_R = params["Ea_R"]
    
    def f(t, x):
        """ODE function for PFR system"""
        n = len(x) // 3
        CA = x[:n]
        CB = x[n:2*n]
        T = x[2*n:]
        
        # Initialize derivatives
        dCA_dt = np.zeros(n)
        dCB_dt = np.zeros(n)
        dT_dt = np.zeros(n)
        
        # Reaction term
        r = k(T) * CA * CB

        ### Convection and diffusion terms for CA
        # Convection term
        NconvA = np.zeros(n+1)
        NconvA[0] = v * CAin        # Inlet Boundary (Dirichlet)
        NconvA[1:n+1] = v * CA[0:n]

        # Diffusion term
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
        # Convection term
        NconvB = np.zeros(n+1)
        NconvB[1:n+1] = v * CB[0:n]
        NconvB[0] = v * CBin        # Inlet Boundary (Dirichlet)

        # Diffusion term
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
        # Convection term
        NconvT = np.zeros(n+1)
        NconvT[1:n+1] = v * T[0:n]
        NconvT[0] = v * Tin         # Inlet Boundary (Dirichlet)

        # Diffusion term
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

        # Combine derivatives
        xdot = np.concatenate([dCA_dt, dCB_dt, dT_dt])
        
        if esdirk:
            return xdot, x
        return xdot
    

    def g(t, x):
        n = len(x) // 3
        T = x[2*n:]  # Only temperature has noise
        
        # Create diagonal matrix with noise only on temperature components
        G = np.zeros(3*n)
        G[2*n:] = sigma_T * T  # Multiplicative noise on temperature
        
        return np.diag(G)  # Return as diagonal matrix

    def jacobian(t, x):
        """Jacobian function for PFR system"""
        n = len(x) // 3
        CA = x[:n]
        CB = x[n:2*n]
        T = x[2*n:]
        
        # Initialize block-diagonal Jacobian structure
        J = np.zeros((3*n, 3*n))
        
        # Reaction rate and its derivatives
        r = k(T) * CA * CB
        dr_dCA = k(T) * CB
        dr_dCB = k(T) * CA
        dr_dT = k(T) * CA * CB * (Ea_R / T**2)
        
        # Fill Jacobian blocks
        for i in range(n):
            # CA equation
            J[i, i] = -v/dz - DA/dz**2 - dr_dCA[i]
            if i > 0:
                J[i, i-1] = DA/dz**2
            J[i, n+i] = -dr_dCB[i]
            J[i, 2*n+i] = -dr_dT[i]
            
            # CB equation
            J[n+i, i] = -2*dr_dCA[i]
            J[n+i, n+i] = -v/dz - DB/dz**2 - 2*dr_dCB[i]
            if i > 0:
                J[n+i, n+i-1] = DB/dz**2
            J[n+i, 2*n+i] = -2*dr_dT[i]
            
            # T equation
            J[2*n+i, i] = beta*dr_dCA[i]
            J[2*n+i, n+i] = beta*dr_dCB[i]
            J[2*n+i, 2*n+i] = -v/dz - DT/dz**2 + beta*dr_dT[i]
            if i > 0:
                J[2*n+i, 2*n+i-1] = DT/dz**2
        
        if esdirk:
            return J, np.eye(3*n)
        return J
    
    if SDE:
        return f, g, jacobian
    else:
        return f, jacobian

import numpy as np

def PFR_1state_model(params, esdirk=False, SDE=False, sigma_T=0.1):
    # Unpack parameters
    dz = params["dz"]
    v = params["v"]
    DT = params["D"]
    beta = params["beta"]
    k = params["k"]
    CAin, CBin, Tin = params["u"]
    Ea_R = params["Ea_R"]
    
    def f(t, x):
        """ODE function for 1-state PFR system"""
        n = len(x)
        dT_dt = np.zeros(n)
        
        # Calculate CA and CB from temperature (assuming equilibrium)
        CA = CAin + (1 / beta) * (Tin - x)
        CB = CBin + (2 / beta) * (Tin - x)
        
        # Reaction term
        r = k(x) * CA * CB
        
        # Convection term
        NconvT = np.zeros(n+1)
        NconvT[0] = v * Tin         # Inlet Boundary (Dirichlet)
        NconvT[1:n+1] = v * x[0:n]
        
        # Diffusion term
        JT = np.zeros(n+1)
        JT[1:n] = (-DT / dz) * (x[1:n] - x[0:n-1])
        JT[0] = 0                   # Inlet Boundary (Dirichlet)
        JT[-1] = 0                  # Outlet Boundary (Neumann)
        
        # Total flux and reaction term
        NT = NconvT + JT
        rT = beta * r
        
        # Differential Equation for T
        dT_dt = (NT[1:n+1] - NT[0:n]) / (-dz) + rT
        
        if esdirk:
            return dT_dt, x
        return dT_dt
    
    def g(t, T):
        return np.array([sigma_T * CAin])

    def jacobian(t, x):
        """Jacobian function for 1-state PFR system"""
        n = len(x)
        
        # Calculate CA and CB from temperature
        CA = CAin + (1 / beta) * (Tin - x)
        CB = CBin + (2 / beta) * (Tin - x)
        
        # Initialize Jacobian matrix
        J = np.zeros((n, n))
        
        # Main diagonal terms (including reaction derivatives)
        main_diag = (-v/dz - DT/dz**2) + beta * k(x) * (
            (Ea_R/x**2)*CA*CB - (1/beta)*CB - (2/beta)*CA
        )
        
        # Lower diagonal (diffusion terms)
        lower_diag = DT/dz**2
        
        # Upper diagonal (would be convection terms, but upwind scheme makes them zero)
        
        # Fill the Jacobian matrix
        for i in range(n):
            J[i, i] = main_diag[i]
            if i > 0:
                J[i, i-1] = lower_diag
        
        if esdirk:
            return J, np.eye(n)
        return J
    
    if SDE:
        return f, g, jacobian
    else:
        return f, jacobian