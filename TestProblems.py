import numpy as np
import matplotlib.pyplot as plt

# Function for the prey-predator model with inputs (t,x,params)

def prey_predator(t, x, params):
    """
    Implementation of the Prey-Predator model (Lotka-Volterra equations)

    Parameters:
    t : float
        Time variable
    x : array-like
        State variables [prey, predator]
    params : array-like
        Model parameters [alpha, beta, delta, gamma]
        alpha : Prey birth rate
        beta : Prey death rate due to predation
        delta : Predator birth rate due to predation
        gamma : Predator death rate
    
    Returns:
    zdot : array-like
        Derivatives [d(prey)/dt, d(predator)/dt]
    """
    # Unpack parameters
    alpha, beta, delta, gamma = params

    # Unpack state variables
    prey, predator = x

    # Lotka-Volterra equations
    dprey_dt = alpha * prey - beta * prey * predator
    dpredator_dt = delta * prey * predator - gamma * predator
    
    return [dprey_dt, dpredator_dt]

# Function for the Prey-Predator Jacobian with inputs (t,x,params)
def prey_predator_jacobian(t, x, params):
    """
    Jacobian of the Prey-Predator model

    Parameters:
    t : float
        Time variable
    x : array-like
        State variables [prey, predator]
    params : array-like
        Model parameters [alpha, beta, delta, gamma]

    Returns:
    J : array-like
        Jacobian matrix
    """
    # Unpack parameters
    alpha, beta, delta, gamma = params

    # Unpack state variables
    prey, predator = x

    # Jacobian matrix
    J = np.array([[alpha - beta * predator, -beta * prey],
                  [delta * predator, delta * prey - gamma]])
    
    return J

# Function for the Van der Pol Model
def VanDerPol(t, x, params):
    """
    Implementation of the Van der Pol model

    Parameters:
    t : float
        Time variable
    x : array-like
        State variables [x1, x2]
    params : array-like
        Model parameters [mu]
        mu : Nonlinearity parameter of the Van der Pol oscillator
 
    Returns:
    xdot : ndarray
        Derivatives [dx1/dt, dx2/dt]
    """
    # Unpack parameters
    mu = params[0]
    # Unpack state variables
    x1, x2 = x
    # Van der Pol equations
    dx1_dt = x2
    dx2_dt = mu * (1 - x1**2) * x2 - x1

    # Return derivatives
    return np.array([dx1_dt, dx2_dt])

# Function for the Van der Pol Jacobian
def VanDerPol_jacobian(t, x, params):
    """
    Jacobian of the Van der Pol model

    Parameters:
    t : float
        Time variable
    x : array-like
        State variables [x1, x2]
    params : array-like
        Model parameters [mu]

    Returns:
    J : array-like
        Jacobian matrix
    """
    # Unpack parameters
    mu = params[0]
    
    # Unpack state variables
    x1, x2 = x

    # Jacobian matrix
    J = np.array([[ 0, 1],
                  [ -2*x1*x2 - 1, mu * (1 - x1**2)]])
    
    return J

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



def CSTR(t, CA, CB, T, params):
    """
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
    """
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
    """
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
    """
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
