import numpy as np
from scipy.linalg import lu_factor, lu_solve

###############################
##### Explicit ODE solver #####
###############################

# Explicit Euler with fixed step size
def ExplicitEulerFixedSteps(fun, t0, tN, N, x0, *args):
    """
    Explicit Euler method with fixed step size.
    Parameters
    ----------
    fun : function
        Function to compute the right-hand side of the ODE.
    t0 : float
        Initial time.
    tN : float
        Final time.
    N : int
        Number of steps.
    x0 : array_like
        Initial condition.
    *args : tuple
        Additional arguments to pass to the function.
    Returns
    -------
    T : array_like
        Time points.
    X : array_like
        Solution at each time point.
    """
    # Compute step size and allocate memory
    dt = (tN - t0) / N
    nx = x0.shape[0]
    X = np.zeros((nx, N + 1))
    T = np.zeros(N + 1)

    # Euler's Explicit Method
    T[0] = t0
    X[:, 0] = x0
    for k in range(1,N+1):
        T[k] = T[k-1] + dt
        X[:,k] = X[:,k-1] + dt*fun(T[k-1], X[:,k-1], *args)

    return T, X

# Explicit Euler with adaptive step size
def ExplicitEulerAdaptiveStep(f, tspan, x0, h0, abstol, reltol, *args):
    """
    Explicit Euler method with adaptive step size.
    Parameters
    ----------
    f : function
        Function to compute the right-hand side of the ODE.
    tspan : tuple
        Tuple containing the initial and final time.
    x0 : array_like
        Initial condition.
    h0 : float
        Initial step size.
    abstol : float
        Absolute tolerance.
    reltol : float
        Relative tolerance.
    *args : tuple
        Additional arguments to pass to the function.
    
    Returns
    -------
    T : array_like
        Time points.
    X : array_like
        Solution at each time point.
    H : array_like
        Step sizes used at each time point.
    """
    # Initialize variables
    t0 = tspan[0]
    tf = tspan[1]
    # Initial conditions
    t = t0
    x = x0
    h = h0

    hmin = 0.1
    hmax = 5
    epstol = 0.8

    # Initialize output arrays
    T = [t0]
    X = [x0]
    H = [h0]

    # Main loop
    while t < tf:
        # Adjust step size
        if t + h > tf:
            h = tf - t

        fun = np.array(f(t, x))

        AcceptStep = False
        while not AcceptStep:
            # Compute the next step
            xnew = x + h * fun

            hm = 0.5 * h
            tm = t + hm
            xm = x + hm * fun
            xnewm = xm + hm * f(tm, xm)

            # Compute the error
            err = np.abs(xnewm - xnew)
            max1 = np.maximum(abstol, np.abs(xnewm) * reltol)
            r = np.max(err / max1)
            AcceptStep = (r <= 1)

            # Check if error is within tolerance
            if AcceptStep:
                # Update time and state
                t = t + h
                x = xnewm
                # Store values
                T.append(t)
                X.append(x)

            # Update step size
            h = np.max([hmin, np.min([hmax, np.sqrt(epstol / r)])]) * h
            H.append(h)

    return np.array(T), np.array(X), np.array(H)



###############################
##### Implicit ODE solver #####
###############################

# Newton's method for implicit Euler
def NewtonsMethodODE(f, jac, t, x, dt, xinit, tol, maxit, *args):
    x_new = xinit.copy()
    for _ in range(maxit):
        # Calculate function and Jacobian at new time step
        F = x_new - x - dt * f(t + dt, x_new, *args)
        J = np.eye(len(x)) - dt * jac(t + dt, x_new, *args)
        
        # Newton update
        dx = np.linalg.solve(J, -F)
        x_new += dx
        
        if np.linalg.norm(dx) < tol:
            break
    return x_new

def ImplicitEulerFixedStep(f, jac, ta, tb, N, xa, *args):
    # Compute step size and allocate memory
    dt = (tb - ta) / N
    nx = len(xa)
    X = np.zeros((N + 1, nx))  # Changed to (N+1, nx) for better indexing
    T = np.zeros(N + 1)
    
    # Solver parameters
    tol = 1.0e-8
    maxit = 100
    
    # Initial conditions
    T[0] = ta
    X[0] = xa.copy()
    
    # Time stepping
    for k in range(N):
        T[k + 1] = T[k] + dt
        # Initial guess (explicit Euler step)
        xinit = X[k] + dt * f(T[k], X[k], *args)
        # Solve implicit step using Newton's method
        X[k + 1] = NewtonsMethodODE(f, jac, T[k], X[k], dt, xinit, tol, maxit, *args)
    
    return T, X.T  # Return transposed X to match original shape (nx, N+1)

# Implicit Euler with adaptive step size
import numpy as np

def ImplicitEulerAdaptiveStep(f, jac, tspan, x0, h0, abstol, reltol, maxit=100, *args):
    """
    Implicit Euler method with adaptive step size control
    Args:
        f: function(t, x, *args) returning f(x,t)
        jac: function(t, x, *args) returning Jacobian matrix
        tspan: tuple (t0, tf)
        x0: initial condition
        h0: initial step size
        abstol: absolute tolerance
        reltol: relative tolerance
        maxit: maximum Newton iterations
        *args: additional arguments for f and jac
    """
    # Initialize variables
    t0, tf = tspan
    t = t0
    x = x0.copy()
    h = h0

    # Step size bounds
    hmin = 0.01
    hmax = 5.0
    epstol = 0.8  # Safety factor

    # Output storage
    T = [t0]
    X = [x0]
    H = [h0]

    while t < tf:
        # Adjust step size to not exceed tf
        if t + h > tf:
            h = tf - t

        AcceptStep = False
        while not AcceptStep:
            # Compute candidate steps
            # Full step
            xinit = x + h * f(t, x, *args)  # Explicit Euler as initial guess
            xnew = NewtonsMethodODE(f, jac, t, x, h, xinit, abstol, maxit, *args)

            # Half steps
            hm = 0.5 * h
            tm = t + hm
            xinit_m = x + hm * f(t, x, *args)
            xm = NewtonsMethodODE(f, jac, t, x, hm, xinit_m, abstol, maxit, *args)
            xnewm = NewtonsMethodODE(f, jac, tm, xm, hm, xm, abstol, maxit, *args)

            # Compute the error
            err = np.abs(xnewm - xnew)
            max1 = np.maximum(abstol, np.abs(xnewm) * reltol)
            r = np.max(err / max1)
            AcceptStep = (r <= 1)

            # Check if error is within tolerance
            if AcceptStep:
                # Update time and state
                t = t + h
                x = xnewm
                # Store values
                T.append(t)
                X.append(x)

            # Update step size
            h = np.max([hmin, np.min([hmax, np.sqrt(epstol / r)])]) * h
            H.append(h)

    return np.array(T), np.array(X), np.array(H)


###############################
###### Solvers for SDEs #######
###############################

# Wiener process
def StdWeinerProcess(T, N, nW, Ns, seed = False):
    if seed:
        np.random.seed(seed)
    dt = T/N
    dW = np.sqrt(dt) * np.random.randn(nW, N, Ns)
    W = np.concatenate((np.zeros((nW, 1, Ns)), np.cumsum(dW, axis=1)), axis=1)
    Tw = np.linspace(0, T, N+1)
    return W, Tw, dW


# Implement Euler-Maruyama SDE solver
def SDEsolverExplicitExplicit(f, g, t, x0, W, p=None):
    n = len(t)
    d = len(x0)
    X = np.zeros((d, n))
    X[:, 0] = x0
    
    for i in range(n-1):
        dt = t[i+1] - t[i]
        dW = W[:, i+1] - W[:, i]
        X[:, i+1] = X[:, i] + f(t[i], X[:, i]) * dt + g(t[i], X[:, i]) @ dW
    
    return X

# Newton's method for SDE
def SDENewtonSolver(f, jac, t, dt, psi, xinit, tol, maxit, *args):
    I = np.eye(len(xinit))
    x = xinit
    fx = f(t, x, *args)
    J = jac(t, x, *args)
    R = x - fx * dt - psi
    
    it = 1
    while (np.linalg.norm(R, np.inf) > tol) and (it <= maxit):
        dRdx = I - J * dt
        mdx = np.linalg.solve(dRdx, R)
        x = x - mdx
        fx = f(t, x, *args)
        J = jac(t, x, *args)
        R = x - fx * dt - psi
        it += 1
    
    return x, fx, J

# Implicit-explicit method for SDE with fixed step size
def SDEsolverImplicitExplicit(f, jac, gfun, T, x0, W, *args):
    tol = 1.0e-8
    maxit = 100
    N = len(T)
    nx = len(x0)
    X = np.zeros((nx, N))
    X[:, 0] = x0

    for k in range(N-1):
        # Explicit part (diffusion)
        g = gfun(T[k], X[:, k], *args)
        dt = T[k+1] - T[k]
        dW = W[:, k+1] - W[:, k]
        psi = X[:, k] + g @ dW  # Matrix multiplication for multi-dimensional case
        
        # Initial guess for implicit step
        xinit = psi + f(T[k], X[:, k], *args) * dt
        
        # Implicit step (drift)
        X[:, k+1], _, _ = SDENewtonSolver(
            f, jac, T[k+1], dt, psi, xinit, tol, maxit, *args
        )

    return X

###############################
######## Runge-Kutta ##########
###############################

# Runge-Kutta with fixed step size
def ExplicitRungeKuttaSolver(f, tspan, x0, h, solver, *args):
    #Solver is a dict

    #Solver parameters
    s = solver["stages"]
    AT = solver["AT"]
    b = solver["b"]
    c = solver["c"]

    #Parameters related to constant step size
    hAT = h*AT
    hb = h*b
    hc = h*c

    #Size parameters
    x = x0
    t = tspan[0]
    tf = tspan[1]
    N = int((tf-t)/h)
    nx = len(x0)

    #Allocate memory
    T = np.zeros((s))
    X = np.zeros((nx,s))
    F = np.zeros((nx,s))

    Tout = np.zeros((N+1))
    Xout = np.zeros((nx, N+1))

    Tout[0] = t
    Xout[:,0] = x

    for i in range(N):
        #Compute stages
        #Stage 1
        T[0] = t
        X[:,0] = x
        F[:,0] = f(T[0],X[:,0],*args)

        #Stage 2, 3, ..., s
        for j in range(1,s):
            T[j] = t + hc[j]
            X[:,j] = x + np.dot(F[:,:j-1],hAT[:j-1,j])
            F[:,j] = f(T[j],X[:,j],*args)

        #Update state
        x = x + np.dot(F,hb)

        #Update time
        t = t + h

        #Store solution
        Tout[i+1] = t
        Xout[:,i+1] = x.T

    return Tout, Xout


def ExplicitRungeKuttaSolverAdaptive(f, tspan, x0, h0, solver, abstol, reltol, *args):
    hmin = 1e-6  # More reasonable minimum step size
    hmax = 5
    epstol = 0.8
    maxiter = 100000  # Safety net to prevent infinite loops

    # Solver parameters
    s = solver["stages"]
    AT = solver["AT"]
    b = solver["b"]
    c = solver["c"]

    # Initial conditions
    x = np.array(x0)
    t = tspan[0]
    tf = tspan[1]
    nx = len(x0)

    # Allocate memory
    T = np.zeros(s)
    X = np.zeros((nx, s))
    F = np.zeros((nx, s))
    Tm = np.zeros(s)
    Xm = np.zeros((nx, s))
    Fm = np.zeros((nx, s))
    
    # Output storage
    Tout = [t]
    Xout = [x.copy()]
    H = [h0]

    h = h0
    iterations = 0

    while t < tf and iterations < maxiter:
        iterations += 1
        
        # Adjust step size to not exceed tf
        if t + h > tf:
            h = tf - t

        # Compute stages for full step and half steps
        # Full step stages
        T[0] = t
        X[:, 0] = x
        F[:, 0] = f(T[0], X[:, 0], *args)
        
        # Half step stages
        hm = 0.5 * h
        tm = t + hm
        Tm[0] = tm
        Xm[:, 0] = x
        Fm[:, 0] = f(Tm[0], Xm[:, 0], *args)

        for j in range(1, s):
            # Full step stages
            T[j] = t + h * c[j]
            X[:, j] = x + np.dot(F[:, :j], h * AT[:j, j])
            F[:, j] = f(T[j], X[:, j], *args)
            
            # Half step stages
            Tm[j] = tm + hm * c[j]
            Xm[:, j] = x + np.dot(Fm[:, :j], hm * AT[:j, j])
            Fm[:, j] = f(Tm[j], Xm[:, j], *args)

        # Compute full step and two half steps
        xnew = x + np.dot(F, h * b)
        
        # First half step
        xm = x + np.dot(Fm, hm * b)
        Xm[:, 0] = xm
        Fm[:, 0] = f(tm, Xm[:, 0], *args)
        
        # Second half step stages
        for j in range(1, s):
            Xm[:, j] = xm + np.dot(Fm[:, :j], hm * AT[:j, j])
            Fm[:, j] = f(Tm[j], Xm[:, j], *args)
        
        xnewm = xm + np.dot(Fm, hm * b)

        # Error estimation
        x_err = np.linalg.norm(xnew - xnewm)
        scale = np.maximum(abstol, np.abs(xnew) * reltol)
        r = np.max(x_err / scale)

        # Step size adjustment
        if r <= 1.0:  # Step accepted
            t += h
            x = xnew
            Tout.append(t)
            Xout.append(x.copy())
            H.append(h)
            # Increase step size for next step
            h = min(hmax, max(hmin, 0.9 * h * (epstol / r)**0.2))
        else:  # Step rejected
            # Decrease step size and try again
            h = max(hmin, min(hmax, 0.9 * h * (epstol / r)**0.25))

    if iterations >= maxiter:
        print("Warning: Maximum iterations reached!")
        
    return np.array(Tout), np.array(Xout), np.array(H)


def rk4():
    return {
        "stages": 4,
        "AT": np.array([[0,   0,   0, 0],
                        [0.5, 0,   0, 0],
                        [0,   0.5, 0, 0],
                        [0,   0,   1, 0]]).T,
        "b": np.array([1/6, 1/3, 1/3, 1/6]),
        "c": np.array([0, 0.5, 0.5, 1])
    }



###############################
##### Dormand-Prince 5(4) #####
###############################

def dormand_prince_45():
    return {
        "stages": 7,
        "AT": np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [1/5, 0, 0, 0, 0, 0, 0],
            [3/40, 9/40, 0, 0, 0, 0, 0],
            [44/45, -56/15, 32/9, 0, 0, 0, 0],
            [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
        ]).T,
        "b": np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]),
        "c": np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
    }

###############################
########## ESDIRK23 ###########
#################

def ESDIRK(fun, jac, t_span, x0, h0, absTol, relTol, Method, *args):
    # ESDIRK23 Parameters 
    #=========================================================================
    # Runge-Kutta method parameters
    if Method == 'ESDIRK12':
        gamma = 1
        AT = np.array([[0, 0], [0, gamma]])
        c = np.array([0, 1])
        b = AT[:, 1]
        bhat = np.array([1/2, 1/2])
        d = b - bhat
        p = 1
        phat = 2
        s = 2
    elif Method == 'ESDIRK23':
        gamma = 1 - 1/np.sqrt(2)
        a31 = (1 - gamma)/2
        AT = np.array([[0, gamma, a31], 
                       [0, gamma, a31], 
                       [0, 0, gamma]])
        c = np.array([0, 2*gamma, 1])
        b = AT[:, 2]
        bhat = np.array([(6*gamma - 1)/(12*gamma),
                         1/(12*gamma*(1 - 2*gamma)),
                         (1 - 3*gamma)/(3*(1 - 2*gamma))])
        d = b - bhat
        p = 2
        phat = 3
        s = 3
    elif Method == 'ESDIRK34':
        gamma = 0.43586652150845899942
        a31 = 0.14073777472470619619
        a32 = -0.1083655513813208000
        AT = np.array([[0, gamma, a31, 0.10239940061991099768],
                       [0, gamma, a32, -0.3768784522555561061],
                       [0, 0, gamma, 0.83861253012718610911],
                       [0, 0, 0, gamma]])
        c = np.array([0, 0.87173304301691799883, 0.46823874485184439565, 1])
        b = AT[:, 3]
        bhat = np.array([0.15702489786032493710,
                         0.11733044137043884870,
                         0.61667803039212146434,
                         0.10896663037711474985])
        d = b - bhat
        p = 3
        phat = 4
        s = 4
    else:
        raise ValueError(f"Unknown method: {Method}")

    # error and convergence controller
    epsilon = 0.8
    tau = 0.1 * epsilon
    itermax = 20
    ke0 = 1.0 / phat
    ke1 = 1.0 / phat
    ke2 = 1.0 / phat
    alpharef = 0.3
    alphaJac = -0.2
    alphaLU = -0.2
    hrmin = 0.01
    hrmax = 10
    
    # Initialize info dictionary
    info = {
        'Method': Method,
        'nStage': s,
        'absTol': absTol,
        'relTol': relTol,
        'iterMax': itermax,
        'tspan': t_span,
        'nFun': 0,
        'nJac': 0,
        'nLU': 0,
        'nBack': 0,
        'nStep': 0,
        'nAccept': 0,
        'nFail': 0,
        'nDiverge': 0,
        'nSlowConv': 0
    }
    
    # Initialize stats dictionary
    stats = {
        't': [],
        'h': [],
        'r': [],
        'iter': [],
        'Converged': [],
        'Diverged': [],
        'AcceptStep': [],
        'SlowConv': []
    }

    # Main ESDIRK Integrator
    nx = len(x0)
    F = np.zeros((nx, s))
    t = t_span[0]
    tf = t_span[1]
    x = x0.copy()
    h = h0

    # Evaluate initial function and Jacobian
    F[:, 0], g = fun(t, x, *args)
    info['nFun'] += 1
    dfdx, dgdx = jac(t, x, *args)
    info['nJac'] += 1
    FreshJacobian = True
    
    if (t + h) > tf:
        h = tf - t
    
    hgamma = h * gamma
    dRdx = dgdx - hgamma * dfdx
    LU = lu_factor(dRdx)
    info['nLU'] += 1
    hLU = h

    FirstStep = True
    ConvergenceRestriction = False
    PreviousReject = False
    iter_counts = np.zeros(s)

    # Output storage
    chunk = 100
    Tout = np.zeros((chunk, 1))
    Xout = np.zeros((chunk, nx))
    Gout = np.zeros((chunk, nx))

    Tout[0, 0] = t
    Xout[0, :] = x
    Gout[0, :] = g

    while t < tf:
        info['nStep'] += 1
        i = 1
        diverging = False
        SlowConvergence = False
        alpha = 0.0
        Converged = True

        # A step in the ESDIRK method
        while (i < s) and Converged:
            i += 1
            phi = g + F[:, :i-1] @ (h * AT[:i-1, i-1])

            # Initial guess for the state
            if i == 2:
                dt = c[i-1] * h
                G = g + dt * F[:, 0]
                X = x + np.linalg.solve(dgdx, (G - g).reshape(-1, 1)).flatten()
            else:
                dt = c[i-1] * h
                G = g + dt * F[:, 0]
                X = x + np.linalg.solve(dgdx, (G - g).reshape(-1, 1)).flatten()

            T = t + dt

            F[:, i-1], G = fun(T, X, *args)
            info['nFun'] += 1
            R = G - hgamma * F[:, i-1] - phi
            rNewton = np.linalg.norm(R / (absTol + np.abs(G) * relTol), np.inf)
            Converged = (rNewton < tau)
            
            # Newton Iterations
            while not Converged and not diverging and not SlowConvergence:
                iter_counts[i-1] += 1
                dX = lu_solve(LU, R)
                info['nBack'] += 1
                X = X - dX
                rNewtonOld = rNewton
                
                F[:, i-1], G = fun(T, X, *args)
                info['nFun'] += 1
                R = G - hgamma * F[:, i-1] - phi
                rNewton = np.linalg.norm(R / (absTol + np.abs(G) * relTol), np.inf)
                alpha = max(alpha, rNewton / rNewtonOld)
                Converged = (rNewton < tau)
                diverging = (alpha >= 1)
                SlowConvergence = (iter_counts[i-1] >= itermax)

        diverging = (alpha >= 1) * i  # Record which stage is diverging

        # Store stats for this step
        stats['t'].append(t)
        stats['h'].append(h)
        stats['r'].append(np.nan)
        stats['iter'].append(iter_counts.copy())
        stats['Converged'].append(Converged)
        stats['Diverged'].append(diverging)
        stats['AcceptStep'].append(False)
        stats['SlowConv'].append(SlowConvergence * i)
        
        iter_counts[:] = 0  # Reset iteration counts

        # Error and Convergence Controller
        if Converged and alpha > 0:
            # Error estimation
            e = F @ (h * d)
            r = np.linalg.norm(e / (absTol + np.abs(G) * relTol), np.inf)
            CurrentStepAccept = (r <= 1.0)
            r = max(r, np.finfo(float).eps)
            stats['r'][-1] = r
            
            # Step Length Controller
            if CurrentStepAccept:
                stats['AcceptStep'][-1] = True
                info['nAccept'] += 1
                
                if FirstStep or PreviousReject or ConvergenceRestriction:
                    # Asymptotic step length controller
                    hr = 0.75 * (epsilon / r) ** ke0
                else:
                    # Predictive controller
                    s0 = h / hacc
                    s1 = max(hrmin, min(hrmax, (racc / r) ** ke1))
                    s2 = max(hrmin, min(hrmax, (epsilon / r) ** ke2))
                    hr = 0.95 * s0 * s1 * s2
                
                racc = r
                hacc = h
                FirstStep = False
                PreviousReject = False
                ConvergenceRestriction = False
                
                # Next Step
                t = T
                x = X
                g = G
                F[:, 0] = F[:, s-1]
            else:  # Reject current step
                info['nFail'] += 1
                if PreviousReject:
                    kest = np.log(r / rrej) / np.log(h / hrej)
                    kest = min(max(0.1, kest), phat)
                    hr = max(hrmin, min(hrmax, (epsilon / r) ** (1 / kest)))
                else:
                    hr = max(hrmin, min(hrmax, (epsilon / r) ** ke0))
                rrej = r
                hrej = h
                PreviousReject = True
            
            # Convergence control
            halpha = alpharef / alpha
            if alpha > alpharef:
                ConvergenceRestriction = True
                if hr < halpha:
                    h = max(hrmin, min(hrmax, hr)) * h
                else:
                    h = max(hrmin, min(hrmax, halpha)) * h
            else:
                h = max(hrmin, min(hrmax, hr)) * h
            
            h = max(1e-8, h)
            if (t + h) > tf:
                h = tf - t
    
            # Jacobian Update Strategy
            FreshJacobian = False
            if alpha > alphaJac:
                dfdx, dgdx = jac(t, x, *args)
                info['nJac'] += 1
                FreshJacobian = True
                hgamma = h * gamma
                dRdx = dgdx - hgamma * dfdx
                LU = lu_factor(dRdx)
                info['nLU'] += 1
                hLU = h
            elif (abs(h - hLU) / hLU) > alphaLU:
                hgamma = h * gamma
                dRdx = dgdx - hgamma * dfdx
                LU = lu_factor(dRdx)
                info['nLU'] += 1
                hLU = h
        else:
            info['nFail'] += 1
            CurrentStepAccept = False
            ConvergenceRestriction = True
            
            if FreshJacobian and diverging:
                h = max(0.5 * hrmin, alpharef / alpha) * h
                info['nDiverge'] += 1
            elif FreshJacobian:
                if alpha > alpharef:
                    h = max(0.5 * hrmin, alpharef / alpha) * h
                else:
                    h = 0.5 * h
            
            if not FreshJacobian:
                dfdx, dgdx = jac(t, x, *args)
                info['nJac'] += 1
                FreshJacobian = True
            
            hgamma = h * gamma
            dRdx = dgdx - hgamma * dfdx
            LU = lu_factor(dRdx)
            info['nLU'] += 1
            hLU = h

        # Storage of variables for output
        if CurrentStepAccept:
            nAccept = info['nAccept']
            if nAccept >= len(Tout):
                Tout = np.vstack([Tout, np.zeros((chunk, 1))])
                Xout = np.vstack([Xout, np.zeros((chunk, nx))])
                Gout = np.vstack([Gout, np.zeros((chunk, nx))])
            
            Tout[nAccept, 0] = t
            Xout[nAccept, :] = x
            Gout[nAccept, :] = g

    info['nSlowConv'] = len([x for x in stats['SlowConv'] if x != 0])
    
    # Trim output arrays
    nAccept = info['nAccept']
    Tout = Tout[:nAccept, 0]
    Xout = Xout[:nAccept, :]
    Gout = Gout[:nAccept, :]

    return Tout, Xout, Gout, info, stats