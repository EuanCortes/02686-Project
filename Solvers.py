import numpy as np
from scipy.linalg import lu_factor, lu_solve

###############################
##### Explicit ODE solver #####
###############################

# Explicit Euler with fixed step size
def ExplicitEulerFixedSteps(fun, t0, tN, N, x0, *args):
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
def NewtonsMethodODE(fun_jac, t, x, dt, xinit, tol, maxit, *args):
    x_new = xinit
    for i in range(maxit):
        f, J = fun_jac(t + dt, x_new, *args)
        dx = np.linalg.solve(J, -f)
        x_new = x_new + dx
        if np.linalg.norm(dx) < tol:
            break
    return x_new

# Implicit Euler with fixed step size
def ImplicitEulerFixedStep(fun_jac, ta, tb, N, xa, *args):
    # Compute step size and allocate memory
    dt = (tb - ta) / N
    nx = xa.shape[0]
    X = np.zeros((nx, N + 1))
    T = np.zeros(N + 1)

    tol = 1.0e-8
    maxit = 100

    # Euler's Implicit Method
    T[0] = ta
    X = xa
    for k in range(N):
        f = fun_jac(T[k], X[k], *args)
        T[k + 1] = T[k] + dt
        xinit = X[k] + f * dt
        X[:, k + 1] = NewtonsMethodODE(fun_jac, T[k], X[k], dt, xinit, tol, maxit, *args)

    return T, X

# Implicit Euler with adaptive step size
def ExplicitEulerAdaptiveStep(f, tspan, x0, h0, abstol, reltol, *args):
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
            max1 = np.max([abstol, np.abs(xnewm) * reltol])
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


# Explicit-explicit method for SDE with fixed step size
def SDEsolverExplicitExplicit(ffun, gfun, T, x0, W, *varargin):
    N = len(T)
    nx = len(x0)
    X = np.zeros((nx, N))

    X[:, 0] = x0
    
    for k in range(N-1):
        f = ffun(T[k], X[:, k], *varargin)
        g = gfun(T[k], X[:, k], *varargin)
        dt = T[k+1] - T[k]
        dW = W[:, k+1] - W[:, k]
        psi = X[:, k] + g * dW
        X[:, k+1] = psi + f * dt
    return X

#Newtons method for SDE
def SDENewtonSolver(ffun, t, dt, psi, xinit, tol, maxit, *varargin):
    I = np.eye(len(xinit))
    x = xinit
    f, J = ffun(t, x, *varargin)
    R = x - f * dt - psi
    it = 1
    while (np.linalg.norm(R, np.inf) > tol) and (it <= maxit):
        dRdx = I - J * dt
        mdx = np.linalg.solve(dRdx, R)
        x = x - mdx
        f, J = ffun(t, x, *varargin)
        R = x - f * dt - psi
        it += 1
    return x, f, J

#Implicit-explicit method for SDE with fixed step size
def SDEsolverImplicitExplicit(ffun, gfun, T, x0, W, *varargin):
    tol = 1.0e-8
    maxit = 100

    N = len(T)
    nx = len(x0)
    X = np.zeros((nx, N))

    X[:, 0] = x0

    for k in range(N-1):
        f = ffun(T[k], X[:, k], *varargin)
        g = gfun(T[k], X[:, k], *varargin)
        dt = T[k+1] - T[k]
        dW = W[:, k+1] - W[:, k]
        psi = X[:, k] + g * dW
        xinit = psi + f * dt
        X[:, k+1], f, _ = SDENewtonSolver(ffun, T[k+1], dt, psi, xinit, tol, maxit, *varargin)

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

    Tout = np.zeros((N+1,1))
    Xout = np.zeros((N+1,nx))

    Tout[0] = t
    Xout[0,:] = x.T

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
        Xout[i+1,:] = x.T

    return Tout, Xout


# Runge-Kutta with adaptive step size
def ExplicitRungeKuttaSolverAdaptive(f, tspan, x0, h0, solver, abstol, reltol, *args):
    hmin = 0.1
    hmax = 5
    epstol = 0.8

    #Solver is a dict

    #Solver parameters
    s = solver["stages"]
    AT = solver["AT"]
    b = solver["b"]
    c = solver["c"]

    #Parameters related to constant step size

    #Size parameters
    x = x0
    t = tspan[0]
    tf = tspan[1]
    nx = len(x0)

    #Allocate memory
    T = np.zeros((s))
    X = np.zeros((nx,s))
    F = np.zeros((nx,s))
    Tm = np.zeros((s))
    Xm = np.zeros((nx,s))
    Fm = np.zeros((nx,s))
    
    Tout = [t]
    Xout = [x.T]
    H = [h0]

    h = h0

    while t < tf:
        #Adjust step size
        if t + h > tf:
            h = tf - t

        #Compute next step

        hm = 0.5*h
        tm = t + hm

        #Compute stages
        #Stage 1
        T[0] = t
        X[:,0] = x
        F[:,0] = f(T[0],X[:,0],*args)
        Tm[0] = tm
        Xm[:,0] = x
        Fm[:,0] = f(Tm[0],Xm[:,0],*args)

        #Stage 2, 3, ..., s
        for j in range(1,s):
            T[j] = t + h*c[j]
            X[:,j] = x + np.dot(F[:,:j],h*AT[:j,j])
            F[:,j] = f(T[j],X[:,j],*args)
            Tm[j] = tm + hm*c[j]
            Xm[:,j] = x + np.dot(Fm[:,:j],hm*AT[:j,j])
            Fm[:,j] = f(Tm[j],Xm[:,j],*args)

        #Update state
        xnew = x + np.dot(F,h*b)

        #Compute second time step
        xm = x + np.dot(Fm,hm*b)
        Xm[:,0] = xm
        for i in range(1,s):
            Xm[:,i] = xm + np.dot(Fm[:,:i-1],hm*AT[:i-1,i])  
            Fm[:,i] = f(Tm[i],Xm[:,i],*args)
            
        xnewm = xm + np.dot(Fm,hm*b)


        x_err = np.linalg.norm(xnew - xnewm)
        #max1 = np.max([abstol, np.abs(xnewm) * reltol])
        #r = np.max(x_err / max1)

        # Check if error is within tolerance
        if x_err < abstol:
            # Update time and state
            t = t + h
            x = xnew
            # Store values
            Tout.append(t)
            Xout.append(x)

        # Update step size
        # Update step size
        h = np.max([hmin, np.min([hmax, np.sqrt(epstol / x_err)])]) * h
        H.append(h)

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