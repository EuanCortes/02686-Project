import numpy as np



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
###############################