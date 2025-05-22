import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from Solvers import *

def abs_error(y, y_exact):
    return np.abs(y - y_exact)/np.abs(y_exact)

def compare_solvers(model_func, t_span, x0, params=None, 
                   fixed_step_sizes=[0.1, 0.01], 
                   tolerances=[1e-3, 1e-6],
                   h0=0.1,
                   reference_solver='RK45', 
                   figsize=(16, 16),  # Increased height for additional subplots
                   model_name="Model",
                   reference_solver_name="ode45",
                   tight_layout=True,
                   fixed_steps=False,
                   adaptive_steps=False,
                   steptype="fixed",
                   explicit=False,
                   implicit=False,
                   euler=False,
                   rk45=False,
                   dopri=False, 
                   esdirk=False,
                   cstr=False):
    
    # Method naming
    if implicit:
        methodname = "Implicit"
    else:
        methodname = "Explicit"

    if euler:
        overall_methodname = "Euler"
    if rk45:
        overall_methodname = "Classic Runge-Kutta"
    if dopri:
        overall_methodname = "Dormand-Prince 4(5)"
    if esdirk:
        overall_methodname = "ESDIRK 23" 
    
    # Dictionary to store results
    results = {}

    # Fixed step comparisons
    if fixed_steps:
        plt.figure(figsize=figsize)
        plt.suptitle(f"{model_name}: Comparison of {methodname} {overall_methodname} with {steptype} step size vs {reference_solver_name}", fontsize=16)
        
        for i, dt in enumerate(fixed_step_sizes):
            N = int((t_span[1] - t_span[0]) / dt) + 1
            f, J = model_func(*params)
            
            if explicit:
                ref_sol = solve_ivp(f, t_span, x0, method=reference_solver, 
                            t_eval=np.linspace(t_span[0], t_span[1], N))
                if euler:
                    t, y = ExplicitEulerFixedSteps(f, t_span[0], t_span[1], N, x0)
                if rk45:
                    solver = rk4()
                    t, y = ExplicitRungeKuttaSolver(f, t_span, x0, dt, solver)
            if implicit:
                ref_sol = solve_ivp(f, t_span, x0, method=reference_solver, 
                            t_eval=np.linspace(t_span[0], t_span[1], N), jac=J)
                t, y = ImplicitEulerFixedStep(f, J, t_span[0], t_span[1], N, x0)

            error = abs_error(y[:,1:], ref_sol.y) if not rk45 else abs_error(y, ref_sol.y)
            avg_error = (error[0,:] + error[1,:])/2
            
            results[dt] = {
                'ref_sol': ref_sol,
                't': t,
                'y': y,
                'error': error,
                'avg_error': avg_error,
                'ref_nfun': ref_sol.nfev
            }
            
            # Phase portrait
            plt.subplot(3, 1, i+1)
            plt.plot(y[0,:], y[1,:], '--', label=f'{methodname} {overall_methodname}', color='blue')
            plt.plot(ref_sol.y[0,:], ref_sol.y[1,:], label=reference_solver_name, color='red')
            plt.legend()
            plt.title(f"Phase Portrait (h = {dt})")
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.grid(True)

        # Error comparison
        plt.subplot(3, 1, 3)
        for dt in fixed_step_sizes:
            res = results[dt]
            t_plot = res['t'][1:] if not rk45 else res['t']
            plt.plot(t_plot, res['avg_error'], label=f'h = {dt}')
        
        plt.yscale('log')
        plt.title("Error Comparison")
        plt.xlabel("Time")
        plt.ylabel("Average Relative Error")
        plt.legend()
        plt.grid(True)
        
        if tight_layout:
            plt.tight_layout()
        plt.show()
        
    # Adaptive step comparisons
    if adaptive_steps:
        fig = plt.figure(figsize=figsize)
        plt.suptitle(f"{model_name}: Comparison of {methodname} {overall_methodname} with {steptype} step size vs {reference_solver_name}", fontsize=16)
        # Create grid for subplots (3 rows, 2 columns)
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
        
        # Phase portraits (top two rows)
        ax1 = fig.add_subplot(gs[0, :])  # First row spans both columns
        ax2 = fig.add_subplot(gs[1, :])  # Second row spans both columns
        
        # Step size and error ratio plots (third row)
        ax3 = fig.add_subplot(gs[2, 0])  # Third row, first column (h)
        ax4 = fig.add_subplot(gs[2, 1])  # Third row, second column (r)
        
        N = int((t_span[1] - t_span[0]) / h0) + 1
        
        for idx, tolerance in enumerate(tolerances):
            f, J = model_func(*params)
            f_esdirk, J_esdirk = model_func(*params, esdirk)
            
            if explicit:
                ref_sol = solve_ivp(f, t_span, x0, method=reference_solver, 
                                  t_eval=np.linspace(t_span[0], t_span[1], N),rtol=tolerance, atol=tolerance)
                if euler:   
                    t, y, h, r, n_accept, n_reject, n_functions = ExplicitEulerAdaptiveStep(f, t_span, x0, h0, tolerance, tolerance)
                if rk4:
                    solver = rk4()
                    t, y, h, r, n_accept, n_reject, n_functions= ExplicitRungeKuttaSolverAdaptive(f, t_span, x0, h0, solver, tolerance, tolerance)
                if dopri:
                    solver = dormand_prince_45()
                    t, y, h, r, n_accept, n_reject, n_functions = ExplicitRungeKuttaSolverAdaptive(f, t_span, x0, h0, solver, tolerance, tolerance)
            if implicit:
                if esdirk:
                    ref_sol = solve_ivp(f, t_span, x0, method=reference_solver, 
                                      t_eval=np.linspace(t_span[0], t_span[1], N), jac=J,rtol=tolerance, atol=tolerance)
                    Method = "ESDIRK23"
                    t, y, g, info, stats = ESDIRK(f_esdirk, J_esdirk, t_span, x0, h0, tolerance, tolerance, Method=Method)
                    h = np.array(stats['h'])
                    r = np.array(stats['r'])
                    n_accept = info['nAccept']
                    n_reject = info['nFail']
                    n_functions = n_accept + n_reject
                else:
                    ref_sol = solve_ivp(f, t_span, x0, method=reference_solver, 
                                      t_eval=np.linspace(t_span[0], t_span[1], N), jac=J,rtol=tolerance, atol=tolerance)
                    t, y, h, r, n_accept, n_reject, n_functions = ImplicitEulerAdaptiveStep(f, J, t_span, x0, h0, tolerance, tolerance)
            
            results[tolerance] = {
                'ref_sol': ref_sol,
                't': t,
                'y': y,
                'h': h[2:],
                'r': r[2:],
                'n_accept': n_accept,
                'n_reject': n_reject,
                'n_functions': n_functions,
                'ref_nfun': ref_sol.nfev,
                'ref_jacfun': ref_sol.njev
            }
            
        for idx, tolerance in enumerate(tolerances):
            res = results[tolerance]
            ax3.semilogy(res['h'],  label=f'tol={tolerance:.0e}')
            ax4.semilogy(res['r'],  label=f'tol={tolerance:.0e}')
            if idx == 0:
                # Plot phase portraits
                ax1.plot(y[:,0], y[:,1], '--', color='blue', 
                    label=f'{methodname} {overall_methodname} (tol={tolerance:.0e})')
                ax1.plot(ref_sol.y[0,:], ref_sol.y[1,:], color='red',
                    label=f'{reference_solver_name} (tol={tolerance:.0e})')
            else:
                # Plot phase portraits
                ax2.plot(y[:,0], y[:,1], '--', color='blue', 
                    label=f'{methodname} {overall_methodname} (tol={tolerance:.0e})')
                ax2.plot(ref_sol.y[0,:], ref_sol.y[1,:], color='red', 
                    label=f'{reference_solver_name} (tol={tolerance:.0e})')
            
            # Plot step sizes and error ratios

        
        # Configure phase portrait plots
        for ax, title in zip([ax1, ax2], [tolerances[0], tolerances[1]]):
            ax.set_title(f"Comparison of {methodname} {overall_methodname} with {steptype} step size and {reference_solver_name} (tolerance = {title})")
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.legend()
            ax.grid(True)
        
        # Configure step size and error ratio plots
        ax3.set_title("Adaptive Step Sizes")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Step Size (log scale)")
        ax3.legend()
        ax3.grid(True)
        
        ax4.set_title("Error Ratios")
        ax4.set_xlabel("Time")
        ax4.set_ylabel("Error Ratio (log scale)")
        ax4.legend()
        ax4.grid(True)
        
        if tight_layout:
            plt.tight_layout()
        plt.show()
        
    return results

def compare_solvers_cstr(model_func, t_span,  
                   fixed_step_sizes=[0.1, 0.01], 
                   tolerances = [1e-3, 1e-6],
                   h0 = 0.1,
                   reference_solver='RK45', 
                   figsize=(16, 12), 
                   model_name="CSTR 1 state",
                   reference_solver_name="ode45",
                   fixed_steps=False,
                   adaptive_steps=False,
                   steptype = "fixed",
                   explicit = False,
                   implicit = False,
                   euler = False,
                   rk45 = False,
                   dopri = False, 
                   esdirk = False,
                   ESDIRK = ESDIRK):
    # Create figure

    if implicit:
        methodname = "Implicit"
    else:
        methodname = "Explicit"

    if euler:
        overall_methodname = "Euler"
    if rk45:
        overall_methodname = "Classic Runge-Kutta "
    if dopri:
        overall_methodname = "Dormand-Prince 4(5)"
    if esdirk:
        overall_methodname = "ESDIRK 23" 

    min = 60
    # This F leads to potential unstable areas
    F = [0.7/min,0.6/min,0.5/min,0.4/min,0.3/min,0.2/min,0.3/min,0.4/min,0.5/min,0.6/min,0.7/min,0.7/min,0.2/min,0.2/min,0.7/min,0.7/min]

    t = np.array([None])
    Tf = np.array([None])
    t_ref = np.array([None])
    Tf_ref = np.array([None])

    # Initial conditions
    Tin = 273.65
    CA_in = 2.4/2
    CB_in = 1.6/2

    x0 = np.array([Tin])
    x0_ref = np.array([Tin])

    R = np.array([None])
    H = np.array([None])


    plt.figure(figsize=figsize)
    plt.suptitle(f"{model_name}: Comparison of {methodname} {overall_methodname} with {steptype} step size vs {reference_solver_name}", fontsize=16)
    
    # Dictionary to store results
    results = {}

    n_accept = 0
    n_reject = 0
    n_functions = 0
    ref_nfun = 0

        
    # Compare for each step size
    if fixed_steps:
        for i, dt in enumerate(fixed_step_sizes):
            N = int((t_span[1] - t_span[0]) / dt) + 1
            t = np.array([None])
            Tf = np.array([None])
            
            t_ref = np.array([None])
            Tf_ref = np.array([None])

            error = np.array([None])


            # Initial conditions
            Tin = 273.65
            CA_in = 1.6/2
            CB_in = 2.4/2

            x0 = np.array([Tin])
            x0_ref = np.array([Tin])

            R = np.array([None])
            H = np.array([None])
            for idx, f in enumerate(F):
            # Determine number of points based on step size
                
                params = [f,0.105,CA_in,CB_in,Tin]

                f, J = model_func(params)
                f_c, J_c = model_func(params, compare = True)
                if explicit:
                    ref_sol = solve_ivp(f, t_span, x0_ref, method=reference_solver, 
                                t_eval=np.linspace(t_span[0], t_span[1], N))
                    if euler:
                        t_sol, y = ExplicitEulerFixedSteps(f, t_span[0], t_span[1], N, x0)
                    if rk45:
                        solver = rk4()
                        t_sol,y = ExplicitRungeKuttaSolver(f, t_span, x0, dt, solver)
                if implicit:
                    ref_sol = solve_ivp(f, t_span, x0_ref, method=reference_solver, 
                                t_eval=np.linspace(t_span[0], t_span[1], N,), jac = J_c)
                    t_sol, y = ImplicitEulerFixedStep(f, J_c, t_span[0], t_span[1], N, x0)

                T_ss_ref = ref_sol.y[0][-1]
                T_ss = y[0][-1]

                x0 = np.array([T_ss])
                x0_ref = np.array([T_ss_ref])
                error = np.append(error,abs_error(T_ss, T_ss_ref))
                Tf = np.concatenate([Tf, y.ravel()-Tin])
                Tf_ref = np.concatenate([Tf_ref, ref_sol.y[0]-Tin])

                t = np.concatenate([t, (t_sol+(idx)*t_span[1])/min])
                t_ref = np.concatenate([t_ref, (ref_sol.t+(idx)*t_span[1])/min])

            
            results[fixed_step_sizes[i]] = {
                'ref_sol': Tf_ref[1:],
                'sol': Tf[1:],
                'error': error[1:],
                't_ref': t_ref[1:],
                't_sol': t[1:],
                'ref_nfun': ref_sol.nfev
                }  
                       
        # Plot error comparison
        plt.subplot(3, 1, 3)
        for i in range(len(fixed_step_sizes)):
            res = results[fixed_step_sizes[i]]
            plt.plot(res['error'], label=f'h = {fixed_step_sizes[i]}')
            plt.yscale('log')
            plt.title(f"{model_name}: Relative Error Comparison between steady state")
            plt.xlabel("Cycle")
            plt.ylabel("Relative Error")
            plt.legend()
            plt.grid(True)
        
        for i in range(len(fixed_step_sizes)):
            res = results[fixed_step_sizes[i]]
            plt.subplot(3, 1, i+1)
            plt.plot(res['t_sol'], res['sol'], label=f'{methodname} {overall_methodname}',linestyle = '--', color='blue')
            plt.plot(res['t_ref'], res['ref_sol'], label=reference_solver_name, color='red')
            plt.legend()
            plt.title(f"Comparison of {methodname} {overall_methodname} with {steptype} step size and {reference_solver_name} (h = {fixed_step_sizes[i]})")
            plt.xlabel(f"Time")
            plt.ylabel("Temperature")
            plt.grid(True)


        plt.tight_layout()
        plt.show()
        
    if adaptive_steps:
        N = int((t_span[1] - t_span[0]) / h0) + 1
        fig = plt.figure(figsize=figsize)
        plt.suptitle(f"{model_name}: Adaptive {methodname} {overall_methodname} vs {reference_solver_name}", fontsize=16)
        
        # Create grid for subplots (3 rows, 2 columns)
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
        
        # Phase portraits (top two rows)
        ax1 = fig.add_subplot(gs[0, :])  # First row spans both columns
        ax2 = fig.add_subplot(gs[1, :])  # Second row spans both columns
        
        # Step size and error ratio plots (third row)
        ax3 = fig.add_subplot(gs[2, 0])  # Third row, first column (h)
        ax4 = fig.add_subplot(gs[2, 1])  # Third row, second column (r)
        
        # Calculate number of points based on 
        for idx, tolerance in enumerate(tolerances):
            t = np.array([None])
            Tf = np.array([None])
            
            t_ref = np.array([None])
            Tf_ref = np.array([None])

            # Initial conditions
            Tin = 273.65
            CB_in = 2.4/2
            CA_in = 1.6/2

            x0 = np.array([Tin])
            x0_ref = np.array([Tin])

            R = np.array([None])
            H = np.array([None])
            for i, f in enumerate(F):
                params = [f,0.105,CA_in,CB_in,Tin]
                f, J = model_func(params)
                f_esdirk, J_esdirk = model_func(params, esdirk)
                f_c, J_c = model_func(params, compare = True)
            
                if explicit:
                    ref_sol = solve_ivp(f, t_span, x0_ref, method=reference_solver, 
                    t_eval=np.linspace(t_span[0], t_span[1], N), jac = J_c, rtol=tolerance, atol=tolerance)
                    if euler:   
                        t_sol, y, h, r, n_accept, n_reject, n_functions = ExplicitEulerAdaptiveStep(f, t_span, x0, h0, tolerance, tolerance)
                    if rk4:
                        solver = rk4()
                        t_sol, y, h, r, n_accept, n_reject, n_functions = ExplicitRungeKuttaSolverAdaptive(f, t_span, x0, h0, solver, tolerance, tolerance)
                    if dopri:
                        solver = dormand_prince_45()
                        t_sol, y, h, r, n_accept, n_reject, n_functions = ExplicitRungeKuttaSolverAdaptive(f, t_span, x0, h0, solver, tolerance, tolerance)
                if implicit:
                    if esdirk:
                        print("starting solve_ivp solver")
                        ref_sol = solve_ivp(f, t_span, x0_ref, method=reference_solver, 
                        t_eval=np.linspace(t_span[0], t_span[1], N), jac = J_c, rtol=tolerance, atol=tolerance)
                        Method = "ESDIRK23"
                        print("Solved via solve_ivp")
                        print("starting ESDIRK solver")
                        t_sol, y, g, info, stats = ESDIRK(f_esdirk, J_esdirk, t_span, x0, h0, tolerance, tolerance, Method=Method)
                        print("Solved via ESDIRK")
                        h = np.array(stats['h'])
                        r = np.array(stats['r'])
                        n_accept = info['nAccept']
                        n_reject = info['nFail']
                    else:
                        ref_sol = solve_ivp(f, t_span, x0_ref, method=reference_solver, 
                        t_eval=np.linspace(t_span[0], t_span[1], N), jac = J_c, rtol=tolerance, atol=tolerance)
                        t_sol, y, h, r, n_accept, n_reject, n_functions = ImplicitEulerAdaptiveStep(f, J_c, t_span, x0, h0, tolerance, tolerance)
            

                T_ss_ref = ref_sol.y[0][-1]
                T_ss = y.ravel()[-1]

                x0 = np.array([T_ss])
                x0_ref = np.array([T_ss_ref])
                Tf = np.concatenate([Tf, y.ravel()-Tin])
                Tf_ref = np.concatenate([Tf_ref, ref_sol.y[0]-Tin])
                t = np.concatenate([t, (t_sol+(i)*t_span[1])/min])
                t_ref = np.concatenate([t_ref, (ref_sol.t+(i)*t_span[1])/min])
                R = np.concatenate([R, r])
                H = np.concatenate([H, h])
                n_accept += n_accept
                n_reject += n_reject
                n_functions += n_functions
                ref_nfun += ref_sol.nfev


            # Store results
            results[tolerance] = {
                'ref_sol': Tf_ref[1:],
                'sol': Tf[1:],
                'r': R,
                't_ref': t_ref[1:],
                't_sol': t[1:],
                'h': H,
                'n_accept': n_accept,
                'n_reject': n_reject,
                'n_functions': n_functions,
                'ref_nfun': ref_nfun
            }
            
        for idx, tolerance in enumerate(tolerances):
            res = results[tolerance]
            ax3.semilogy(res['h'],  label=f'tol={tolerance:.0e}')
            ax4.semilogy(res['r'],  label=f'tol={tolerance:.0e}')
            if idx == 0:
                # Plot phase portraits
                ax1.plot(res['t_sol'], res['sol'], '--', color='blue', 
                    label=f'{methodname} {overall_methodname} (tol={tolerance:.0e})')
                ax1.plot(res['t_ref'], res['ref_sol'], color='red',
                    label=f'{reference_solver_name} (tol={tolerance:.0e})')
            else:
                # Plot phase portraits
                ax2.plot(res['t_sol'], res['sol'], '--', color='blue', 
                    label=f'{methodname} {overall_methodname} (tol={tolerance:.0e})')
                ax2.plot(res['t_ref'], res['ref_sol'], color='red', 
                    label=f'{reference_solver_name} (tol={tolerance:.0e})')
            
            # Plot step sizes and error ratios

        
        # Configure phase portrait plots
        for ax, title in zip([ax1, ax2], [tolerances[0], tolerances[1]]):
            ax.set_title(f"Comparison of {methodname} {overall_methodname} with {steptype} step size and {reference_solver_name} (tolerance = {title})")
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.legend()
            ax.grid(True)
        
        
        # Configure step size and error ratio plots
        ax3.set_title("Adaptive Step Sizes")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Step Size (log scale)")
        ax3.legend()
        ax3.grid(True)
        
        ax4.set_title("Error Ratios")
        ax4.set_xlabel("Time")
        ax4.set_ylabel("Error Ratio (log scale)")
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    return results

def compare_solvers_pfr(model_func, t_span, x0,
                      fixed_step_sizes=[0.1, 0.01], 
                      tolerances=[1e-3, 1e-6],
                      h0=0.1,
                      reference_solver='BDF', 
                      figsize=(16, 12), 
                      model_name="PFR 3 state",
                      reference_solver_name="ode15s",
                      fixed_steps=False,
                      adaptive_steps=False,
                      steptype="fixed",
                      explicit=False,
                      implicit=False,
                      euler=False,
                      rk45=False,
                      dopri=False, 
                      esdirk=False,
                      s = 2,
                      n = 5,
                      ESDIRK = ESDIRK):
    """
    Compare different numerical solvers for PFR model with visualization.
    
    Fixed issues in original code:
    - Proper handling of solver selection flags
    - Correct array indexing for solution components
    - Improved plotting layout and labels
    - Better error calculation
    - Fixed reference solution evaluation points
    - Proper handling of adaptive step results
    """
    
    # Validate input parameters
    if not (fixed_steps or adaptive_steps):
        raise ValueError("Must select either fixed_steps or adaptive_steps")
    if not (explicit or implicit):
        raise ValueError("Must select either explicit or implicit")
    
    # Determine method names
    methodname = "Implicit" if implicit else "Explicit"
    
    if euler:
        overall_methodname = "Euler"
    elif rk45:
        overall_methodname = "Classic Runge-Kutta"
    elif dopri:
        overall_methodname = "Dormand-Prince 4(5)"
    elif esdirk:
        overall_methodname = "ESDIRK 23"
    else:
        overall_methodname = "Unknown Method"

    # Define parameters
    params = {
        "dz": 0.1,
        "v": 0.01,
        "D": [0.1, 0.1, 0.1],
        "beta": 560.0 / (1.0 * 4.186),
        "k": lambda T: 1.0 * np.exp(-5000/T),
    }
    Tin = 273.65
    CA_in = 1.6 / 2
    CB_in = 2.4 / 2
    u = [CA_in, CB_in, Tin]

      # Number of spatial points
    results = {}

    # Fixed step size comparison
    if fixed_steps:
        plt.figure(figsize=figsize)
        plt.suptitle(f"{model_name}: {methodname} {overall_methodname} vs {reference_solver_name}", 
                    fontsize=16)
        
        # Create subplots for each step size
        n_plots = len(fixed_step_sizes)
        axes = []
        for i in range(n_plots):
            if i  == 0:
                ax = plt.subplot(n_plots+1, 1, i+1)
            else:
                ax = plt.subplot(n_plots+1, 1, i+1, sharex=axes[0])
            axes.append(ax)
        
        # Error plot at the bottom
        error_ax = plt.subplot(n_plots+1, 1, n_plots+1)
        
        for i, dt in enumerate(fixed_step_sizes):
            N = int((t_span[1] - t_span[0]) / dt) + 1
            t_eval = np.linspace(t_span[0], t_span[1], N)
            
            # Get model functions
            f, J = model_func(p=params, u=u)
            
            # Compute reference solution
            ref_sol = solve_ivp(f, t_span, x0, method=reference_solver, 
                               t_eval=t_eval, rtol=1e-8, atol=1e-8)
            
            # Compute numerical solution
            if explicit:
                if euler:
                    t_sol, y = ExplicitEulerFixedSteps(f, t_span[0], t_span[1], N, x0)
                elif rk45:
                    solver = rk4()
                    t_sol, y = ExplicitRungeKuttaSolver(f, t_span, x0, dt, solver)
            elif implicit:
                if esdirk:
                    raise NotImplementedError("ESDIRK not implemented for fixed steps")
                else:
                    t_sol, y = ImplicitEulerFixedStep(f, J, t_span[0], t_span[1], N, x0)
            
            # Extract components
            CA_ref = ref_sol.y[:n,:]
            CB_ref = ref_sol.y[n:2*n,:]
            T_ref = ref_sol.y[2*n:,:]
            
            # Handle different solution array shapes
            if len(y.shape) == 2:  # Time steps in columns
                CA = y[:n, :]
                CB = y[n:2*n, :]
                T = y[2*n:, :]
            else:  # Single time step
                CA = y[:n].reshape(-1, 1)
                CB = y[n:2*n].reshape(-1, 1)
                T = y[2*n:].reshape(-1, 1)
            
            # Calculate errors
            diff_len = abs(len(CA_ref[0])-len(CA[0]))
            CA_error = np.mean(abs_error(CA[:,diff_len:], CA_ref), axis=0)
            CB_error = np.mean(abs_error(CB[:,diff_len:], CB_ref), axis=0)
            T_error = np.mean(abs_error(T[:,diff_len:], T_ref), axis=0)
            error = np.mean(np.array([CA_error, CB_error, T_error]), axis=0)
            
            # Store results
            results[dt] = {
                'CA_ref': CA_ref,
                'CB_ref': CB_ref,
                'T_ref': T_ref,
                'CA_sol': CA,
                'CB_sol': CB,
                'T_sol': T,
                'error': error,
                't_sol': t_sol,
                't_ref': ref_sol.t
            }
            
            # Plot concentrations and temperature
            ax = axes[i]
            ax_right = ax.twinx()
            
            # Plot reference solutions
            ax.plot(ref_sol.t, CA_ref[s,:], 'r-', label='CA ref')
            ax.plot(ref_sol.t, CB_ref[s,:], 'g-', label='CB ref')
            ax_right.plot(ref_sol.t, T_ref[s,:], 'b-', label='T ref')
            
            # Plot numerical solutions
            ax.plot(t_sol, CA[0], 'r--', label='CA num')
            ax.plot(t_sol, CB[0], 'g--', label='CB num')
            ax_right.plot(t_sol, T[0], 'b--', label='T num')
            
            ax.set_title(f"h = {dt:.4f}")
            ax.set_ylabel("Concentration")
            ax_right.set_ylabel("Temperature")
            
            # Combine legends
            lines, labels = ax.get_legend_handles_labels()
            lines_right, labels_right = ax_right.get_legend_handles_labels()
            ax.legend(lines + lines_right, labels + labels_right, loc='upper right')
            ax.grid(True)
            
            # Plot error
            diff_len_error = abs(len(t_sol)-len(error))
            error_ax.semilogy(t_sol[diff_len_error:], error, label=f'dt = {dt:.4f}')
        
        # Configure error plot
        error_ax.set_title("Mean Absolute Error")
        error_ax.set_xlabel("Time")
        error_ax.set_ylabel("Error (log scale)")
        error_ax.legend()
        error_ax.grid(True)
        
        plt.tight_layout()
        plt.show()

    # Adaptive step size comparison
    if adaptive_steps:
        fig = plt.figure(figsize=figsize)
        plt.suptitle(f"{model_name}: Adaptive {methodname} {overall_methodname} vs {reference_solver_name}", 
                    fontsize=16)
        
        # Create grid for subplots
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
        
        # Phase portraits (top two rows)
        ax1 = fig.add_subplot(gs[0, :])
        ax1_right = ax1.twinx()
        ax2 = fig.add_subplot(gs[1, :])
        ax2_right = ax2.twinx()
        
        # Step size and error ratio plots (third row)
        ax3 = fig.add_subplot(gs[2, 0])
        ax4 = fig.add_subplot(gs[2, 1])
        
        # Calculate reference solution with dense output
        f, J = model_func(p=params, u=u)
        ref_sol = solve_ivp(f, t_span, x0, method=reference_solver, 
                           rtol=1e-8, atol=1e-8, dense_output=True)
        
        for idx, tolerance in enumerate(tolerances):
            if explicit:
                if euler:
                    t_sol, y, h, r, n_accept, n_reject, n_functions = ExplicitEulerAdaptiveStep(
                        f, t_span, x0, h0, tolerance, tolerance)
                elif rk45:
                    solver = rk4()
                    t_sol, y, h, r, n_accept, n_reject, n_functions = ExplicitRungeKuttaSolverAdaptive(
                        f, t_span, x0, h0, solver, tolerance, tolerance)
                elif dopri:
                    solver = dormand_prince_45()
                    t_sol, y, h, r, n_accept, n_reject, n_functions = ExplicitRungeKuttaSolverAdaptive(
                        f, t_span, x0, h0, solver, tolerance, tolerance)
            elif implicit:
                if esdirk:
                    f_esdirk, J_esdirk = model_func(p=params, u=u, esdirk=True)
                    Method = "ESDIRK23"
                    t_sol, y, g, info, stats = ESDIRK(f_esdirk, J_esdirk, t_span, x0, h0, 
                                                  tolerance, tolerance, Method=Method)
                    h = np.array(stats['h'])
                    r = np.array(stats['r'])
                    n_accept = info['nAccept']
                    n_reject = info['nFail']
                    #n_functions = stats['n_functions']
                else:
                    t_sol, y, h, r, n_accept, n_reject, n_functions = ImplicitEulerAdaptiveStep(
                        f, J, t_span, x0, h0, tolerance, tolerance)
            
            # Evaluate reference solution at numerical solution time points
            CA_ref = ref_sol.sol(t_sol)[:n]
            CB_ref = ref_sol.sol(t_sol)[n:2*n]
            T_ref = ref_sol.sol(t_sol)[2*n:]
            
            # Extract numerical solution components
            CA = y[:, :n].T
            CB = y[:, n:2*n].T
            T = y[:, 2*n:].T
            
            # Calculate errors
            CA_error = np.mean(abs_error(CA, CA_ref), axis=0)
            CB_error = np.mean(abs_error(CB, CB_ref), axis=0)
            T_error = np.mean(abs_error(T, T_ref), axis=0)
            error = np.mean(np.array([CA_error, CB_error, T_error]), axis=0)
            
            # Store results
            results[tolerance] = {
                'CA_ref': CA_ref,
                'CB_ref': CB_ref,
                'T_ref': T_ref,
                'CA_sol': CA,
                'CB_sol': CB,
                'T_sol': T,
                'error': error,
                't_sol': t_sol,
                't_ref': ref_sol.t,
                'r': r,
                'h': h,
                'n_accept': n_accept,
                'n_reject': n_reject,
                #'n_functions': n_functions,
                'ref_nfun': ref_sol.nfev
            }
            
            # Plot step sizes and error ratios
            ax3.semilogy(h, label=f'tol={tolerance:.0e}')
            ax4.semilogy(r, label=f'tol={tolerance:.0e}')
            
            # Plot phase portraits
            target_ax = ax1 if idx == 0 else ax2
            target_ax_right = ax1_right if idx == 0 else ax2_right
            
            # Plot concentrations
            target_ax.plot(t_sol, CA[0], 'b--', label=f'CA (tol={tolerance:.0e})')
            target_ax.plot(t_sol, CB[0], 'g--', label=f'CB (tol={tolerance:.0e})')
            
            # Plot reference concentrations
            target_ax.plot(ref_sol.t, ref_sol.y[0], 'r-', label='CA ref')
            target_ax.plot(ref_sol.t, ref_sol.y[n], 'm-', label='CB ref')
            
            # Plot temperatures
            target_ax_right.plot(t_sol, T[0], 'k--', label=f'T (tol={tolerance:.0e})')
            target_ax_right.plot(ref_sol.t, ref_sol.y[2*n], 'c-', label='T ref')
            
            # Configure axes
            target_ax.set_title(f"Phase Portrait (tol={tolerance:.0e})")
            target_ax.set_xlabel("Time")
            target_ax.set_ylabel("Concentration")
            target_ax_right.set_ylabel("Temperature")
            
            # Combine legends
            lines, labels = target_ax.get_legend_handles_labels()
            lines_right, labels_right = target_ax_right.get_legend_handles_labels()
            target_ax.legend(lines + lines_right, labels + labels_right, loc='best')
            target_ax.grid(True)
        
        # Configure step size and error ratio plots
        ax3.set_title("Adaptive Step Sizes")
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Step Size (log scale)")
        ax3.legend()
        ax3.grid(True)
        
        ax4.set_title("Error Ratios")
        ax4.set_xlabel("Step")
        ax4.set_ylabel("Error Ratio (log scale)")
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    return results