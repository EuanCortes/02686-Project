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
                    t, y, g, _, stats = ESDIRK(f_esdirk, J_esdirk, t_span, x0, h0, tolerance, tolerance, Method=Method)
                    h = np.array(stats['h'])
                    r = np.array(stats['r'])
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
                   esdirk = False):
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
                        ref_sol = solve_ivp(f, t_span, x0_ref, method=reference_solver, 
                        t_eval=np.linspace(t_span[0], t_span[1], N), jac = J_c, rtol=tolerance, atol=tolerance)
                        Method = "ESDIRK23"


                        t_sol, y, g, _, stats = ESDIRK(f_esdirk, J_esdirk, t_span, x0, h0, tolerance, tolerance, Method=Method)
                        h = np.array(stats['h'])
                        r = np.array(stats['r'])
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



def compare_solvers_pfr(model_func, t_span,  x0,
                   fixed_step_sizes=[0.1, 0.01], 
                   tolerances = [1e-3, 1e-6],
                   h0 = 0.1,
                   reference_solver='BDF', 
                   figsize=(16, 12), 
                   model_name="PFR 3 state",
                   reference_solver_name="ode15s",
                   fixed_steps=False,
                   adaptive_steps=False,
                   steptype = "fixed",
                   explicit = False,
                   implicit = False,
                   euler = False,
                   rk45 = False,
                   dopri = False, 
                   esdirk = False):
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


    params = {
        "dz": 0.1,
        "v": 0.00004, # velocity, F/A, F = 400 ml/min A = 0.1 m^2 => v = 400 / 1000 / 1000 / 60 / 0.1 = 
        "D": [0.1, 0.1, 0.1],
        "beta": 560.0 / (1.0 * 4.186),
        "k": lambda T: 1.0 * np.exp(-5000/T),
    }
    Tin = 273.65
    CA_in = 1.6 / 2
    CB_in = 2.4 / 2
    u = [CA_in, CB_in, Tin]


    plt.figure(figsize=figsize)
    plt.suptitle(f"{model_name}: Comparison of {methodname} {overall_methodname} with {steptype} step size vs {reference_solver_name}", fontsize=16)
    
    # Dictionary to store results
    results = {}

    n = 50

        
    # Compare for each step size
    if fixed_steps:
        for i, dt in enumerate(fixed_step_sizes):
            N = int((t_span[1] - t_span[0]) / dt) + 1
            # Determine number of points based on step size
            f, J = model_func(p = params, u = u)
            
            if explicit:
                ref_sol = solve_ivp(f, t_span, x0, method=reference_solver, 
                            t_eval=np.linspace(t_span[0], t_span[1], N))
                if euler:
                    t_sol, y = ExplicitEulerFixedSteps(f, t_span[0], t_span[1], N, x0)
                if rk45:
                    solver = rk4()
                    t_sol,y = ExplicitRungeKuttaSolver(f, t_span, x0, dt, solver)
            if implicit:
                ref_sol = solve_ivp(f, t_span, x0, method=reference_solver, 
                            t_eval=np.linspace(t_span[0], t_span[1], N,))
                t_sol, y = ImplicitEulerFixedStep(f, J, t_span[0], t_span[1], N, x0)

            CA_ref  = ref_sol.y[:n]
            CB_ref  = ref_sol.y[n:2*n]
            T_ref   = ref_sol.y[2*n:3*n]
            CA = y[:n]
            CB = y[n:2*n]
            T = y[2*n:3*n]

            CA_error = np.mean(abs_error(CA[:,1:], CA_ref),axis = 0)
            CB_error = np.mean(abs_error(CB[:,1:], CB_ref), axis = 0)
            T_error = np.mean(abs_error(T[:,1:], T_ref), axis = 0)
            error = np.mean(np.array([CA_error, CB_error, T_error]),axis = 0)
            
            results[dt] ={
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
                     
        # Plot error comparison
        plt.subplot(3, 1, 3)
        for i, dt in enumerate(fixed_step_sizes):
            res = results[dt]
            plt.plot(res['t_sol'][1:],res['error'], label=f'h = {dt}')
            plt.yscale('log')
            plt.title(f"{model_name}: Mean Relative Error Comparison")
            plt.xlabel("Time")
            plt.ylabel("Relative Error")
            plt.legend()
            plt.grid(True)
        
        for i in range(len(fixed_step_sizes)):
            ax1 = plt.subplot(3, 1, i+1)
            ax2 = ax1.twinx()
            ax1.plot(res['t_sol'], res['CA_sol'][0], label=f'C_A: {methodname} {overall_methodname}',linestyle = '--', color='blue')
            ax1.plot(res['t_ref'], res['CA_ref'][0], label=f'C_A: {reference_solver_name}', color='red')
            ax1.plot(res['t_sol'], res['CB_sol'][0], label=f'C_B: {methodname} {overall_methodname}',linestyle = '--', color='green')
            ax1.plot(res['t_ref'], res['CB_ref'][0], label=f'C_B: {reference_solver_name}', color='orange')
            ax2.plot(res['t_sol'], res['T_sol'][0], label=f'T: {methodname} {overall_methodname}',linestyle = '--', color='purple')
            ax2.plot(res['t_ref'], res['T_ref'][0], label=f'T: {reference_solver_name}', color='lightblue')
            plt.legend()
            plt.title(f"Comparison of {methodname} {overall_methodname} with {steptype} step size and {reference_solver_name} (dt = {dt})")
            plt.xlabel("Time")
            ax1.set_ylabel("Concentration")
            ax2.set_ylabel("Temperature")
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
        ax1_right = ax1.twinx()          # Right axis for T (first row)
        ax2 = fig.add_subplot(gs[1, :])  # Second row spans both columns
        ax2_right = ax2.twinx()          # Right axis for T (second row)
        
        # Step size and error ratio plots (third row)
        ax3 = fig.add_subplot(gs[2, 0])  # Third row, first column (h)
        ax4 = fig.add_subplot(gs[2, 1])  # Third row, second column (r)
            
        # Calculate number of points based on 
        for idx, tolerance in enumerate(tolerances):
            f, J = model_func(p = params, u = u)
            f_esdirk, J_esdirk = model_func(p=params, u = u, esdirk=esdirk)
        
            if explicit:
                ref_sol = solve_ivp(f, t_span, x0, method=reference_solver, t_eval=np.linspace(t_span[0], t_span[1], N))
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
                    ref_sol = solve_ivp(f, t_span, x0, method=reference_solver, 
                    t_eval=np.linspace(t_span[0], t_span[1], N))
                    Method = "ESDIRK23"


                    t_sol, y, g, _, stats = ESDIRK(f_esdirk, J_esdirk, t_span, x0, h0, tolerance, tolerance, Method=Method)
                    h = np.array(stats['h'])
                    r = np.array(stats['r'])
                else:
                    ref_sol = solve_ivp(f, t_span, x0, method=reference_solver, 
                    t_eval=np.linspace(t_span[0], t_span[1], N))
                    t_sol, y, h, r, n_accept, n_reject, n_functions = ImplicitEulerAdaptiveStep(f, J, t_span, x0, h0, tolerance, tolerance)

            CA_ref  = ref_sol.y[:n]
            CB_ref  = ref_sol.y[n:2*n]
            T_ref   = ref_sol.y[2*n:3*n]
            CA = y[:,:n].T
            CB = y[:,n:2*n].T
            T = y[:,2*n:3*n].T

            CA_error = np.mean(abs_error(CA[:,-1], CA_ref[:,-1]),axis = 0)
            CB_error = np.mean(abs_error(CB[:,-1], CB_ref[:,-1]), axis = 0)
            T_error = np.mean(abs_error(T[:,-1], T_ref[:,-1]), axis = 0)
            error = np.mean(np.array([CA_error, CB_error, T_error]),axis = 0)
            
            results[tolerance] ={
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
                'n_functions': n_functions,
                'ref_nfun': ref_sol.nfev
        }    
        
        for idx, tolerance in enumerate(tolerances):
            res = results[tolerance]
            
            # Plot step sizes and error ratios (unchanged)
            ax3.semilogy(res['h'], label=f'tol={tolerance:.0e}')
            ax4.semilogy(res['r'], label=f'tol={tolerance:.0e}')
            
            if idx == 0:
                # Plot CA and CB on left y-axis (ax1)
                ax1.plot(res['t_sol'], res['CA_sol'][0], '--', color='blue', 
                        label=f'C_A: {methodname} (tol={tolerance:.0e})')
                ax1.plot(res['t_ref'], res['CA_ref'][0], color='red', 
                        label=f'C_A: {reference_solver_name}')
                ax1.plot(res['t_sol'], res['CB_sol'][0], '--', color='green', 
                        label=f'C_B: {methodname} (tol={tolerance:.0e})')
                ax1.plot(res['t_ref'], res['CB_ref'][0], color='orange', 
                        label=f'C_B: {reference_solver_name}')
                
                # Plot T on right y-axis (ax1_right)
                ax1_right.plot(res['t_sol'], res['T_sol'][0], '--', color='purple', 
                            label=f'T: {methodname} (tol={tolerance:.0e})')
                ax1_right.plot(res['t_ref'], res['T_ref'][0], color='lightblue', 
                            label=f'T: {reference_solver_name}')
                
                # Configure axes
                ax1.set_title(f"Phase Portrait - Numerical Method vs Reference (tol={tolerance:.0e})")
                ax1.set_xlabel("Time")
                ax1.set_ylabel("Concentration (CA, CB)")
                ax1_right.set_ylabel("Temperature (T)")
                
                # Combine legends
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax1_right.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
                ax1.grid(True)
            else:
                # Plot CA and CB on left y-axis (ax2)
                ax2.plot(res['t_sol'], res['CA_sol'][0], '--', color='blue', 
                        label=f'C_A: {methodname} (tol={tolerance:.0e})')
                ax2.plot(res['t_ref'], res['CA_ref'][0], color='red', 
                        label=f'C_A: {reference_solver_name}')
                ax2.plot(res['t_sol'], res['CB_sol'][0], '--', color='green', 
                        label=f'C_B: {methodname} (tol={tolerance:.0e})')
                ax2.plot(res['t_ref'], res['CB_ref'][0], color='orange', 
                        label=f'C_B: {reference_solver_name}')
                
                # Plot T on right y-axis (ax2_right)
                ax2_right.plot(res['t_sol'], res['T_sol'][0], '--', color='purple', 
                            label=f'T: {methodname} (tol={tolerance:.0e})')
                ax2_right.plot(res['t_ref'], res['T_ref'][0], color='lightblue', 
                            label=f'T: {reference_solver_name}')
                
                # Configure axes
                ax2.set_title(f"Phase Portrait - Numerical Method vs Reference (tol={tolerance:.0e})")
                ax2.set_xlabel("Time")
                ax2.set_ylabel("Concentration (CA, CB)")
                ax2_right.set_ylabel("Temperature (T)")
                
                # Combine legends
                lines1, labels1 = ax2.get_legend_handles_labels()
                lines2, labels2 = ax2_right.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
                ax2.grid(True)
        
        # Configure step size and error ratio plots (unchanged)
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
