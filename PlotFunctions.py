import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def plot_multiple_relative_errors(reference_series, comparison_series_list,
                                param1_name, param2_name,
                                series_names=None,
                                time=None,
                                title=None,
                                figsize=(12, 10),
                                ylim=None,
                                display_as_percentage=False):
    
    # Prepare reference data
    if isinstance(reference_series, dict):
        ref_param1 = reference_series[param1_name]
        ref_param2 = reference_series[param2_name]
    else:
        ref_param1 = reference_series[:, 0]
        ref_param2 = reference_series[:, 1]
    
    num_comparisons = len(comparison_series_list)
    
    # Default series names if not provided
    if series_names is None:
        series_names = [f'Series {i+1}' for i in range(num_comparisons)]
    elif len(series_names) != num_comparisons:
        raise ValueError("series_names must have same length as comparison_series_list")
    
    # Calculate relative absolute errors for all comparison series
    def relative_absolute_error(true, pred):
        return np.abs(true - pred) / (np.abs(true) + 1e-10)
    
    rae_param1 = []
    rae_param2 = []
    
    for comp_series in comparison_series_list:
        # Prepare comparison data
        if isinstance(comp_series, dict):
            comp_param1 = comp_series[param1_name]
            comp_param2 = comp_series[param2_name]
        else:
            comp_param1 = comp_series[:, 0]
            comp_param2 = comp_series[:, 1]
        
        rae_param1.append(relative_absolute_error(ref_param1, comp_param1))
        rae_param2.append(relative_absolute_error(ref_param2, comp_param2))
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Use a colormap for multiple series
    colors = plt.cm.tab10(np.linspace(0, 1, num_comparisons))
    
    # Plot parameter 1 errors
    for i in range(num_comparisons):
        mean_rae = np.mean(rae_param1[i])
        label = f'{series_names[i]} (Mean: {mean_rae:.2%})' if display_as_percentage else f'{series_names[i]} (Mean: {mean_rae:.4f})'
        
        if time is None:
            ax1.plot(rae_param1[i] * (100 if display_as_percentage else 1),
                    color=colors[i], label=label)
        else:
            ax1.plot(time, rae_param1[i] * (100 if display_as_percentage else 1),
                    color=colors[i], label=label)
    
    ax1.set_ylabel('Relative Absolute Error (%)' if display_as_percentage else 'Relative Absolute Error')
    ax1.set_title(f'{param1_name} Error Comparison')
    ax1.legend()
    ax1.grid(True)
    if ylim is not None:
        ax1.set_ylim(ylim)
    if display_as_percentage:
        ax1.yaxis.set_major_formatter(PercentFormatter())
    
    # Plot parameter 2 errors
    for i in range(num_comparisons):
        mean_rae = np.mean(rae_param2[i])
        label = f'{series_names[i]} (Mean: {mean_rae:.2%})' if display_as_percentage else f'{series_names[i]} (Mean: {mean_rae:.4f})'
        
        if time is None:
            ax2.plot(rae_param2[i] * (100 if display_as_percentage else 1),
                    color=colors[i], label=label)
        else:
            ax2.plot(time, rae_param2[i] * (100 if display_as_percentage else 1),
                    color=colors[i], label=label)
    
    ax2.set_ylabel('Relative Absolute Error (%)' if display_as_percentage else 'Relative Absolute Error')
    ax2.set_title(f'{param2_name} Error Comparison')
    ax2.legend()
    ax2.grid(True)
    if ylim is not None:
        ax2.set_ylim(ylim)
    if display_as_percentage:
        ax2.yaxis.set_major_formatter(PercentFormatter())
    
    # Set common x label if time is provided
    if time is not None:
        ax2.set_xlabel('Time')
    
    # Set overall title if provided
    if title is not None:
        fig.suptitle(title, y=1.02)
    
    plt.tight_layout()
    plt.show()
    
    # Print comprehensive error statistics
    print("\nError Statistics:")
    print("="*50)
    
    print(f"\n{param1_name}:")
    print("-"*30)
    for i in range(num_comparisons):
        mean_err = np.mean(rae_param1[i])
        max_err = np.max(rae_param1[i])
        std_err = np.std(rae_param1[i])
        
        if display_as_percentage:
            print(f"{series_names[i]}:")
            print(f"  Mean Error: {mean_err:.2%}")
            print(f"  Max Error:  {max_err:.2%}")
            print(f"  Std Dev:    {std_err:.2%}")
        else:
            print(f"{series_names[i]}:")
            print(f"  Mean Error: {mean_err:.6f}")
            print(f"  Max Error:  {max_err:.6f}")
            print(f"  Std Dev:    {std_err:.6f}")
        print("-"*30)
    
    print(f"\n{param2_name}:")
    print("-"*30)
    for i in range(num_comparisons):
        mean_err = np.mean(rae_param2[i])
        max_err = np.max(rae_param2[i])
        std_err = np.std(rae_param2[i])
        
        if display_as_percentage:
            print(f"{series_names[i]}:")
            print(f"  Mean Error: {mean_err:.2%}")
            print(f"  Max Error:  {max_err:.2%}")
            print(f"  Std Dev:    {std_err:.2%}")
        else:
            print(f"{series_names[i]}:")
            print(f"  Mean Error: {mean_err:.6f}")
            print(f"  Max Error:  {max_err:.6f}")
            print(f"  Std Dev:    {std_err:.6f}")
        print("-"*30)