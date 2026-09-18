import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def generate_diverging_trajectories(
    num_trajectories=8,
    total_angle_deg=120,
    d1=10,
    d2=5,
    n_straight=50,
    n_arc=50,
    n_hold_origin=20,
    n_hold_target=20,
    augmentation_factor=10,
    noise_level=0.02,
    plot=True,
    figsize=(18, 6),
    n_show=40,
    random_seed=42,
    scale=0.5, 
    shape='Y', 
    sigma=2
):
    """
    Generate trajectories that:

    1. Stay at origin for a while
    2. Move together vertically
    3. Diverge with different angles
    4. Stay at target for a while

    Returns
    -------
    inputs  : trajectory conditions
    outputs : labels & trajectory coordinates & angles
              shape1 = (N_samples, 1)
              shape2 = (N_samples, N_points, 2)
              shape3 = (N_trajectories, 1)
    """

    #################
    np.random.seed(random_seed)
    angles_deg = np.linspace( - total_angle_deg / 2, total_angle_deg / 2, num_trajectories )
    # if shape == 'Y':
    #     angles_deg = np.linspace( - total_angle_deg / 2, total_angle_deg / 2, num_trajectories )
    # elif shape == 'V':
    #     angles_deg = np.linspace( 0, total_angle_deg, num_trajectories )    
    #################
    trajectories = []
    for angle_deg in angles_deg:
        #################
        angle_rad = np.deg2rad(angle_deg)
        #################
        x0 = np.zeros(n_hold_origin)
        y0 = np.zeros(n_hold_origin)
        #################
        y1 = np.linspace(0, d1, n_straight)
        x1 = np.zeros_like(y1)
        #################
        t = np.linspace(0, 1, n_arc)
        #################
        x2 = d2 * t * np.sin(angle_rad)
        y2 = d1 + d2 * t * np.cos(angle_rad)   
        # if shape == 'Y':
        #     x2 = d2 * t * np.sin(angle_rad)
        #     y2 = d1 + d2 * t * np.cos(angle_rad)    
        # elif shape == 'V':
        #     x2 = d2 * t * np.cos(angle_rad)
        #     y2 = d1 + d2 * t * np.sin(angle_rad)
        
        #################
        x3 = np.ones(n_hold_target) * x2[-1]
        y3 = np.ones(n_hold_target) * y2[-1]
        #################
        x = np.concatenate([x0, x1, x2, x3])
        y = np.concatenate([y0, y1, y2, y3])
        #################
        trajectories.append((x, y))
    #################
    X = np.array([traj[0] for traj in trajectories]) #; print(f'X: {X.shape}')
    Y = np.array([traj[1] for traj in trajectories]) #; print(f'Y: {Y.shape}')
    #################
    labels = np.arange(num_trajectories) * scale
    #################
    data_variance = noise_level * np.max(np.abs(X))
    #################
    for _ in range(augmentation_factor):
        #################
        X = np.concatenate((X, X + ( data_variance * np.random.uniform(-1, 1, size=X.shape) )), axis=0)
        Y = np.concatenate((Y, Y + ( data_variance * np.random.uniform(-1, 1, size=Y.shape) )), axis=0)
        labels = np.concatenate((labels, labels), axis=0)
    #################
    outputs = np.stack((X, Y), axis=2)
    inputs = labels.reshape(-1, 1)
    #################
    outputs = gaussian_filter1d( outputs, sigma=sigma, axis=1 )
    #################
    if plot:
        #################
        time = np.arange(outputs.shape[1])
        #################
        fig, axes = plt.subplots( 1, 3, figsize=figsize, gridspec_kw={'width_ratios': [1, 1, 1]})
        ax = axes[0]
        #################
        for i in range(n_show):
            #################
            ax.plot( outputs[i, :, 0], outputs[i, :, 1], alpha=0.7 )
        #################
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("2D Trajectories")
        #################
        ax = axes[1]
        for i in range(n_show):
            #################
            ax.plot( time, outputs[i, :, 0], alpha=0.7 )
        #################
        ax.axvline(n_hold_origin, color='k', linestyle='--')
        ax.axvline(outputs.shape[1] - n_hold_target, color='r', linestyle='--')
        #################
        ax.axvspan(0, n_hold_origin, color='blue', alpha=0.2, label='Sampling')
        ax.axvspan(n_hold_origin, outputs.shape[1] - n_hold_target, color='gray', alpha=0.2, label='Movement')
        ax.axvspan(outputs.shape[1] - n_hold_target, outputs.shape[1], color='green', alpha=0.2, label='Reward')
        #################
        ax.grid(True)
        ax.set_xlabel("Time")
        ax.set_ylabel("X")
        ax.set_title("X evolution : X(t)")
        ax.legend()
        #################
        ax = axes[2]
        for i in range(n_show):
            #################
            ax.plot( time, outputs[i, :, 1], alpha=0.7 )
        #################
        ax.axvline(n_hold_origin, color='k', linestyle='--')
        ax.axvline(outputs.shape[1] - n_hold_target, color='r', linestyle='--')
        #################
        ax.axvspan(0, n_hold_origin, color='blue', alpha=0.2, label='Sampling')
        ax.axvspan(n_hold_origin, outputs.shape[1] - n_hold_target, color='gray', alpha=0.2, label='Movement')
        ax.axvspan(outputs.shape[1] - n_hold_target, outputs.shape[1], color='green', alpha=0.2, label='Reward')
        #################
        ax.grid(True)
        ax.set_xlabel("Time")
        ax.set_ylabel("Y")
        ax.set_title("Y evolution : Y(t)")
        plt.tight_layout()
        ax.legend()
        plt.show()
    #################
    # print(f'INPUTS  : {inputs.shape}')
    # print(f'OUTPUTS : {outputs.shape}')
    #################
    return inputs, outputs, angles_deg


