import numpy as np
import tensorflow as tf
from utils import CustomReLU, CustomMSE 
from model import Model_1





#####################
## Colors & Labels ##
#####################
COLORS = ["#2E8B57", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#808080"]
PAIR = [COLORS[0], COLORS[-1]]
MODEL_LABELS = ("Control", "Constrained")
MODEL_LABELS_SHORT = ("Contr", "Constr")

############################
## Task : Data Parameters ##
############################
NUM_TRAJECTORIES = 5          # K for the main analyses
ANGLE = 20                    # separation between adjacent rays, degrees
TOTAL = 100                   # base length; the phases are derived from it
D1 = 0                        # common straight segment (=> n_straight = 0)
D2 = 10                       # condition-specific arc (=> n_arc = 100)
AUGMENTATION_FACTOR = 13      # DOUBLINGS -> K * 2**13 trials
NOISE_LEVEL = 0.05            # jitter, fraction of max |x|
SIGMA = 3                     # Gaussian smoothing along the time axis
SCALE = 1.0                   # cue spacing: c_k = k * scale
RANDOM_SEED = 42

N_HOLD_ORIGIN = int(TOTAL / 6)                 # 16  -- sampling epoch
N_HOLD_TARGET = int(TOTAL / 6)                 # 16  -- reward epoch
N_STRAIGHT = int(TOTAL * (0.1 * D1))           # 0
N_ARC = int(TOTAL * (0.1 * D2))                # 100 -- movement epoch
NUM_STEPS = N_HOLD_ORIGIN + N_STRAIGHT + N_ARC + N_HOLD_TARGET   # T = 132

######################
## Model Parameters ##
######################
INPUT_DIM = 1
OUTPUT_UNITS = 2
HIDDEN_UNITS = 200            # N
CONNECTIVITY_RATE = 0.4       # p  -> exactly 16 000 active connections
NOISE_AMP = 0.1               # per-step state-noise scale
INIT_SCALE = 0                # h_0 = init_scale * U(-1, 1)  =>  h_0 = 0
ALPHA0, ALPHA1, ALPHA2, ALPHA3, ALPHA4 = 0, 1, 1, 1, 0
LEARNING_RATE = 0.0002
BATCH_SIZE = 2000
REG_A1 = 0.05                 # lambda_1, ||W_in|| + ||W_out||
REG_A2 = 0.005                # lambda_2, mean squared activity
REG_A3 = 0.05                 # lambda_3, ||W_rec||
GRAD_CLIP_NORM = 1.0
BEHAVIORAL_LOSS_THRESHOLD = 0.06     # theta*
MAX_TRAINING_TRIALS = 200_000
LOG_EVERY = 50
NUM_RUNS = 20                 # independent seeds

################################################
## Controls: Alpha Dosing Sweep & Angle Sweep ##
################################################
ALPHAS = [0, 0.2, 0.4, 0.6, 0.8, 1]
KS = [2, 3, 4, 5, 6]
ANGLES = [10, 20, 40, 80]
MODEL_ALPHAS = {
    "Control": 0,
    "Constrained 0.2": 0.2,
    "Constrained 0.4": 0.4,
    "Constrained 0.6": 0.6,
    "Constrained 0.8": 0.8,
    "Constrained 1": 1,
}

#############################################
## Sampling Loss Curves for 2Ways RM ANOVA ##
#############################################
CRITERION_LADDER_START, CRITERION_LADDER_STOP, CRITERION_LADDER_STEP = 0.06, 0.50, 0.01
LEVELS = CRITERION_LADDER_STEP * np.arange(6, 51)



###########################
## Configuration Summary ##
###########################
def settings() -> dict:
    return {
        "task": {
            "K": NUM_TRAJECTORIES,
            "angle_deg": ANGLE,
            "num_steps": NUM_STEPS,
            "n_hold_origin": N_HOLD_ORIGIN,
            "n_movement": N_STRAIGHT + N_ARC,
            "n_hold_target": N_HOLD_TARGET,
            "n_trials": NUM_TRAJECTORIES * (2 ** AUGMENTATION_FACTOR),
            "augmentation_factor": AUGMENTATION_FACTOR,
            "noise_level": NOISE_LEVEL,
            "sigma": SIGMA,
        },
        "model": {
            "hidden_units": HIDDEN_UNITS,
            "connectivity_rate": CONNECTIVITY_RATE,
            "n_active_connections": int(HIDDEN_UNITS * HIDDEN_UNITS * CONNECTIVITY_RATE),
            "d_in": 2 * NUM_STEPS - 1,
            "d_out": OUTPUT_UNITS,
            "noise_amp": NOISE_AMP,
            "init_scale": INIT_SCALE,
            "mixing": (ALPHA1, ALPHA2, ALPHA3),
        },
        "training": {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "reg": (REG_A1, REG_A2, REG_A3),
            "grad_clip_norm": GRAD_CLIP_NORM,
            "loss_threshold": BEHAVIORAL_LOSS_THRESHOLD,
            "max_epochs": MAX_TRAINING_TRIALS,
            "num_runs": NUM_RUNS,
        },
    }


def print_config(config, indent=0):
    for key, value in config.items():
        prefix = " " * indent
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_config(value, indent + 6)
        else:
            print(f"{prefix}{key}: {value}")
    print("")



#####################
## Data Dictionary ##
#####################
KWARGS_DATA= {
    "num_trajectories": NUM_TRAJECTORIES,
    "total_angle_deg": ANGLE * (NUM_TRAJECTORIES - 1),
    "d1": D1,
    "d2": D2,
    "n_straight": N_STRAIGHT,
    "n_arc": N_ARC,
    "n_hold_origin": N_HOLD_ORIGIN,
    "n_hold_target": N_HOLD_TARGET,
    "augmentation_factor": AUGMENTATION_FACTOR,
    "noise_level": NOISE_LEVEL,
    "plot": True,
    "scale": SCALE,
    "random_seed": RANDOM_SEED,
    "shape": "V",
    "sigma": SIGMA,
    "n_show":200,
}


######################
## Model Dictionary ##
######################
KWARGS_MODEL = {
    "input_dim": INPUT_DIM,
    "hidden_units": HIDDEN_UNITS,
    "output_units": OUTPUT_UNITS,
    "activation": CustomReLU(),
    "num_steps": NUM_STEPS,
    "connectivity_rate": CONNECTIVITY_RATE,
    "num_trajectories": NUM_TRAJECTORIES,
    "alpha": 0,
    "data_dtype": tf.float32,
    "batch_size": BATCH_SIZE,
    "noise_amp": NOISE_AMP,
    "init_scale": INIT_SCALE,
    "alpha0": ALPHA0, "alpha1": ALPHA1, "alpha2": ALPHA2,
    "alpha3": ALPHA3, "alpha4": ALPHA4,
    "reg_a1": REG_A1, "reg_a2": REG_A2, "reg_a3": REG_A3,
    "learning_rate": LEARNING_RATE,
    "seed": 42,
}


#####################################
## Load Model & Predict Dictionary ##
#####################################
KWARGS_LOAD_AND_PREDICT = {
    "I": None,
    "alpha": 0,
    "hidden_units": HIDDEN_UNITS,
    "activation": CustomReLU(),
    "num_steps": NUM_STEPS,
    "connectivity_rate": CONNECTIVITY_RATE,
    "K": NUM_TRAJECTORIES,
    "batch_size": BATCH_SIZE,
    "noise_amp": NOISE_AMP,
    "init_scale": INIT_SCALE,
    "alpha0": ALPHA0, "alpha1": ALPHA1, "alpha2": ALPHA2,
    "alpha3": ALPHA3, "alpha4": ALPHA4,
    "LOSS": CustomMSE,
    "learning_rate": LEARNING_RATE,
    "model_name": None,
    "Model_1": Model_1,
}
