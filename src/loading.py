import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union
from config import *



###########################
## Model PATH Dictionary ##
###########################
KWARGS_MODEL_PATH = {
    "K": NUM_TRAJECTORIES,
    "angle": ANGLE,
    "run": 1,
    "alpha": 0,
    "models_dir": None,
}

def model_path( K: int, angle: float, run: int, alpha, models_dir: Union[str, Path, None] = None ) -> Path:
    """
    Return the path of one trained network's weights.

    """
    if models_dir is None:
        models_dir = Path("Models")
    else:
        models_dir = Path(models_dir)

    model_name = (
        f"Penalized_model_____Num_Trajectories_{K}"
        f"_____Angle_{angle}_____run_{run}_____Alpha_{alpha}.weights.h5"
    )
    return models_dir / model_name



##### Loading Model & Getting Predictions & Activations for The Given Input 
def load_and_predict(
    I,
    hidden_units,
    activation,
    num_steps,
    connectivity_rate,
    K,
    batch_size,
    noise_amp,
    init_scale,
    alpha0,
    alpha1,
    alpha2,
    alpha3,
    alpha4,
    LOSS,
    learning_rate,
    model_name,
    Model_1,
    seed=1,
    alpha=0
):
    #####
    inputs = tf.keras.Input( shape=(1,), name="Scent_Representation" )
    #####
    model = Model_1(
        input_dim=1,
        hidden_units=hidden_units,
        output_units=2,
        activation=activation,
        num_steps=num_steps,
        connectivity_rate=connectivity_rate,
        num_trajectories=K,
        alpha=alpha,
        data_dtype=tf.float32,
        batch_size=batch_size,
        noise_amp=noise_amp,
        init_scale=init_scale,
        alpha0=alpha0,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        alpha4=alpha4,
        seed=seed,
    )
    #####
    model.build(input_shape=(None, 1))
    model.compile(loss=LOSS, optimizer=tf.keras.optimizers.Adam( learning_rate=learning_rate, clipnorm=1.0, ))
    #####
    model_name = Path(model_name)
    model.load_weights(model_name)
    #####
    I_t = tf.cast(I, model.data_dtype)
    preds, hidden_s = model(I_t)
    #####
    return preds, hidden_s, model


