import numpy as np
import tensorflow as tf


##### MSE : The Behavioral Loss
class CustomMSE(tf.keras.losses.Loss):
    def __init__(self, name="custom_mse"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def get_config(self):
        return {"name": self.name}


##### Activation Function : Gelu()
class CustomGelu(tf.keras.layers.Layer):
    def __init__(self, alpha=0.5, beta=0.044715, bias=-1., **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.bias = bias
    def call(self, x):
        x_mod = self.alpha * x + self.bias
        return 0.5 * x_mod * ( 1 + tf.math.tanh( tf.sqrt(2.0 / np.pi) * (x_mod + self.beta * tf.pow(x_mod, 3)) ))


##### Activation Function : Elu()
class CustomELU(tf.keras.layers.Layer):
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
    def call(self, x):
        return tf.where(x >= 0, x, self.alpha * (tf.exp(x) - 1))


##### Activation Function : Tanh()
class CustomTanh(tf.keras.layers.Layer):
    def call(self, x):
        return tf.tanh(x)


##### Activation Function : ReLU()
class CustomReLU(tf.keras.layers.Layer):
    def call(self, x):
        return tf.maximum(0.0, x)


##### Activation Function : Sigmoid()
class CustomSigmoid(tf.keras.layers.Layer):
    def call(self, x):
        return tf.sigmoid(x)


def gelu(x, alpha=0.5, beta=0.044715, bias=-1):
    x_mod = alpha * x + bias
    return 0.5 * x_mod * (1 + np.tanh(np.sqrt(2/np.pi) * (x_mod + beta * x_mod**3)))


def elu(x, alpha=1.0):
    """
    Exponential Linear Unit (ELU).
    """
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def tanh(x):
    """
    Hyperbolic tangent.
    """
    return np.tanh(x)


def relu(x):
    """
    Rectified Linear Unit (ReLU).
    """
    return np.maximum(0, x)


def sigmoid(x):
    """
    Sigmoid (logistic) function.
    """
    return 1 / (1 + np.exp(-x))


def normalize(matrix, MAX=False, STD=True):
    # Compute row-wise mean and max
    row_means = matrix.mean(axis=1, keepdims=True)
    epsilon = 1e-8
    
    if MAX: 
        row_maxs = np.abs(matrix).max(axis=1, keepdims=True)
        normalized_matrix = (matrix - row_means) / (row_maxs + epsilon)

    if STD:
        std = np.nanstd(matrix, axis= 1, keepdims=True)
        normalized_matrix = (matrix - row_means) / (std + epsilon)
        
    return normalized_matrix



##### Batching Dataset 
def make_dataset(inputs: np.ndarray,
                 outputs: np.ndarray,
                 batch_size: int,
                 shuffle_buffer: int = 10_000,
                 seed: int = 42) -> tf.data.Dataset:
    """
    Build a `tf.data.Dataset` that:
      * shuffles each epoch (so we don't see the same order forever),
      * batches with `drop_remainder=True` so all batches have an identical
        static shape — this prevents `@tf.function` from retracing,
      * caches in RAM (data is small enough for this RNN task),
      * prefetches AUTOTUNE batches so the CPU stages the next batch while
        the GPU is still computing the current one.
    """
    ds = tf.data.Dataset.from_tensor_slices((inputs, outputs))
    ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.cache()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds



