import tensorflow as tf
from tensorflow.keras import mixed_precision
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from utils import make_dataset




class Model_1(tf.keras.Model):

    """
    Recurrent network with a sparse intra-layer connectivity mask, a
    correlation penalty between trajectory groups, and a closed-loop
    auto-regressive readout. Fully `@tf.function`-friendly.
    """

    # ---------- 1. construction -------------------------------------------
    def __init__(self,
                 input_dim: int = 2,
                 hidden_units: int = 200,
                 num_steps: int = 125,
                 activation  = tf.nn.relu,
                 output_units: int = 2,
                 connectivity_rate: float = 0.2,
                 learning_rate: float = 1e-4,
                 num_trajectories: int = 2,
                 alpha: float = 0.5,
                 data_dtype=tf.float32,
                 batch_size: int = 2000,
                 noise_amp: float = 1e-4,
                 init_scale: float = 0.1,
                 alpha0: float = 0.1,
                 alpha1: float = 0.1,
                 alpha2: float = 0.1,
                 alpha3: float = 0.1,
                 alpha4: float = 0.1,
                 seed: int = 42,
                 use_mixed_precision: bool = False,
                 reg_a1: float = 0.05,
                 reg_a2: float = 0.005,
                 reg_a3: float = 0.05,
                 **kw):
        """
        All hyperparameters used inside the traced step are stored as plain
        Python attributes (so they're captured at trace time as constants)
        or as `tf.Variable` / `tf.constant` tensors when they need to be
        mutable / device-resident.

        `use_mixed_precision=True` switches the global policy to
        `mixed_float16` and wraps Adam in `LossScaleOptimizer`. On Ampere+
        GPUs this often gives a 1.5–2× throughput boost for free.
        """
        super().__init__(**kw)

        if use_mixed_precision:
            mixed_precision.set_global_policy('mixed_float16')

        # ---- static (Python-side) hyperparameters
        self.input_dim         = int(input_dim)
        self.hidden_units      = int(hidden_units)
        self.output_units      = int(output_units)
        self.num_steps         = int(num_steps)
        self.connectivity_rate = float(connectivity_rate)
        self.learning_rate     = float(learning_rate)
        self.num_trajectories  = int(num_trajectories)
        self.data_dtype        = data_dtype
        self.batch_size        = int(batch_size)
        self.noise_amp         = float(noise_amp)
        self.init_scale        = float(init_scale)
        self.seed              = int(seed)

        # ---- mixing coefficients used in `one_step`. They are constants
        # captured at trace time, so they cost nothing on the GPU.
        self.alpha0 = tf.constant(alpha0, dtype=data_dtype)
        self.alpha1 = tf.constant(alpha1, dtype=data_dtype)
        self.alpha2 = tf.constant(alpha2, dtype=data_dtype)
        self.alpha3 = tf.constant(alpha3, dtype=data_dtype)
        self.alpha4 = tf.constant(alpha4, dtype=data_dtype)

        # ---- regularisation weights (also constants in the graph)
        self._reg_a1 = tf.constant(reg_a1, dtype=data_dtype)
        self._reg_a2 = tf.constant(reg_a2, dtype=data_dtype)
        self._reg_a3 = tf.constant(reg_a3, dtype=data_dtype)

        # ---- penalty coefficient (lives on device; can be assigned later)
        self.alpha = tf.Variable(alpha, trainable=False, dtype=data_dtype,
                                 name='penalty_alpha')

        # ---- activation: keep as a callable, NOT a Keras Activation layer,
        # to avoid the layer-call overhead inside `tf.while_loop`.
        if callable(activation) and not isinstance(activation, str):
            self.activation_fn = activation
        else:
            self.activation_fn = tf.keras.activations.get(activation)

        # ---- loss & optimizer
        self.criterion = tf.keras.losses.MeanSquaredError()
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        self.optimizer = (mixed_precision.LossScaleOptimizer(opt)
                          if use_mixed_precision else opt)
        self._use_mp = bool(use_mixed_precision)

        # ---- on-device "best loss so far"
        self.best_loss = tf.Variable(np.inf, trainable=False, dtype=tf.float32, name='best_loss')
        self.penalty_loss = tf.Variable(np.inf, trainable=False, dtype=tf.float32, name='penalty_loss')

    # ---------- 2. parameter creation -------------------------------------
    def build(self, input_shape):
        """
        Allocate the trainable weights and the (non-trainable) connectivity
        mask. The mask is built with `tf.random.shuffle` so it's
        deterministic given `self.seed` and lives in the model's dtype from
        the start (no fp16→fp32 silent cast as in the original).
        """
        init = tf.keras.initializers.VarianceScaling(
            seed=self.seed, scale=0.01, mode='fan_avg',
            distribution='uniform')

        # NOTE: the original `call()` feeds `w_input` a vector of length
        # `input_dim + 2 * (num_steps - 1)` (inputs concatenated with the
        # full output history). We preserve that semantics here.
        full_in = self.input_dim + 2 * (self.num_steps - 1)

        self.w_input  = self.add_weight(
            shape=(full_in, self.hidden_units),
            initializer=init, trainable=True, name='w_input',
            dtype=self.data_dtype)
        self.bias = self.add_weight(
            shape=(self.hidden_units,),
            initializer=init, trainable=True, name='bias',
            dtype=self.data_dtype)
        self.w_intra = self.add_weight(
            shape=(self.hidden_units, self.hidden_units),
            initializer=init, trainable=True, name='w_intra',
            dtype=self.data_dtype)
        self.w_output = self.add_weight(
            shape=(self.hidden_units, self.output_units),
            initializer=init, trainable=True, name='w_output',
            dtype=self.data_dtype)

        # ---- connectivity mask, built in TF (deterministic, on device)
        total = self.hidden_units * self.hidden_units
        num_active = int(total * self.connectivity_rate)
        flat = tf.concat([
            tf.ones([num_active], dtype=self.data_dtype),
            tf.zeros([total - num_active], dtype=self.data_dtype)
        ], axis=0)
        flat = tf.random.shuffle(flat, seed=self.seed)
        mask = tf.reshape(flat, (self.hidden_units, self.hidden_units))

        self.mask = self.add_weight(
            shape=(self.hidden_units, self.hidden_units),
            initializer=tf.constant_initializer(mask.numpy()),
            trainable=False, name='connectivity_mask',
            dtype=self.data_dtype)

        # Precompute the offsets used by `call()` to write the new (x, y)
        # outputs into the flat history vector. These are pure Python ints
        # so they're baked into the graph at trace time.
        self._x_offset = self.input_dim                          # start of x-history
        self._y_offset = self.input_dim + (self.num_steps - 1)   # start of y-history

        super().build(input_shape)

    # ---------- 3. hidden initialisation ----------------------------------
    def init_hidden(self, batch_size: tf.Tensor,
                    init_scale: float = 0.1
                    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Sample an initial hidden state uniformly in [-init_scale, init_scale]
        and return `(hidden_state, activation)`. All-tensor — safe in graph.
        """
        shape = (batch_size, self.hidden_units)
        h = init_scale * tf.random.uniform(shape, minval=-1.0, maxval=1.0,
                                           dtype=self.data_dtype)
        return h, self.activation_fn(h)

    # ---------- 4. single recurrence step ---------------------------------
    def one_step(self,
                 memory_tensor: tf.Tensor,
                 old_hidden_state: tf.Tensor,
                 old_activation: tf.Tensor
                 ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        One update of the recurrent map:

            h_{t+1} = a0·φ(h_t) + a1·h_t + a2·W_in·x_t + b
                    + a3·(W_rec ⊙ M)·h_t + a4·(W_rec ⊙ M)·φ(h_t) + ξ

        Identical math to the original; just type-clean and free of `tf.print`
        side effects.
        """
        batch_size = tf.shape(memory_tensor)[0]
        noise = self.noise_amp * tf.random.normal(
            shape=(batch_size, self.hidden_units), dtype=self.data_dtype)

        masked_W = self.w_intra * self.mask

        new_hidden = (self.alpha0 * old_activation
                      + self.alpha1 * old_hidden_state
                      + self.alpha2 * tf.matmul(memory_tensor, self.w_input)
                      + self.bias
                      + self.alpha3 * tf.matmul(old_hidden_state, masked_W)
                      + self.alpha4 * tf.matmul(old_activation, masked_W)
                      + noise)
        new_activation = self.activation_fn(new_hidden)
        output = tf.matmul(new_activation, self.w_output)
        return new_hidden, new_activation, output

    # ---------- 5. full forward roll-out ----------------------------------
    def call(self, inputs, training=False):
        """
        Roll the recurrence forward for `num_steps` steps using a
        `tf.while_loop`. The full output history that the original code
        carried in `X_in` is now stored implicitly inside the loop's
        `X_flat` carry — only the two scalar positions for the new (x, y)
        are scatter-written each step (instead of rebuilding the whole
        flat tensor via transpose + reshape + concat).

        Returns
        -------
        outputs : (B, T, output_units)
        hidden  : (B, T, hidden_units)
        """
        inputs = tf.cast(inputs, self.data_dtype)
        B = tf.shape(inputs)[0]
        T = self.num_steps
        H = self.hidden_units

        h0, a0 = self.init_hidden(B, self.init_scale)

        # Initial X_flat = [inputs, zeros(2*(T-1))]
        zero_hist = tf.zeros((B, 2 * (T - 1)), dtype=self.data_dtype)
        X_flat0 = tf.concat([inputs, zero_hist], axis=1)

        out_ta = tf.TensorArray(self.data_dtype, size=T,
                                element_shape=tf.TensorShape([None, self.output_units]))
        act_ta = tf.TensorArray(self.data_dtype, size=T,
                                element_shape=tf.TensorShape([None, H]))

        batch_idx = tf.range(B)                       # (B,)
        x_off = tf.constant(self._x_offset, tf.int32)
        y_off = tf.constant(self._y_offset, tf.int32)

        def cond(t, *_):
            return t < T

        def body(t, h, a, X_flat, out_ta, act_ta):
            h, a, y = self.one_step(X_flat, h, a)
            out_ta = out_ta.write(t, y)
            act_ta = act_ta.write(t, a)

            # Write the new output into positions [x_off + t] and
            # [y_off + t] of every row in X_flat — a single scatter that
            # touches 2·B entries (vs. the original full-tensor rebuild).
            def update_xflat():
                # Build (2B, 2) index pairs: (batch, x_off+t) and (batch, y_off+t)
                rows = tf.concat([batch_idx, batch_idx], axis=0)            # (2B,)
                cols = tf.concat([tf.fill([B], x_off + t),
                                  tf.fill([B], y_off + t)], axis=0)         # (2B,)
                idx = tf.stack([rows, cols], axis=1)                        # (2B, 2)
                vals = tf.concat([y[:, 0], y[:, 1]], axis=0)                # (2B,)
                return tf.tensor_scatter_nd_update(X_flat, idx, vals)

            X_flat = tf.cond(t < T - 1, update_xflat, lambda: X_flat)
            return t + 1, h, a, X_flat, out_ta, act_ta

        _, _, _, _, out_ta, act_ta = tf.while_loop(
            cond, body,
            loop_vars=(tf.constant(0, tf.int32), h0, a0, X_flat0, out_ta, act_ta),
            parallel_iterations=1,    # the recurrence is strictly sequential
            maximum_iterations=T)

        outputs = tf.transpose(out_ta.stack(), [1, 0, 2])   # (B, T, out)
        hidden  = tf.transpose(act_ta.stack(), [1, 0, 2])   # (B, T, H)
        return outputs, hidden

    # ---------- 6. correlation penalty (vectorised) -----------------------
    @tf.function(reduce_retracing=True)
    def tf_corr_penalty(self, I: tf.Tensor, hidden: tf.Tensor,
                        eps: float = 1e-8) -> tf.Tensor:
        """
        Mean |correlation| between the per-trajectory mean activations of
        every pair of trajectory groups.

        The original version called `tf.unique` (CPU-only) and looped in
        Python over `num_figs`. Here we bin samples by their input value
        and reduce with `tf.math.unsorted_segment_sum`, then compute all
        pairwise correlations in a single `einsum`. Everything runs on
        the GPU.
        """
        K = self.num_trajectories                        # Python int — graph constant
        num_pairs = self.num_trajectories * (self.num_trajectories - 1) / 2
        I_flat = tf.reshape(I, [-1])
        Imin = tf.reduce_min(I_flat)
        Imax = tf.reduce_max(I_flat)

        # Assign each sample to one of K equal-width bins
        denom = tf.maximum(Imax - Imin, tf.cast(eps, I_flat.dtype))
        bins = tf.cast(
            tf.floor((I_flat - Imin) / denom * tf.cast(K, I_flat.dtype)),
            tf.int32)
        bins = tf.clip_by_value(bins, 0, K - 1)          # (B,)

        H = tf.transpose(hidden, [0, 2, 1])              # (B, U, T)

        sums = tf.math.unsorted_segment_sum(H, bins, K)  # (K, U, T)
        ones = tf.ones_like(I_flat, dtype=H.dtype)
        counts = tf.math.unsorted_segment_sum(ones, bins, K)         # (K,)
        means = sums / (tf.reshape(counts, [K, 1, 1]) + eps)         # (K, U, T)

        # Normalise each (k, unit) trace: subtract temporal mean, divide
        # by max|.| — equivalent to the `tf_normalize(..., MAX=True)` path.
        means = means - tf.reduce_mean(means, axis=2, keepdims=True)
        means = means / (tf.reduce_max(tf.abs(means), axis=2, keepdims=True) + eps)

        # Sort units by the argmax time of group 0 (matches original logic)
        argmax_t = tf.argmax(means[0], axis=1, output_type=tf.int32)
        order = tf.argsort(argmax_t)
        means = tf.gather(means, order, axis=1)                       # (K, U, T)

        # Pairwise channel-wise Pearson correlation, all K·K at once
        m_c = means - tf.reduce_mean(means, axis=2, keepdims=True)
        norms = tf.sqrt(tf.reduce_sum(m_c * m_c, axis=2) + eps)       # (K, U)
        dots  = tf.einsum('kut,jut->kju', m_c, m_c)                   # (K, K, U)
        corr  = dots / (norms[:, None, :] * norms[None, :, :] + eps)  # (K, K, U)
        abs_corr_unit_mean = tf.reduce_mean(tf.abs(corr), axis=2)     # (K, K)

        # Take strict upper triangle only (i < j)
        upper = tf.linalg.band_part(tf.ones((K, K), dtype=H.dtype), 0, -1) \
                - tf.eye(K, dtype=H.dtype)
        return tf.reduce_sum(abs_corr_unit_mean * upper) / num_pairs

    # ---------- 7. the traced training step -------------------------------
    @tf.function(reduce_retracing=True)
    def _train_step(self, I: tf.Tensor, O: tf.Tensor,
                    use_penalty: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        One full optimisation step, compiled into a single graph:
            forward → losses → backward → clip → apply.

        `use_penalty` is a `tf.bool` tensor so we can flip the correlation
        regulariser on/off without retracing. Loss-scaling is handled here
        when mixed precision is on.

        Returns `(loss_total, loss_train, penalty)` — three scalars left on
        the device; sync them at most every `LOG_EVERY` steps.
        
        """
        num_pairs = self.num_trajectories * (self.num_trajectories - 1) / 2
        with tf.GradientTape() as tape:
            P, hidden = self(I, training=True)
            loss_train = self.criterion(O, P)

            reg = (self._reg_a1 * tf.norm(self.w_input)
                   + self._reg_a1 * tf.norm(self.w_output)
                   + self._reg_a2 * tf.reduce_mean(tf.square(hidden))
                   + self._reg_a3 * tf.norm(self.w_intra * self.mask))

            penalty = tf.cond(
                use_penalty,
                lambda: self.tf_corr_penalty(I, hidden),
                lambda: tf.zeros((), dtype=loss_train.dtype))

            loss_total = loss_train + reg + self.alpha * penalty

            if self._use_mp:
                scaled = self.optimizer.get_scaled_loss(loss_total)

        vars_ = self.trainable_variables
        if self._use_mp:
            scaled_grads = tape.gradient(scaled, vars_)
            grads = self.optimizer.get_unscaled_gradients(scaled_grads)
        else:
            grads = tape.gradient(loss_total, vars_)

        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer.apply_gradients(zip(grads, vars_))

        # On-device "best loss" bookkeeping — no host sync needed.
        improved = tf.cast(loss_train, tf.float32) < self.best_loss
        self.best_loss.assign(tf.where(improved, tf.cast(loss_train, tf.float32), self.best_loss))
        
        self.penalty_loss.assign(penalty)
        return loss_total, loss_train, penalty




    # ---------- 8. driver loop --------------------------------------------
    def train(self,
              INPUTS: np.ndarray,
              OUTPUTS: np.ndarray,
              batch_size: int = 2000,
              correlation_penalty: bool = False,
              behavioral_loss_threshold: float = 0.15,
              MAX_TRAINING_TRIALS: int = 50_000,
              LOG_EVERY: int = 200,
              CKPT_EVERY: int = 500,
              ckpt_dir: str = 'ckpts',
              save_name: str = 'Standard_Model') -> Tuple[list, int, np.ndarray]:
        """
        Outer driver:
          * builds the prefetched dataset once,
          * loops over batches,
          * calls the compiled `_train_step`,
          * syncs scalars to the host only every `LOG_EVERY` steps via
            `tf.keras.metrics.Mean` accumulators,
          * checkpoints every `CKPT_EVERY` steps with
            `tf.train.CheckpointManager` (instead of `save_weights` on
            every improvement, which serialised the GPU stream).

        Returns: `(loss_history, n_steps, losses_array)`.
        """
        self.num_trajectories = int(len(np.unique(INPUTS)))

        ds = make_dataset(INPUTS, OUTPUTS, batch_size, seed=self.seed)
        use_penalty_t = tf.constant(bool(correlation_penalty))

        ckpt = tf.train.Checkpoint(model=self, optimizer=self.optimizer)
        mgr  = tf.train.CheckpointManager(ckpt, directory=ckpt_dir,
                                          max_to_keep=3,
                                          checkpoint_name=save_name)

        loss_mean = tf.keras.metrics.Mean()
        pen_mean  = tf.keras.metrics.Mean()
        tot_mean  = tf.keras.metrics.Mean()
        losses_log: list[list[float]] = []
        lc: list[list[float]] = []
        #################
        lt_arr, tt_arr, pn_arr = [], [], []
        #################
        step = 0
        finished = False
        #################
        while not finished:
            for I, O in ds:
                #################
                loss_total, loss_train, penalty = self._train_step(
                    I, O, use_penalty_t)
                #################
                # store tensors directly (GPU friendly)
                lt_arr.append(loss_train)
                tt_arr.append(loss_total)
                pn_arr.append(penalty)
                #################
                loss_mean.update_state(loss_train)
                pen_mean.update_state(penalty)
                tot_mean.update_state(loss_total)

                # ---- logging (single host sync every LOG_EVERY steps)
                if step % LOG_EVERY == 0:
                    lt = float(loss_mean.result().numpy())
                    pn = float(pen_mean.result().numpy())
                    tt = float(tot_mean.result().numpy())
                    bst = float(self.best_loss.numpy())
                    print(f"step={step:>6d}/{MAX_TRAINING_TRIALS} | "
                          f"total={tt:.4f} | loss={lt:.4f} | "
                          f"pen={pn:.4f} | best={bst:.4f} | "
                          f"alpha={float(self.alpha.numpy()):.3f}")
                    losses_log.append([tt, lt, pn])
                    lc.append([[tt, lt, pn]])
                    loss_mean.reset_state()
                    pen_mean.reset_state()
                    tot_mean.reset_state()

                # ---- periodic checkpointing
                if step > 0 and step % CKPT_EVERY == 0:
                    mgr.save()

                # ---- termination
                if step >= MAX_TRAINING_TRIALS or (
                        float(self.best_loss.numpy()) < behavioral_loss_threshold):  # and self.alpha.numpy() == 0.0
                    finished = True
                    break

                step += 1
        #################
        lt_arr = [float(x) for x in lt_arr]
        tt_arr = [float(x) for x in tt_arr]
        pn_arr = [float(x) for x in pn_arr]
        lc= [lt_arr, pn_arr, tt_arr]
        #################
        mgr.save()  # final checkpoint
        self._print_summary(step, MAX_TRAINING_TRIALS, behavioral_loss_threshold)
        return lc, step, np.array(losses_log)
    




    # ---------- 9. one-line training summary ------------------------------
    def _print_summary(self, step, max_steps, thr):
        """Pure-Python pretty print at the end of `train()` (off the hot path)."""
        print('\n------------------------ Training finished ------------------------')
        print(f'  steps           : {step} / {max_steps}')
        print(f'  best loss       : {float(self.best_loss.numpy()):.6f}')
        print(f'  penalty loss    : {float(self.penalty_loss.numpy()):.6f}')
        print(f'  threshold       : {thr}')
        print(f'  hidden_units    : {self.hidden_units}')
        print(f'  num_steps       : {self.num_steps}')
        print(f'  connectivity    : {self.connectivity_rate}')
        print(f'  learning_rate   : {self.learning_rate}')
        print(f'  batch_size      : {self.batch_size}')
        print(f'  alpha (penalty) : {float(self.alpha.numpy()):.4f}')
        print(f'  noise_amp       : {self.noise_amp}')
        print(f'  init_scale      : {self.init_scale}')
        print('  ---- mixing coefficients ----')
        print(f'  alpha0 (old_act) ........ {float(self.alpha0.numpy())}')
        print(f'  alpha1 (old_hidden) ..... {float(self.alpha1.numpy())}')
        print(f'  alpha2 (input) .......... {float(self.alpha2.numpy())}')
        print(f'  alpha3 (W_rec·old_h) .... {float(self.alpha3.numpy())}')
        print(f'  alpha4 (W_rec·old_act) .. {float(self.alpha4.numpy())}')
        print('-------------------------------------------------------------------\n')

    # =====================================================================
    # The plotting / diagnostic methods below stay essentially as they were
    # — they're called outside the hot training loop, so eager execution
    # there is fine. The only changes are: replacing `self.activation`
    # (Keras layer) with `self.activation_fn`, and minor dtype cleanup.
    # =====================================================================

    # ---------- 10. row-wise normalisation --------------------------------
    def tf_normalize(self, matrix, MAX=False, STD=True, eps=1e-8):
        """Row-wise centre, then scale by either max|.| or std. Used only by
        the (now-deprecated) eager penalty path and by the plotters."""
        row_means = tf.reduce_mean(matrix, axis=1, keepdims=True)
        if MAX:
            row_maxs = tf.reduce_max(tf.abs(matrix), axis=1, keepdims=True)
            return (matrix - row_means) / (row_maxs + eps)
        std = tf.math.reduce_std(
            tf.where(tf.math.is_nan(matrix), 0.0, matrix),
            axis=1, keepdims=True)
        return (matrix - row_means) / (std + eps)





    # ---------- 11. diagnostic plotting -----------------------------------
    def plot_activations(self, I, PLOT=False, figsize=(10, 8),
                         save=False, FIG_NAME='Acts.png',
                         MAX=True, STD=False, ax=None,
                         SHOW_MAX=False, SHOW_MIN=False,
                         VLINES=None):
        """
        Group activations by input trajectory and visualise them. Eager,
        diagnostic, not on the training path.
        
        """

        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        unique_values, _ = tf.unique(tf.reshape(I, [-1]))
        num_figs = int(tf.shape(unique_values)[0].numpy())

        Imin, Imax = float(np.min(I)), float(np.max(I))
        threshold1 = (Imax - Imin) / (num_figs - 1)
        threshold2 = (Imax - Imin) / (num_figs + 1)

        ACTIVATIONS = []
        created_fig = False

        if PLOT:
            if ax is None:
                fig, ax = plt.subplots(1, num_figs, figsize=figsize, sharey=True)
                created_fig = True
            else:
                fig = ax[0].figure if np.ndim(ax) else ax.figure
            ax = np.atleast_1d(ax)        


        for i in range(num_figs):
            mask = ((I > (i * threshold1 - threshold2 + Imin)) &
                    (I < (i * threshold1 + threshold2 + Imin)))
            inputs = tf.reshape(I[mask], [-1, 1])
            _, activations = self.predict(inputs, verbose=0)
            mean_activations = np.mean(activations, axis=0)
            ACTIVATIONS.append(mean_activations.T)

            A = self.tf_normalize(mean_activations.T, MAX=MAX, STD=STD).numpy()
            if i == 0:
                sort_order = np.argsort(np.argmax(A, axis=1))
            A_sorted = A[sort_order]

            if PLOT:
                vmin, vmax = np.percentile(A_sorted, 1), np.percentile(A_sorted, 99)
                im = ax[i].imshow(A_sorted, aspect='auto', cmap='viridis',
                                  norm=Normalize(vmin=vmin, vmax=vmax))
                if SHOW_MAX:
                    ax[i].scatter(np.argmax(A_sorted, axis=1),
                                  np.arange(A_sorted.shape[0]),
                                  color='red', marker='o',
                                  edgecolors='white', linewidths=0.5)
                if SHOW_MIN:
                    ax[i].scatter(np.argmin(A_sorted, axis=1),
                                  np.arange(A_sorted.shape[0]),
                                  color='yellow', marker='o',
                                  edgecolors='white', linewidths=0.5)
                fig.colorbar(im, ax=ax[i], fraction=0.14, pad=0.084)
                ax[i].set_title(f'Trajectory {i + 1}', fontsize=10,
                                fontweight='bold', pad=10)
                
                # ---- vertical dashed red lines ----
                if VLINES is not None:
                    for x in VLINES:
                        ax[i].axvline( x=x, color='red', linestyle='--', linewidth=3, alpha=0.9, )

                ax[i].set_xlabel('Time Steps')
                ax[i].set_ylabel('Units')
                ax[i].invert_yaxis()

        if PLOT and created_fig:
            fig.tight_layout()
            if save:
                plt.savefig(FIG_NAME, bbox_inches='tight', dpi=200)
            plt.show()

        return ACTIVATIONS




    # ---------- 12. scatter comparison ------------------------------------
    def scatter_plot_Comparison(self, inputs, trajectories, save=False, ax=None,
                                name_figure='scatter.png', figsize=(30, 10)):
        """Side-by-side scatter of ground-truth vs predicted trajectories."""
        import matplotlib.pyplot as plt
        unique_values, _ = tf.unique(tf.reshape(inputs, [-1]))
        num_figs = int(tf.shape(unique_values)[0].numpy())

        X, Y = trajectories[:, :, 0], trajectories[:, :, 1]
        traj, _ = self.predict(inputs, verbose=0)
        X_pred, Y_pred = traj[:, :, 0], traj[:, :, 1]

        Imin, Imax = float(np.min(inputs)), float(np.max(inputs))
        threshold1 = (Imax - Imin) / (num_figs - 1)
        threshold2 = (Imax - Imin) / (num_figs + 1)

        colors = plt.cm.viridis(np.linspace(0, 1, num_figs))
        created_fig = False
        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            created_fig = True
        else:
            axes = np.atleast_1d(ax)
            fig = axes[0].figure

        for i, color in enumerate(colors):
            mask = ((inputs > (i * threshold1 - threshold2 + Imin)) &
                    (inputs < (i * threshold1 + threshold2 + Imin)))
            mask = tf.reshape(mask, [-1])
            axes[0].scatter(X[mask], Y[mask], color=color, s=20,
                            label=f'Trajectory {i + 1}')
            axes[1].scatter(X_pred[mask], Y_pred[mask], color=color, s=20,
                            label=f'Trajectory {i + 1}')

        handles, labels = axes[0].get_legend_handles_labels()

        if created_fig:
            fig.legend(handles, labels,
                    loc='center left',
                    bbox_to_anchor=(0.85, 0.5),
                    frameon=False)

            fig.subplots_adjust(right=0.85)

        for k, ttl in enumerate(['Original Trajectories', 'Predicted Trajectories']):
            axes[k].set_title(ttl); axes[k].set_xlabel('X')
            axes[k].set_ylabel('Y'); axes[k].grid(True)

        if save:
            fig.savefig(name_figure)

        if created_fig:
            plt.show()


    # ---------- 13. per-trajectory comparison -----------------------------
    def plot_each_trajectory(self, inputs, trajectoires, figsize=(15, 4),
                             save=False, name='trajs.png'):
        """One subplot per trajectory, ground truth + prediction overlaid."""
        import matplotlib.pyplot as plt
        unique_values, _ = tf.unique(tf.reshape(inputs, [-1]))
        num_figs = int(tf.shape(unique_values)[0].numpy())

        X, Y = trajectoires[:, :, 0], trajectoires[:, :, 1]
        traj, _ = self.predict(inputs, verbose=0)
        X_pred, Y_pred = traj[:, :, 0], traj[:, :, 1]

        Imin, Imax = float(np.min(inputs)), float(np.max(inputs))
        threshold1 = (Imax - Imin) / (num_figs - 1)
        threshold2 = (Imax - Imin) / (num_figs + 1)

        colors_true = plt.cm.tab10(np.linspace(0, 1, num_figs))
        colors_pred = plt.cm.Dark2(np.linspace(0, 1, num_figs))
        fig, ax = plt.subplots(1, num_figs, figsize=figsize)
        for i in range(num_figs):
            mask = ((inputs > (i * threshold1 - threshold2 + Imin)) &
                    (inputs < (i * threshold1 + threshold2 + Imin)))
            mask = tf.reshape(mask, [-1])
            ax[i].scatter(X[mask], Y[mask], color=colors_true[i], s=20,
                          label=f'True {i + 1}')
            ax[i].scatter(X_pred[mask], Y_pred[mask], color=colors_pred[i],
                          s=20, label=f'Pred {i + 1}')
            ax[i].set_title(f'Trajectory {i + 1}')
            ax[i].set_xlabel('X'); ax[i].set_ylabel('Y')
            ax[i].grid(True); ax[i].legend()
        if save:
            plt.savefig(name)
        plt.show()





