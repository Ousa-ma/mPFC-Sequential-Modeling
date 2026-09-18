import numpy as np
from scipy.stats import pearsonr
import tensorflow as tf
import pandas as pd
import pingouin as pg

def compute_rsm(hidden):
    """
    Parameters
    ----------
    hidden : ndarray
        Shape (conditions, neurons, time)
        Example: (5,300,132)

    Returns
    -------
    RSM : ndarray
        Shape (conditions, conditions)
    """
    n_cond, n_units, _ = hidden.shape

    RSM = np.zeros((n_cond, n_cond))

    for i in range(n_cond):
        for j in range(i, n_cond):

            corr = np.empty(n_units)

            for u in range(n_units):

                corr[u] = np.abs(pearsonr(
                    hidden[i, u],
                    hidden[j, u]
                )[0])

            mean_corr = np.nanmean(corr)

            RSM[i, j] = mean_corr
            RSM[j, i] = mean_corr
    return RSM



def epochs_to_reach(loss_curve, levels):
    epochs = []
    for l in levels:
        idx = np.where(loss_curve <= l)[0]
        if len(idx) == 0:
            epochs.append(np.nan)  # or len(loss_curve)
        else:
            epochs.append(idx[0] + 1)
    return np.array(epochs)



def metric_curves(METRICS, feature, K=5, models=("Control", "Constrained")):
    """
    METRICS -> two lists of per-run curves, for epochs_to_reach_thresholds.
    """
    key = f"K={K}"
    return [list(METRICS[m][feature][key].values()) for m in models]



def metric_matrix(METRICS, feature, K=5, models=("Control", "Constrained")):
    """
    METRICS -> (2, n_runs) array, the shape boxplot_with_lines expects.
    """
    key = f"K={K}"
    return np.array([list(METRICS[m][feature][key].values()) for m in models])



def collect_mae(all_runs, key, field="mae"):
    """
    field = 'mae' or 'mae_shuf'
    key   = 'control_within' or 'control_cross'
    
    """
    return np.array([r[key][field] for r in all_runs])



def flatten_for_pca(activity_3d):
    """
    (n_conditions, n_timesteps, n_units) -> (n_conditions * n_timesteps, n_units).

    """
    #########################
    n_cond, n_time, n_units = activity_3d.shape
    return activity_3d.reshape(n_cond * n_time, n_units)



def average_conditions(hidden, labels):
    labels = tf.squeeze(labels)          # (2000,)
    conds = tf.unique(labels).y          # unique labels
    H = []
    for c in conds:
        mask = tf.equal(labels, c)
        H.append(tf.reduce_mean(tf.boolean_mask(hidden, mask), axis=0))
    return np.asarray(tf.stack(H))



THRESH= 90
def n_comp_90(curve, thresh=THRESH):
    """
    How many PCs are needed to reach 90% cumulative variance?
    Returns np.nan if the threshold is never reached.

    """
    for i in range(len(curve)):
        if curve[i] >= thresh:
            return i + 1          
    return np.nan



def epochs_array(METRICS, model, Ks, N_runs):
    return np.array([
        [METRICS[model]["Epochs"][f"K={K}"][run] for K in Ks]
        for run in range(1, N_runs)
    ])



def correlation_array(METRICS, model, Ks, N_runs):
    return np.array([
        [METRICS[model]["Correlation"][f"K={K}"][run] for K in Ks]
        for run in range(1, N_runs)
    ])



def epochs_dataframe(METRICS, K, MODELS, feature= 'Epochs'):
    """Your data_1w_ANOVA snippet, wrapped so it can be called per K."""
    rows = []
    for mdl in MODELS:
        for run, ep in METRICS[mdl][feature][f'K={K}'].items():
            rows.append({"seed": run, "model": mdl, "epochs": ep})
    return pd.DataFrame(rows)




def adjacent_p_for_K(METRICS, K, MODELS, padjust="holm", feature='Epochs'):
    df = epochs_dataframe(METRICS, K, MODELS, feature=feature)

    aov = pg.rm_anova(data=df, dv="epochs", within="model", subject="seed", detailed=True)
    p_unc_col = "p-unc" if "p-unc" in aov.columns else "p_unc"   # pingouin version quirk
    print(f"K={K}: F={aov.loc[0, 'F']:.2f}, p={aov.loc[0, p_unc_col]:.3g}")

    post = pg.pairwise_tests(data=df, dv="epochs", within="model", subject="seed", padjust=padjust)
    # pingouin only emits "p-corr" when >1 comparison exists to correct for;
    # with exactly 2 model levels present (1 comparison), it's absent and p-unc IS the relevant value
    pcol = "p-corr" if "p-corr" in post.columns else "p-unc"

    pvals = []
    for a, b in zip(MODELS[:-1], MODELS[1:]):
        row = post[((post["A"] == a) & (post["B"] == b)) |
                    ((post["A"] == b) & (post["B"] == a))]
        if row.empty:
            raise ValueError(f"no post-hoc row found for {a} vs {b} at K={K}")
        pvals.append(float(row[pcol].iloc[0]))
    return pvals
