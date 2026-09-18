import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
from matplotlib.ticker import AutoMinorLocator
from scipy import stats
from scipy.stats import ttest_rel, gaussian_kde
from config import PAIR
from analysis import epochs_to_reach

def plot_sorted_averaged_activations(ax, I, activations, vmin=0.1, vmax=0.9, sorting_ref=0):
    epsilon = 1e-8
    I=I.reshape(-1)
    conditions= np.unique(I)
    ########################
    ######### Sort #########
    idx = I == sorting_ref
    x = activations[idx].mean(axis=0)
    ########################
    x_centered = x - x.mean(axis=1, keepdims=True)
    x_norm = x_centered / ( np.abs(x_centered).max(axis=1, keepdims=True) + epsilon )
    order = np.argsort(np.argmax(x_norm, axis=1))[::-1]
    ########################


    X, Labels= [], []
    for c in conditions:
        ########################
        idx = I == c
        x = activations[idx].mean(axis=0)
        ########################
        x_centered = x - x.mean(axis=1, keepdims=True)
        x_norm = x_centered / ( np.abs(x_centered).max(axis=1, keepdims=True) + epsilon )
        x_norm= x_norm[order]
        ########################
        X.append(x_norm)
        Labels.append(int(c))


    ########################    
    if ax is None :    
        return Labels, np.array(X)
    else:
        fig = ax[0].figure if isinstance(ax, np.ndarray) else ax
        axes = np.atleast_1d(ax)
        
    fig.subplots_adjust(wspace=0.3)
    for i, (x, L, ax) in enumerate(zip(X, Labels, axes)):
        im = ax.imshow(x, 
                       aspect="auto", 
                       cmap="hot", 
                       extent=[0, x.shape[1], 0, x.shape[0]],
                       vmin=vmin,
                       vmax=vmax )
        ax.set_title(f"Cond {L+1}")
        ax.set_xlabel("Time steps")
        if i==0: 
            ax.set_ylabel("Units")
        else:
            ax.set_yticklabels([])
    cbar = fig.colorbar( im, ax=axes, location="right", shrink=0.99, pad=0.02 )
    cbar.set_label("Normalized Activity")


    return Labels, np.array(X)


 



TRACE_LW = 0.5
TRACE_ALPHA = 0.5
cmap=	cc.cm.fire


def condition_colours(inputs, cmap=cmap, cmap_range=(0.15, 0.80)):
    """
    One colour per unique condition label, ordered low to high.
    Returns (labels, {label: colour}).

    """
    labels = np.unique(np.asarray(inputs).squeeze())
    ramp = plt.get_cmap(cmap)
    xs = np.linspace(*cmap_range, len(labels))
    return labels, {lab: ramp(x) for lab, x in zip(labels, xs)}




 
def _condition_slices(inputs, outputs, n_show):
    """
    Yield (colour, row indices) per condition, capped at n_show each.
    
    """
    if inputs is None:
        yield "green", np.arange(min(n_show, len(outputs)))
        return
    lab = np.asarray(inputs).squeeze()
    labels, colours = condition_colours(inputs)
    for c in labels:
        yield colours[c], np.where(lab == c)[0][:n_show]
 
 



def plot_trajectories_2d(ax, outputs, inputs=None,
                         n_show=200, linewidth=TRACE_LW, alpha=TRACE_ALPHA,
                         xlabel="X", ylabel="Y", title=None,
                         use_legend=True, legend_kwargs={"loc": "upper left"}):
    """
    Plotting 2D trajectories, colored by condition.
    outputs : (n_samples, T, 2)
    inputs  : (n_samples, 1) condition labels. 
    legends  : (K,) target legends
    n_show  : trajectories per condition.

    """
 
    for k, (colour, idx) in enumerate(_condition_slices(inputs, outputs, n_show)):
        name = None
        if inputs is not None:
            name = f"C {k + 1}"
        for j, i in enumerate(idx):
            ax.plot(outputs[i, :, 0], outputs[i, :, 1], 
                    color=colour, linewidth=linewidth, 
                    alpha=alpha, label=name if j == 0 else None)
 
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.spines[['right', 'top']].set_visible(False)
    if use_legend :
        if legend_kwargs is None:
            legend_kwargs = {}
        ax.legend( frameon=False, markerscale=2, alignment="left", labelspacing=0.1, handlelength=0.8, **legend_kwargs )
    return ax





def plot_rsm(
    RSM,
    title="Representational Similarity Matrix",
    cmap=cmap,
    vmin=0,
    vmax=1,
    xlabel="Condition",
    ylabel="Condition",
    cbar_label="Mean Pearson correlation",
    condition_labels=None,
    annotate=True,
    figsize=(10, 10),
    dpi=300,
    save_path=None,
    ax=None,
):
    """
    Plot a publication-quality Representational Similarity Matrix (RSM).

    Parameters
    ----------
    RSM : ndarray (N,N)
        Representational similarity matrix.

    title : str
        Figure title.

    cmap : matplotlib colormap
        Defaults to colorcet fire, the raster colormap of the reference style.

    vmin, vmax : float
        Color limits.

    xlabel, ylabel : str
        Axis labels.

    cbar_label : str
        Colorbar label.

    condition_labels : list or None
        Labels shown on x/y axes.
        Default: C1, C2, ...

    annotate : bool
        Display correlation values inside cells.

    figsize : tuple
        Figure size. Used only when the function creates its own figure.

    dpi : int
        Figure resolution. Used only when saving.

    save_path : str or None
        If given, save figure to this path.

    ax : matplotlib axes or None
        Draw into this axes. If None a new figure is created.
    """

    n = RSM.shape[0]

    if condition_labels is None:
        condition_labels = [f"C{i+1}" for i in range(n)]

    owns_figure = ax is None
    if owns_figure:
        fig = plt.figure(tight_layout=True, figsize=figsize)
        ax = fig.add_subplot()
    else:
        fig = ax.figure

    im = ax.imshow(
        RSM,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="none",
        aspect="equal",
    )

    # ---------------------------------------------------
    # ticks
    # ---------------------------------------------------

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))

    ax.set_xticklabels(condition_labels)
    ax.set_yticklabels(condition_labels)

    ax.tick_params(width=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_title(title)

    ax.spines[['right', 'top']].set_visible(False)

    # ---------------------------------------------------
    # Cell values
    # ---------------------------------------------------

    if annotate:

        threshold = (vmin + vmax) / 2

        for i in range(n):
            for j in range(n):

                ax.text(
                    j,
                    i,
                    f"{RSM[i,j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if RSM[i, j] < threshold else "black",
                )

    # ---------------------------------------------------
    # Colorbar
    # ---------------------------------------------------

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.045,
        pad=0.03,
        shrink=0.85,
        aspect=30,
    )

    cbar.set_label(cbar_label)

    cbar.ax.tick_params(width=0.8, length=3)

    cbar.outline.set_linewidth(0.8)

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if owns_figure:
        plt.show()





def boxplot_with_lines(ax,data,xlabel=['data','shuffled'],colours=['green','gray'],ylabel="Reactivation proportion",ymin=0,ymax=1,use_legend=True):
    x1=np.ones(len(data[0]))
    x2=x1+1

    x_line=[1,2]
    for n in range(len(data.T)):
        _=ax.plot(x_line,data.T[n],color="k",linewidth=0.5)
    
    box=ax.boxplot(data.T,showfliers=False,widths=0.5,medianprops=dict(c="k"),patch_artist=True,capwidths=0.5)  
    colors=["white","white"]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        
    ax.scatter(x1,data[0],s=20,c=colours[0],alpha=0.5,edgecolor="k",label=xlabel[0],zorder=3)
    ax.scatter(x2,data[1],s=20,c=colours[1],alpha=0.5,edgecolor="k",label=xlabel[1],zorder=3)
        
    ax.set_ylim(ymin,ymax)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel(ylabel)
    if use_legend==True:
        ax.legend(loc=[0.02,0],frameon=False,markerscale=1,alignment="left", labelspacing=0.1,handlelength=0.1)
        ax.xaxis.set_visible(False)
    else:
        ax.set_xticklabels(xlabel)





def plot_siginficance_symbol_paired(ax,data,stat_res,height=0.1):
    
    bar_position= np.max(data)+ np.max(data)*height
    text_position=1.5
    text_yoffset=bar_position
    
    if stat_res['p']<0.001:
        ax.text(text_position,text_yoffset,"***",ha="center", va="bottom")
        ax.hlines(bar_position,text_position-0.4,text_position+0.4,"k",linewidth=0.8)
        
    elif 0.001<stat_res['p']<0.01:
        ax.text(text_position,text_yoffset,"**",ha="center", va="bottom")
        ax.hlines(bar_position,text_position-0.4,text_position+0.4,"k",linewidth=0.8)
        
    elif 0.01<stat_res['p']<0.05:
       ax.text(text_position,text_yoffset,"*",ha="center", va="bottom")
       ax.hlines(bar_position,text_position-0.4,text_position+0.4,"k",linewidth=0.8)           




rng = np.random.default_rng(0)
def plot_box_comparison_multiple(
        data,
        alpha_values=(0, 0.2, 0.4, 0.6, 0.8, 1.0),
        model_labels=("Control", "Constrained"),
        ylabel="Correlation", xlabel=r"$\alpha$", title="",
        colours=("green", "gray"),
        alpha_palette=None,       # per-alpha colors, used only when M == 1
        ax=None, figsize=None,
        box_width=0.5, dodge=0.40, jitter=0.06,
        ms=20, box_facecolor="white", seed=1,
        ymin=None, ymax=None,
        use_legend=True, connect=False,
        annot=None,               # None | 'p' | 'stars' : per-alpha Wilcoxon (2 models only)
        savepath=None,
        show_significance= False, 
        y_offset_significance= 200,
        legend_args= { "loc": "upper left", "bbox_to_anchor": (0.85, 1), "fontsize": 9 }
    ):
    """Box + jittered scatter of values per alpha, for 1 or 2 models.
 
    Parameters
    ----------
    data : array (n_runs, n_alpha)  OR  list/tuple of 1-2 such arrays.
        Column j holds the n_runs values for alpha_values[j].
        Passing a single array == one model; two arrays == dodged comparison.
    alpha_values : x-axis tick values (length must == n_alpha).
    model_labels : legend labels, one per model.
    colours : scatter colour per model. Boxes stay white.
    alpha_palette : list of colors (one per alpha) for the single-model plot.
        Ignored when two models are given (those colour by model instead).
        None -> a default categorical cycle.
    box_facecolor : face colour for every box. None -> colour by model/alpha.
    connect : link the same run across the two models within each alpha
              (2 models only; requires equal n_runs and row-aligned seeds).
    annot : per-alpha paired Wilcoxon annotation, 2 models only.
 
    Returns
    -------
    dict : {alpha_value: {'wilcoxon': p}} if annot and 2 models, else {}.
    """


    def plot_siginficance_symbol_paired(ax,data,stat_res,height=0.1, text_position=1.5):
        
        bar_position=np.max(data) + y_offset_significance
        # text_position=1.5
        text_yoffset=bar_position
        
        if stat_res['p']<0.001:
            ax.text(text_position,text_yoffset,"***",ha="center", va="bottom")
            ax.hlines(bar_position,text_position-0.25,text_position+0.25,"k",linewidth=0.8)
            
        elif 0.001<stat_res['p']<0.01:
            ax.text(text_position,text_yoffset,"**",ha="center", va="bottom")
            ax.hlines(bar_position,text_position-0.25,text_position+0.25,"k",linewidth=0.8)
            
        elif 0.01<stat_res['p']<0.05:
            ax.text(text_position,text_yoffset,"*",ha="center", va="bottom")
            ax.hlines(bar_position,text_position-0.25,text_position+0.25,"k",linewidth=0.8)           






    # --- normalise to a list of (n_runs, n_alpha) arrays ------------------
    if isinstance(data, np.ndarray) and data.ndim == 2:
        models = [np.asarray(data, float)]
    else:
        models = [np.asarray(d, float) for d in data]
    M = len(models)
    if M not in (1, 2):
        raise ValueError(f"expected 1 or 2 arrays, got {M}")
    A = models[0].shape[1]               # number of alphas
    if any(m.shape[1] != A for m in models):
        raise ValueError("all models must have the same number of alpha columns")
    if len(alpha_values) != A:
        raise ValueError(f"alpha_values has {len(alpha_values)} entries, "
                         f"data has {A} alpha columns")
 
    colours = list(colours)[:M]
    model_labels = list(model_labels)[:M]
    offs = (np.arange(M) - (M - 1) / 2) * (dodge if M > 1 else 0.0)
    box_w = box_width if M > 1 else min(box_width * 1.1, 0.55)
    pos = np.arange(1, A + 1)                    # 1-based, as in the reference
 
    # per-alpha palette (single-model only); fall back to a default cycle
    if alpha_palette is None:
        _default = ["green", "gray", "blue", "m", "k", "orange", "c", "y"]
        alpha_palette = [_default[j % len(_default)] for j in range(A)]
    else:
        alpha_palette = list(alpha_palette)
 
    owns_figure = ax is None
    if owns_figure:
        fig = plt.figure(tight_layout=True,
                         figsize=figsize or (max(0.9 * A, 4.0), 3.0))
        ax = fig.add_subplot()
    else:
        fig = ax.figure
 
    # per-alpha jitter, shared across models so connect lines align
    J = [[rng.uniform(-jitter, jitter, models[m].shape[0]) for j in range(A)]
         for m in range(M)]
 
    # --- connecting lines, drawn under the boxes --------------------------
    if connect and M == 2 and models[0].shape[0] == models[1].shape[0]:
        for j in range(A):                       # within-alpha, across models
            X = np.array([pos[j] + offs[m] + J[m][j] for m in range(M)])
            Y = np.array([models[m][:, j] for m in range(M)])
            _ = ax.plot(X, Y, color="k", linewidth=0.5, zorder=1)
    elif connect and M == 1:                     # across alphas, per run
        m0 = models[0]
        for i in range(m0.shape[0]):             # one line per run
            X = np.array([pos[j] + J[0][j][i] for j in range(A)])
            _ = ax.plot(X, m0[i, :], color="k", linewidth=0.5, zorder=1)
 
    # --- boxes ------------------------------------------------------------
    for m in range(M):
        xs = pos + offs[m]
        box = ax.boxplot([models[m][:, j] for j in range(A)], positions=xs,
                         showfliers=False, widths=box_w, capwidths=box_w,
                         medianprops=dict(c="k"), patch_artist=True, zorder=2)
        for j, patch in enumerate(box["boxes"]):
            c = (box_facecolor if box_facecolor is not None
                 else (alpha_palette[j] if M == 1 else colours[m]))
            patch.set_facecolor(c)

    if show_significance:
        for i in range(A):
            data1= models[0][:, i]
            data2= models[1][:, i]
            data= [data1, data2]
            stat={}
            stat['t'], stat['p'] = ttest_rel(data1, data2)   # PAIRED T-TEST
            plot_siginficance_symbol_paired(ax,data,stat,height=0.1, text_position= i+1)
    # --- scatter ----------------------------------------------------------
    for m in range(M):
        for j in range(A):
            c = alpha_palette[j] if M == 1 else colours[m]
            ax.scatter(pos[j] + offs[m] + J[m][j], models[m][:, j], s=ms,
                       c=c, alpha=0.9, edgecolor="k", zorder=3,
                       label=model_labels[m] if j == 0 else None)
 
    # --- per-alpha stats (2 models only) ----------------------------------
    res = {}
    if annot and M == 2:
        allv = np.concatenate([m.ravel() for m in models])
        span = np.ptp(allv) or 1.0
        top = max(models[m][:, j].max() for m in range(M) for j in range(A))
        ybase = top + 0.06 * span
        for j in range(A):
            a, b = models[0][:, j], models[1][:, j]
            try:
                _, p = stats.wilcoxon(a, b)
            except ValueError:
                p = 1.0
            res[alpha_values[j]] = dict(wilcoxon=p)
            x1, x2 = pos[j] + offs[0], pos[j] + offs[1]
            h = 0.025 * span
            ax.plot([x1, x1, x2, x2],
                    [ybase, ybase + h, ybase + h, ybase],
                    linewidth=0.8, color="k", zorder=4)
            if annot == "stars":
                txt = ("***" if p < 1e-3 else "**" if p < 1e-2
                       else "*" if p < 5e-2 else "n.s.")
            else:
                txt = f"$p$ = {p:.2g}"
            ax.text(pos[j], ybase + h, txt, ha="center", va="bottom", zorder=4)
        ax.set_ylim(top=ybase + 0.16 * span)
 
    # --- cosmetics --------------------------------------------------------
    ax.set_xticks(pos)
    ax.set_xticklabels([a if isinstance(a, str) else f"{a:g}" for a in alpha_values])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xlim(0.5, A + 0.5)
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)
 
    ax.spines[['right', 'top']].set_visible(False)
    if M > 1 and use_legend:
        ax.legend(frameon=False, markerscale=1, alignment="left",
                  labelspacing=0.1, handlelength=0.1, **legend_args)
 
    if savepath:
        fig.savefig(savepath, bbox_inches="tight", transparent=True)
 
    return res





def plot_weight_distribution(ax, aw_c, aw_p, ks_stat, ks_p, save_path=None, figsize=(15, 8), colors= PAIR, legend_kwarg = {
    "loc": "upper right", "bbox_to_anchor": (0.98, 0.98),}):
    """
    Histogram + smoothed density of active connection weights.
    
    """
    lo, hi = np.percentile(np.concatenate([aw_c, aw_p]), [0.5, 99.5])
    bins = np.linspace(lo, hi, 60)
    grid = np.linspace(lo, hi, 200)

    if ax is None :
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        
    ax.hist(aw_c, bins=bins, density=True, alpha=0.6, color=colors[0],
            label="Control", edgecolor="white", lw=0.3)
    ax.hist(aw_p, bins=bins, density=True, alpha=0.6, color=colors[1],
            label="Constrained", edgecolor="white", lw=0.3)
    ax.plot(grid, gaussian_kde(aw_c)(grid), color=colors[0], lw=2.2)
    ax.plot(grid, gaussian_kde(aw_p)(grid), color=colors[1], lw=2.2)

    ax.set_xlabel("active connection weight")
    ax.set_ylabel("density")
    # ax.set_title(f"Weight distribution, active connections\n(KS: D = {ks_stat:.3f}, p = {ks_p:.2f})")
    ax.legend(frameon=False, **legend_kwarg) #fontsize=7)
    if save_path is not None:
        plt.savefig(save_path)
    return fig





def plot_weight_correspondence(
        ax,
        data,
        xlabel="Control weight",
        ylabel="Constrained weight",
        title="Weight-by-weight correspondence",
        colour="green",
        sample_size=6000,
        seed=0,
        use_identity=True,
):
    
    x, y = data
    x, y = np.array(x), np.array(y) 
    lo, hi = np.percentile( np.concatenate([x, y]), [0.5, 99.5] )
    lim = [lo, hi]
    rng = np.random.default_rng(seed)
    idx = rng.choice( len(x), size=min(sample_size, len(x)), replace=False )
    ax.scatter( x[idx], y[idx], s=10, color=colour, alpha=0.35, linewidths=0 )
    
    if use_identity:
        ax.plot( lim, lim, "k--", lw=1.2 )

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12)
    ax.spines[["right", "top"]].set_visible(False)





def plot_box_comparison_multiple_1(
        data,
        alpha_values=(0, 0.2, 0.4, 0.6, 0.8, 1.0),
        model_labels=("Control", "Constrained"),
        ylabel="Correlation", xlabel=r"$\alpha$", title="",
        colors=("#4C72B0", "#C44E52"),
        alpha_palette=None,       # per-alpha colors, used only when M == 1
        ax=None, figsize=None,
        box_width=0.34, dodge=0.40, jitter=0.06,
        ms=22, band_alpha=0.30, seed=0,
        legend=True, connect=False,
        annot=None,                # None | "p" | "stars": one bracket per ADJACENT
                                    # pair of x positions (j, j+1) -- e.g.
                                    # Control-0.2, 0.2-0.4, 0.4-0.6, ... -- stacked
                                    # in a rising staircase so consecutive brackets
                                    # never fuse into one line. Only meaningful for
                                    # the single-array (M == 1) case: A-1 brackets
                                    # total, never the full pairwise matrix.
        adjacent_pvals=None,        # optional, length A-1: precomputed p-values for
                                    # each adjacent pair, e.g. Holm-corrected paired
                                    # t-test results from a repeated-measures ANOVA
                                    # post-hoc. If given, these are drawn as-is and
                                    # no test is recomputed here. If None, a paired
                                    # Wilcoxon signed-rank test is run on the fly for
                                    # each adjacent pair (rows are matched seeds).
        savepath=None,
        legend_args= { "loc": "upper left", "bbox_to_anchor": (0., 1), "fontsize": 9 },
        y_offset_significance=0, DESCENDANT= None
    ):
    if isinstance(data, np.ndarray) and data.ndim == 2:
        models = [np.asarray(data, float)]
    else:
        models = [np.asarray(d, float) for d in data]
    M = len(models) 
    if M < 1:
        raise ValueError(f"expected at least 1 array, got {M}")
    A = models[0].shape[1]; 
    if any(m.shape[1] != A for m in models):
        raise ValueError("all models must have the same number of x columns")
    if len(alpha_values) != A:
        raise ValueError(f"alpha_values has {len(alpha_values)} entries, "
                         f"data has {A} columns")
    if adjacent_pvals is not None:
        if M == 1:
            if len(adjacent_pvals) != A - 1:
                raise ValueError(f"adjacent_pvals needs {A - 1} entries (one per "
                                f"neighbouring x-pair), got {len(adjacent_pvals)}")
        else:
            if len(adjacent_pvals) != A:
                raise ValueError(f"adjacent_pvals needs {A} entries (one list per "
                                f"x position), got {len(adjacent_pvals)}")
            bad = [j for j, pv in enumerate(adjacent_pvals) if len(pv) != M - 1]
            if bad:
                raise ValueError(f"adjacent_pvals[{bad[0]}] needs {M - 1} entries "
                                f"(one per neighbouring model-pair), got "
                                f"{len(adjacent_pvals[bad[0]])}")

        
    colors = list(colors)[:M]
    model_labels = list(model_labels)[:M]
    offs = (np.arange(M) - (M - 1) / 2) * (dodge if M > 1 else 0.0)
    box_w = box_width if M > 1 else min(box_width * 1.6, 0.55)
    rng = np.random.default_rng(seed)

    if alpha_palette is None:
        _default = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
                    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]
        alpha_palette = [_default[j % len(_default)] for j in range(A)]
    else:
        alpha_palette = list(alpha_palette)

    # ctx = JOURNAL_STYLE if "JOURNAL_STYLE" in globals() else {}
    # with plt.rc_context(ctx):
    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize or (max(2 * A, 4.0), 3.0),
            constrained_layout=True)
    else:
        fig = ax.figure

    J = [[rng.uniform(-jitter, jitter, models[m].shape[0]) for j in range(A)]
            for m in range(M)]

    for m in range(M):
        xs = np.arange(A) + offs[m]
        bp = ax.boxplot([models[m][:, j] for j in range(A)], positions=xs,
                        widths=box_w, patch_artist=True, showfliers=False,
                        zorder=2, medianprops=dict(color="black", lw=1.2),
                        whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
        for j, patch in enumerate(bp["boxes"]):
            c = alpha_palette[j] if M == 1 else colors[m]
            patch.set_facecolor(c)
            patch.set_alpha(band_alpha)
            patch.set_edgecolor(c)
            patch.set_linewidth(0.8)

    if connect :  #  and M == 2 and models[0].shape[0] == models[1].shape[0]:
        for j in range(A):
            X = np.array([j + offs[m] + J[m][j] for m in range(M)])
            Y = np.array([models[m][:, j] for m in range(M)])
            ax.plot(X, Y, color="k", linewidth=0.5, zorder=1)
    elif connect and M == 1:
        m0 = models[0]
        n_runs = m0.shape[0]
        for i in range(n_runs):
            X = np.array([j + J[0][j][i] for j in range(A)])
            Y = m0[i, :]
            ax.plot(X, Y, color="0.65", lw=0.6, alpha=0.45, zorder=1)

    for m in range(M):
        for j in range(A):
            c = alpha_palette[j] if M == 1 else colors[m]
            ax.scatter(j + offs[m] + J[m][j], models[m][:, j], s=ms,
                        color=c, alpha=0.85, edgecolor="black",
                        linewidth=0.5, zorder=3,
                        label=model_labels[m] if j == 0 else None)

    # --- adjacent-pair significance brackets (Control-0.2, 0.2-0.4, ...) ---
    res = {}
    if annot and M == 1 and A > 1:
        m0 = models[0]
        xs0 = np.arange(A) + offs[0]
        allv = m0.ravel()
        span = np.ptp(allv) or 1.0
        ybase = m0.max() + 0.3 * span  + y_offset_significance
        step = 0.10 * span
        h = 0.025 * span
        for j in range(A - 1):
            if adjacent_pvals is not None:
                p = adjacent_pvals[j]
            else:
                try:
                    _, p = stats.wilcoxon(m0[:, j], m0[:, j + 1])
                except ValueError:
                    p = 1.0
            res[(alpha_values[j], alpha_values[j + 1])] = dict(p=p)

            x1, x2 = xs0[j], xs0[j + 1]
            y = ybase + j * step          # rising staircase: never fuses
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
                    lw=0.8, color="0.25", zorder=4)
            if annot == "stars":
                txt = ("***" if p < 1e-3 else "**" if p < 1e-2
                        else "*" if p < 5e-2 else "n.s.")
            else:
                txt = f"$p$ = {p:.2g}"
            ax.text((x1 + x2) / 2, y + h, txt, ha="center", va="bottom", fontsize=9, color="k", zorder=4)
        ax.set_ylim(top=ybase + (A - 1) * step + 0.12 * span)

    elif annot and M > 1:
        ###########################
        allv = np.concatenate([m.ravel() for m in models])
        span = np.ptp(allv) or 1.0
        h = 0.02 * span
        step = 0.07 * span
        max_top = 0.0
        ###########################
        for j in range(A):                          # once per K
            ###########################
            local_top = max(models[m][:, j].max() for m in range(M))
            ybase = local_top + 0.05 * span + y_offset_significance
            ###########################
            for m in range(M - 1):                  # adjacent alpha groups within this K
                if adjacent_pvals is not None:
                    p = adjacent_pvals[j][m]
                else:
                    try:
                        _, p = stats.wilcoxon(models[m][:, j], models[m + 1][:, j])
                    except ValueError:
                        p = 1.0
                ###########################
                res[(alpha_values[j], model_labels[m], model_labels[m + 1])] = dict(p=p)
                x1, x2 = j + offs[m], j + offs[m + 1]
                ###########################
                if DESCENDANT is not None:
                    y = ybase - m * step
                else:
                    y = ybase + m * step
                ###########################
                ax.plot([x1, x2], [y + h, y + h], lw=1, color="k", zorder=4)
                if annot == "stars":
                    txt = ("***" if p < 1e-3 else "**" if p < 1e-2
                        else "*" if p < 5e-2 else "")
                else:
                    txt = f"$p$={p:.2g}"
                ###########################
                ax.text((x1 + x2) / 2, y + h, txt, ha="center", va="bottom", fontsize=9, color="k", zorder=4)
                ###########################
            max_top = max(max_top, ybase + (M - 2) * step)
            ###########################
        ax.set_ylim(top=max_top + 0.12 * span)


    ax.set_xticks(np.arange(A))
    ax.set_xticklabels([a if isinstance(a, str) else f"{a:g}" for a in alpha_values])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_xlim(-0.5, A - 0.5)

    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="minor", length=2, width=0.6)
    ax.tick_params(which="major", length=3.5)
    ax.tick_params(axis="x", which="minor", bottom=False)
    ###########################
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # ax.grid(axis="y", lw=0.4, alpha=0.35, zorder=0)
    ###########################
    ax.set_axisbelow(True)
    if M > 1 and legend:
        # ax.legend(frameon=False, **legend_args) # fontsize=6.5)
        ax.legend(frameon=False, markerscale=1, alignment="left", labelspacing=0.1, handlelength=0.1, **legend_args)
    ###########################
    if savepath:
        fig.savefig(savepath, bbox_inches="tight", transparent=True)
    ###########################
    
    return res






def epochs_to_reach_thresholds(ax, Levels, Loss_1, Loss_2,
                               xlabel=['Control', 'Constrained'],
                               colours=['green', 'gray'],
                               ylabel="Epochs to threshold",
                               xlabel_axis="Behavioural loss (MSE)",
                               ymin=None, ymax=None, use_legend=True,
                               errorbar="sem", orient="loss-x",
                               invert_loss=True, Loss_threshold=0.06,
                               show_runs=True, loc=[0.02, 0.75]):
    """
    Mean epochs-to-threshold with an error band, one trace per conditionp.

    Loss_1, Loss_2 : sequences of loss curves, one per run; lengths may differ
    Levels         : loss thresholds, low = strict
    orient         : "loss-x" puts loss on x, "loss-y" swaps the axes
    returns        : Epochs_1, Epochs_2, each (n_runs, n_levels), NaN where a
                     run never reached the level
    
    """
    Levels = np.asarray(Levels, dtype=float)
    Epochs_1 = np.array([epochs_to_reach(tr, Levels) for tr in Loss_1], dtype=float)
    Epochs_2 = np.array([epochs_to_reach(tr, Levels) for tr in Loss_2], dtype=float)

    if ax is None:
        return Epochs_1, Epochs_2


    # resolve the orientation 
    loss_on_x = (orient == "loss-x")
    band = ax.fill_between if loss_on_x else ax.fill_betweenx
    guide = ax.axvline if loss_on_x else ax.axhline
    invert = ax.invert_xaxis if loss_on_x else ax.invert_yaxis

    if Loss_threshold is not None:
        guide(Loss_threshold, color="k", linestyle="dashed", linewidth=0.8, zorder=0)

    for E, colour, label in zip([Epochs_1, Epochs_2], colours, xlabel):
        mu = np.nanmean(E, axis=0)
        n = np.sum(~np.isnan(E), axis=0)
        sd = np.nanstd(E, axis=0, ddof=1)
        err = sd / np.sqrt(np.maximum(n, 1)) if errorbar == "sem" else sd

        if show_runs:
            for run in E:
                x, y = (Levels, run) if loss_on_x else (run, Levels)
                _ = ax.plot(x, y, color=colour, linewidth=0.5, alpha=0.5, zorder=1)

        band(Levels, mu - err, mu + err, facecolor=colour, alpha=0.2, edgecolor=colour, linewidth=0.8, zorder=2)
        x, y = (Levels, mu) if loss_on_x else (mu, Levels)
        
        _ = ax.plot(x, y, color=colour, linewidth=2, zorder=3,
                    solid_capstyle="round",
                    label="%s (n = %d)" % (label, int(np.nanmax(n))))

    ax.set_xlabel(xlabel_axis if loss_on_x else ylabel)
    ax.set_ylabel(ylabel if loss_on_x else xlabel_axis)
    if invert_loss & loss_on_x :
        invert()
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)

    ax.spines[['right', 'top']].set_visible(False)
    if use_legend == True:
        ax.legend(loc=loc, frameon=False, markerscale=1, alignment="left",
                  labelspacing=0.1, handlelength=0.8)

    return Epochs_1, Epochs_2





def plot_mean_sem(data, labels, title="", n_hold_origin= 15, n_hold_target= 15):
    ############################################
    fig, axes = plt.subplots( 1, 2, figsize=(14, 5) )
    dims = ["X", "Y"]
    unique_labels = np.unique(labels)
    time = np.arange(data.shape[1])
    ############################################
    for dim in range(2):
        ############################################
        ax = axes[dim]
        k=0
        for label in unique_labels:
            ############################################
            idx = np.where(labels == label)[0]
            trials = data[idx, :, dim]
            mean_traj = np.mean(trials, axis=0)
            sem_traj = np.std(trials, axis=0)   # 10*( np.std(trials, axis=0) / np.sqrt(trials.shape[0]) )
            ############################################
            ax.plot( time, mean_traj, linewidth=3, label=f"Trajectory {k+1}" )
            ax.fill_between( time, mean_traj - sem_traj, mean_traj + sem_traj, alpha=0.2 )
            k=k+1
        ############################################
        ax.axvline( n_hold_origin, color='k', linestyle='--' )
        ax.axvline( data.shape[1] - n_hold_target, color='r', linestyle='--' )
        ax.axvspan( 0, n_hold_origin, color='blue', alpha=0.1 )
        ax.axvspan( n_hold_origin, data.shape[1] - n_hold_target, color='gray', alpha=0.1 )
        ax.axvspan( data.shape[1] - n_hold_target, data.shape[1], color='green', alpha=0.1 )
        ax.grid(True)
        ax.set_xlabel("Time")
        ax.set_ylabel(dims[dim])
        ax.set_title(f"{title} : {dims[dim]}")
        ax.legend()
        ############################################
    plt.tight_layout()
    plt.show()


