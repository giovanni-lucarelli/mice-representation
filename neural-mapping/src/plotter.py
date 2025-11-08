import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D



# 1) colori base per famiglia
FAMILY_COLORS = {
    "Rand":    "#4B0082",
    "INet":    sns.color_palette("tab10")[0],
    "INet-Td": sns.color_palette("tab10")[2],
    "FT":      sns.color_palette("tab10")[1],
}

# 2) ordine delle condizioni dentro la famiglia
COND_ORDER = ["In", "Id", "Iny", "Ir"]

def get_family_and_cond(model_name: str):
    # es: "INet-Td-Id" -> family="INet-Td", cond="Id"
    parts = model_name.split("-")
    if len(parts) == 2:     # es: Rand-In
        return parts[0], parts[1]
    else:                   # es: INet-Td-Id
        return "-".join(parts[:2]), parts[2]

def get_model_color(model_name: str):
    family, cond = get_family_and_cond(model_name)
    base = FAMILY_COLORS[family]

    # genera tante sfumature quante condizioni possibili
    shades = sns.light_palette(base, len(COND_ORDER) + 1, reverse=True).as_hex()[:-1]
    idx = COND_ORDER.index(cond) if cond in COND_ORDER else 0
    return shades[idx]

def nice_label(model_name: str) -> str:
    """
    Converte nomi modello come 'INet-In' o 'INet-Td-Id' in etichette leggibili per le legende.
    """
    label_map = {
        "Rand-In": "Random (no diet)",
        "Rand-Id": "Random (diet)",
        "Rand-Iny": "Random (Nayebi diet)",
        "Rand-Ir": "Random (random diet)",
        "INet-In": "ImageNet (no diet)",
        "INet-Id": "ImageNet (diet)",
        "INet-Iny": "ImageNet (Nayebi diet)",
        "INet-Td-In": "ImageNet TD (no diet)",
        "INet-Td-Id": "ImageNet TD (train+infer diet)",
        "INet-Td-Iny": "ImageNet TD (Nayebi diet)",
    }
    # fallback: restituisci il nome originale se non è nella mappa
    return label_map.get(model_name, model_name)

def plot_comparison(model_dfs: dict[str, pd.DataFrame], metric_name: str):
    # unisco
    dfs = []
    for model_name, df in model_dfs.items():
        df = df.copy()
        df["model"] = model_name
        dfs.append(df)
    combined_scores = pd.concat(dfs, ignore_index=True)

    # ordine layer (come prima)
    try:
        layer_order = sorted(
            combined_scores["layer"].unique(),
            key=lambda x: int("".join([c for c in x if c.isdigit()]) or 0),
        )
    except Exception:
        layer_order = sorted(combined_scores["layer"].unique())

    # palette coerente
    palette = {m: get_model_color(m) for m in model_dfs.keys()}

    g = sns.FacetGrid(
        combined_scores,
        col="area",
        hue="model",
        col_wrap=3,
        height=4,
        aspect=1.2,
        sharey=True,
        palette=palette,
    )

    def plot_with_ribbon(data, **kwargs):
        color = kwargs.get("color", None)
        data = data.set_index("layer").reindex(layer_order).reset_index()
        ax = plt.gca()
        sns.lineplot(data=data, x="layer", y="score", ax=ax, color=color)
        ax.fill_between(
            x=data["layer"],
            y1=data["score"] - data["sem"],
            y2=data["score"] + data["sem"],
            color=color,
            alpha=0.2,
        )

    g.map_dataframe(plot_with_ribbon)

    # legenda costruita a mano
    if g._legend is not None:
        g._legend.remove()

    models_in_plot = list(model_dfs.keys())
    handles = [Line2D([0], [0], color=get_model_color(m), lw=2) for m in models_in_plot]
    labels = [nice_label(m) for m in models_in_plot]

    # metto la legenda in alto, orizzontale
    g.fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(models_in_plot),
        frameon=True,
        title=None,
        fontsize=12,
    )

    # nascondo i tick x per tutte le righe tranne l'ultima
    axes = g.axes.flat
    n_axes = len(axes)
    ncols = g._ncol  # numero di colonne del facet
    nrows = int(np.ceil(n_axes / ncols))

    for i, ax in enumerate(axes):
        if ax is None:
            continue
        # se NON è nell'ultima riga -> togli x
        if i < (nrows - 1) * ncols:
            ax.set_xticklabels([])
            ax.set_xlabel("")
        else:
            # ultima riga: metti xtick orizzontali
            ax.set_xticklabels(layer_order, rotation=0)
            ax.set_xlabel("Layer")

    g.set_titles("{col_name}")
    g.set_axis_labels(None, f"Median {metric_name} Score")

    # lascia spazio in alto per la legenda
    g.fig.subplots_adjust(top=0.88)
    plt.show()




def plot_delta_vs_baseline(scores_to_plot_rsa):
    baseline_key = "Rand-In"
    # tutte le condizioni tranne la baseline
    conditions = [k for k in scores_to_plot_rsa.keys() if k != baseline_key]

    cond_style = {
        cond: {
            "color": get_model_color(cond),
            "label": nice_label(cond)
        }
        for cond in conditions
    }

    layer_order = ["conv1", "conv2", "conv3", "conv4", "conv5"]
    area_order = scores_to_plot_rsa[baseline_key]["area"].drop_duplicates().tolist()

    def diff_vs_baseline(df_cond, df_base):
        m = pd.merge(df_cond, df_base, on=["area", "layer"], suffixes=("", "_base"), how="inner")
        m["diff"] = m["score"] - m["score_base"]
        m["sem_diff"] = np.sqrt(m["sem"]**2 + m["sem_base"]**2)
        return m[["area", "layer", "diff", "sem_diff"]]

    baseline_df = scores_to_plot_rsa[baseline_key]
    plot_df = []
    for cond in conditions:
        d = diff_vs_baseline(scores_to_plot_rsa[cond], baseline_df)
        d["condition"] = cond
        plot_df.append(d)
    plot_df = pd.concat(plot_df, ignore_index=True)

    plot_df["layer"] = pd.Categorical(plot_df["layer"], categories=layer_order, ordered=True)
    plot_df["area"] = pd.Categorical(plot_df["area"], categories=area_order, ordered=True)
    plot_df.sort_values(["area", "layer", "condition"], inplace=True)

    # --- plot ---
    n_areas = len(area_order)
    ncols = 3
    nrows = int(np.ceil(n_areas / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 6.75), sharey=True, sharex=True)
    axes = np.atleast_2d(axes)

    x = np.arange(len(layer_order))
    n_conds = len(conditions)
    bar_width = 0.8 / n_conds
    offsets = (np.arange(n_conds) - (n_conds - 1) / 2) * bar_width

    for i, area in enumerate(area_order):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        df_area = plot_df[plot_df["area"] == area]
        for j, cond in enumerate(conditions):
            dfc = (
                df_area[df_area["condition"] == cond]
                .set_index("layer")
                .reindex(layer_order)
            )
            style = cond_style[cond]
            ax.bar(
                x + offsets[j],
                dfc["diff"].to_numpy(),
                width=bar_width,
                yerr=dfc["sem_diff"].to_numpy(),
                capsize=2,
                label=style["label"],
                color=style["color"],
                error_kw=dict(elinewidth=0.8, capsize=2),
            )
        ax.axhline(0, linewidth=1, color="gray")
        ax.set_title(area)

    # nascondi gli assi x NON dell’ultima riga
    for r in range(nrows - 1):
        for c in range(ncols):
            if r * ncols + c >= n_areas:
                continue
            ax = axes[r, c]
            ax.set_xticklabels([])
            ax.set_xlabel("")

    # ultima riga: metti xticks e etichette dritte
    for c in range(ncols):
        idx = (nrows - 1) * ncols + c
        if idx >= n_areas:
            axes[nrows - 1, c].axis("off")
            continue
        ax = axes[nrows - 1, c]
        ax.set_xticks(x)
        ax.set_xticklabels(layer_order, rotation=0)
        ax.set_xlabel("Layer")
    axes[0, 0].set_ylabel("Delta RSA vs. Rand-In")
    axes[1, 0].set_ylabel("Delta RSA vs. Rand-In")

    # legenda orizzontale in alto
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(cond_style),
        frameon=True,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=12,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()