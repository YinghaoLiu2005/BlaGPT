import os

import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from zoology.analysis.utils import fetch_wandb_runs


model2color = {
    "Hyena": "#BAB0AC",
    "H3": "#B07AA1",
    
    'Attention': "black", 
    'Sliding window attention': "black", 

    'Based': "#59A14F", # "#F28E2B"
    "Mamba2": "#4E79A7",
    'Gated delta net': "#9C755F", 
    'Rwkv7': "#EDC948", 
    'Mamba': "#76B7B2", 
    'Delta net': "#E15759", 
    'Gla': "#F28E2B",

    'Ttt linear': "#D3722C",
    'Ttt mlp': "#EDC948",
}

graph_order = [
    "Attention",
    "Sliding Window",
    "H3",
    "Hyena",
    "Based",
    "Mamba",
    "GLA",
    "Mamba-2",
    "DeltaNet",
    "RWKV-7",
    "Gated DeltaNet",
    'TTT Linear',
    'TTT MLP',
]

name_replacements = {
    "Mamba2": "Mamba-2",
    "Gla": "GLA",
    "Rwkv7": "RWKV-7",
    "Mamba": "Mamba",
    "Delta net": "DeltaNet",
    "Gated delta net": "Gated DeltaNet",
    "Sliding window attention": "Sliding Window",
    "Ttt linear": "TTT Linear",
    "Ttt mlp": "TTT MLP",
}

def _normalize_model_key(s: str) -> str:
    """
    Normalize raw model names to match keys in name_replacements.
    We preserve hyphens as given in the raw name (e.g., Mamba2 -> Mamba2),
    but convert underscores to spaces and standardize case for lookup.
    """
    if not isinstance(s, str):
        return s
    s2 = s.strip().replace("_", " ")
    # Don't force hyphen removal—our replacements decide final punctuation.
    # Use case-insensitive lookup by lowercasing.
    return s2

def _apply_name_replacements(series: pd.Series) -> pd.Series:
    # case-insensitive map using provided name_replacements keys
    lower_map = {k.lower(): v for k, v in name_replacements.items()}
    return series.apply(lambda x: lower_map.get(_normalize_model_key(x).lower(), _normalize_model_key(x)))

def _mapped_palette(base_palette: dict, replacements: dict) -> dict:
    """
    Map your existing model2color keys through name_replacements so
    colors line up with the final display names.
    """
    lower_map = {k.lower(): v for k, v in replacements.items()}
    mapped = {}
    for k, color in base_palette.items():
        key_norm = _normalize_model_key(k)
        final_name = lower_map.get(key_norm.lower(), key_norm)
        mapped[final_name] = color
    return mapped


def plot(
    df: pd.DataFrame,
    title: str,
    metric: str="valid/accuracy",
):

    idx = df.groupby(
        ["state_size", "model.name"]
    )[metric].idxmax(skipna=True).dropna()
    plot_df = df.loc[idx]

    # upper case the model names first letter
    plot_df["model.name"] = plot_df["model.name"].str.capitalize()
    plot_df["model.name"] = plot_df["model.name"].str.replace("-", " ")
    plot_df["Model"] = _apply_name_replacements(plot_df["model.name"])

    palette = _mapped_palette(model2color, name_replacements)

    # list to include
    list_to_include = plot_df["Model"].unique()
    list_to_include = [o for o in list_to_include if o != "Sliding Window"]
    plot_df = plot_df[plot_df["Model"].isin(list_to_include)]
    filtered_order = [o for o in graph_order if o in list_to_include]
    palette = {k: v for k, v in palette.items() if k in list_to_include}

    # sns.set_theme(style="whitegrid")
    g = sns.relplot(
        data=plot_df,
        y=metric,
        x="state_size",
        hue="Model",
        kind="scatter",
        marker="o",
        hue_order=filtered_order,           # enforce your order
        height=5,
        aspect=1,
        palette=palette,
        s=60,
        edgecolor="black",    # <-- thin black border
        linewidth=0.5,        # <-- thickness of the border
    )
    g.set(xscale="log", ylabel="Recall Accuracy", xlabel="State Size (log scale)")

    ax = g.ax
    ax.set_xlabel("State Size (log scale)", fontsize=16)
    ax.set_ylabel("Recall Accuracy", fontsize=16)
    # set min and max y to 0 and 1
    # ax.set_ylim(0, 1.)
    if g._legend is not None:
        g._legend.set_title("Model", prop={"size": 16})
    # label the point with the d_model
    for i, row in plot_df.iterrows():
        ax.text(row["state_size"], row[metric], row["model.d_model"], fontsize=12, ha="right", va="bottom")
    #title
    loss = "CE" if "ce" in title else "MSE"
    project = "1d" if "1d" in title else "2d"
    ax.set_title(f"Stacked MQAR - {loss} loss, d_model={project} x embed_dim", fontsize=16)


    ax = g.ax

    try:
        # --- Find the leftmost point (smallest state_size) with accuracy near 1.0 ---
        # tweak tolerance if needed (here: >= 0.99 accuracy)
        point = plot_df.loc[
            plot_df[metric] >= 0.99, ["state_size", metric]
        ].sort_values("state_size").iloc[0]

        x_val, y_val = point["state_size"], point[metric]

        # --- Draw vertical dashed line ---
        ax.axvline(
            x=x_val,
            ymin=0, ymax=y_val,  # scale 0–1 relative to axis
            linestyle="--",
            color="black",
            linewidth=1,
        )
    except Exception as e:
        pass


# You can find the "default-2025..." tags in the wandb UI under the "launch_id" key for a run. 
# Each sweep you launch (with an experiments config file) will have a shared launch_id. 
if __name__ == "__main__" :
    df = fetch_wandb_runs(
        launch_id=[
            # original mqar
            # "default-2024-02-09-05-44-06",
            # "default-2024-02-09-14-59-58",
            # "default-2024-12-28-14-12-35",

            # m01d11y26
            # "default-2026-01-11-12-13-16", # mqar repro. 
            # "default-2026-01-11-12-18-49", # mqar-random-false repro.
            # "default-2026-01-11-12-38-22", # mqar-random-false spherical-emb learnable.
            # "default-2026-01-11-13-30-08", # mqar-continuous spherical-emb non-learnable.

            # "default-2026-01-11-20-28-03", # mqar-continuous spherical-emb non-learnable. 2xd.
            # "default-2026-01-11-20-41-52", # mqar-continuous spherical-emb non-learnable. 2xd. No conv.
            # "default-2026-01-11-20-48-55", # mqar-continuous spherical-emb non-learnable. 2xd. MSE loss.
            # "default-2026-01-11-20-49-18", # mqar-continuous spherical-emb non-learnable. 2xd. MSE loss. No conv.

            # "default-2026-01-11-21-28-47", # mqar-continuous spherical-emb non-learnable. 2xd. Temp 10.0.  
            # "default-2026-01-11-21-30-50", # mqar-continuous spherical-emb non-learnable. 2xd. Temp 1.0. Heads 2.
            # "default-2026-01-11-21-58-29", # mqar-continuous spherical-emb non-learnable. 2xd. Temp 1.0. Heads 2. Layers 1.
            # "default-2026-01-11-22-16-15", # mqar-continuous spherical-emb non-learnable. 2xd. Temp 1.0. Heads 2. Layers 1. No conv. Add Posenc.
            # "default-2026-01-11-22-29-44", # mqar-continuous spherical-emb non-learnable. 2xd. Temp 1.0. Heads 2. Layers 1. No conv. Add Posenc. Torch eye.
            # "default-2026-01-11-22-59-12", # mqar-continuous spherical-emb non-learnable. 2xd. Temp 1.0. Heads 2. Layers 1. Torch eye. Adjust data.
            # "default-2026-01-11-23-18-47" # 1 layer, continuous, no 2d model.

            # sweep across models.
            # "default-2026-01-12-00-24-40", # CE; 2x.
            # "default-2026-01-12-00-23-46", # MSE; 2x.
            "default-2026-01-12-00-21-50", # CE; 1x.
            # "default-2026-01-12-00-21-06", # MSE; 1x.

        ], 
        project_name="0325_zoology"
    )

    title = "loss=ce_proj=1d_stackedMQAR"

    plot(df=df, title=title)

    # save in high resolution
    plt.savefig(f"{title}.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {title}.png")


