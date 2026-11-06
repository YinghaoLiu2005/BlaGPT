import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from zoology.analysis.utils import fetch_wandb_runs


model2color = {
    "Hyena": "#BAB0AC",
    "H3": "#B07AA1",
    
    'Attention': "black", 
    'Sliding window attention': "black", 

    'Based': "#59A14F",
    "Mamba2": "#4E79A7",
    'Gated delta net': "#9C755F", 
    'Rwkv7': "#EDC948", 
    'Mamba': "#76B7B2", 
    'Delta net': "#E15759", 
    'Gla': "#59A14F",

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
    if not isinstance(s, str):
        return s
    s2 = s.strip().replace("_", " ")
    return s2

def _apply_name_replacements(series: pd.Series) -> pd.Series:
    lower_map = {k.lower(): v for k, v in name_replacements.items()}
    return series.apply(lambda x: lower_map.get(_normalize_model_key(x).lower(), _normalize_model_key(x)))

def _mapped_palette(base_palette: dict, replacements: dict) -> dict:
    lower_map = {k.lower(): v for k, v in replacements.items()}
    mapped = {}
    for k, color in base_palette.items():
        key_norm = _normalize_model_key(k)
        final_name = lower_map.get(key_norm.lower(), key_norm)
        mapped[final_name] = color
    return mapped


def extract_num_kv_pairs_columns(df: pd.DataFrame) -> dict:
    """
    Extract num_kv_pairs values from column names.
    Pattern: valid/num_kv_pairs/accuracy-{N}
    
    Returns: dict mapping num_kv_pairs (int) -> column name (str)
    """
    import re
    
    kv_accuracy_cols = {}
    pattern = r'valid/num_kv_pairs/accuracy-(\d+)'
    
    for col in df.columns:
        match = re.search(pattern, col)
        if match:
            num_kv = int(match.group(1))
            kv_accuracy_cols[num_kv] = col
    
    return kv_accuracy_cols


def compute_fact_capacity(
    df: pd.DataFrame,
    accuracy_threshold: float = 0.99,
) -> pd.DataFrame:
    """
    For each (model, state_size), find the maximum num_kv_pairs 
    that achieves accuracy >= threshold.
    """
    kv_accuracy_cols = extract_num_kv_pairs_columns(df)
    
    if not kv_accuracy_cols:
        raise ValueError("Could not find num_kv_pairs accuracy columns. "
                         "Expected pattern: valid/num_kv_pairs/accuracy-{N}")
    
    print(f"Found slice accuracy columns for num_kv_pairs: {sorted(kv_accuracy_cols.keys())}")
    
    results = []
    
    for idx, row in df.iterrows():
        model_name = row.get('model.name', 'unknown')
        state_size = row.get('state_size')
        d_model = row.get('model.d_model', None)
        
        # Find max num_kv_pairs with accuracy >= threshold
        max_kv = 0
        for num_kv in sorted(kv_accuracy_cols.keys()):
            col = kv_accuracy_cols[num_kv]
            if col in df.columns and pd.notna(row[col]):
                acc = row[col]
                if acc >= accuracy_threshold:
                    max_kv = num_kv
        
        results.append({
            'model.name': model_name,
            'state_size': state_size,
            'model.d_model': d_model,
            'max_num_kv_pairs': max_kv,
        })
    
    return pd.DataFrame(results)


def plot_fact_capacity(
    df: pd.DataFrame,
    title: str,
    accuracy_threshold: float = 0.99,
):
    """
    Create a fact capacity plot: for each state size, show the maximum 
    number of KV pairs (facts) that achieve >= accuracy_threshold.
    """
    # Compute fact capacity
    capacity_df = compute_fact_capacity(df, accuracy_threshold)
    capacity_df = capacity_df[capacity_df['max_num_kv_pairs'] > 0].copy()
    capacity_df["model.name"] = capacity_df["model.name"].str.capitalize()
    capacity_df["model.name"] = capacity_df["model.name"].str.replace("-", " ")
    capacity_df["model.name"] = capacity_df["model.name"].str.replace("_", " ")
    capacity_df["Model"] = _apply_name_replacements(capacity_df["model.name"])

    palette = _mapped_palette(model2color, name_replacements)
    list_to_include = capacity_df["Model"].unique()
    plot_df = capacity_df[capacity_df["Model"].isin(list_to_include)]
    filtered_order = [o for o in graph_order if o in list_to_include]
    palette = {k: v for k, v in palette.items() if k in list_to_include}

    # For each (Model, state_size), take the maximum capacity across runs
    plot_df = plot_df.groupby(['Model', 'state_size']).agg({
        'max_num_kv_pairs': 'max'
    }).reset_index()

    g = sns.relplot(
        data=plot_df,
        y="max_num_kv_pairs",
        x="state_size",
        hue="Model",
        kind="line",
        marker="o",
        hue_order=filtered_order,
        height=5,
        aspect=1,
        palette=palette,
        linewidth=0.5,
    )
    g.set(xscale="log", yscale="log", 
          ylabel="Fact Capacity (# KV Pairs)", 
          xlabel="State Size (log scale)")

    ax = g.ax
    ax.set_xlabel("State Size (log scale)", fontsize=16)
    ax.set_ylabel(f"Fact Capacity (# KV Pairs @ {accuracy_threshold:.0%} acc)", fontsize=16)
    if g._legend is not None:
        g._legend.set_title("Model", prop={"size": 16})
    loss = "CE" if "ce" in title else "MSE"
    project = "1d" if "1d" in title else "2d"
    ax.set_title(f"Stacked MQAR Facts - {loss} loss, d_model={project} x embed_dim", fontsize=16)

    # ax = g.ax
    # # Plot optimal capacity line: facts = state_size² (d² facts in d parameters)
    # x_min, x_max = plot_df["state_size"].min(), plot_df["state_size"].max()
    # # Extend slightly beyond data range for visual clarity
    # x_line = np.logspace(np.log10(x_min * 0.8), np.log10(x_max * 1.2), 100)
    # y_optimal = x_line #** 2  # d² facts in d parameters
    # ax.plot(x_line, y_optimal, 'k--', linewidth=0.5, alpha=0.7, label=r"O ($d$ facts)")

    return g


if __name__ == "__main__":

    title = "facts_loss=ce_project=2d_stackedMQAR"
    df = fetch_wandb_runs(
        launch_id=[
            # "default-2026-01-11-12-13-16",

            # sweep across models.
            "default-2026-01-12-00-24-40", # CE; 2x.
            # "default-2026-01-12-00-23-46", # MSE; 2x.
            # "default-2026-01-12-00-21-50", # CE; 1x.
            # "default-2026-01-12-00-21-06", # MSE; 1x.
        ], 
        project_name="0325_zoology"
    )
    plot_fact_capacity(df=df, title=title, accuracy_threshold=0.99)
    plt.savefig(f"{title}.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {title}.png")

