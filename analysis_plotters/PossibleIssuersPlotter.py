import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patheffects as pe
import seaborn as sns


# ---------- helpers ----------
def ensure_columns(df, required):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Found: {list(df.columns)}")


def compute_freq_domain(df: pd.DataFrame) -> pd.DataFrame:
    """Dataset that only (or mainly) contains hasPossibleIssuers facts."""
    ensure_columns(df, ["s", "p", "o"])
    mask = df["p"].astype(str).str.lower().str.contains("haspossibleissuers")
    poss = (
        df.loc[mask, ["s", "o"]]
          .rename(columns={"s": "person", "o": "issuer"})
          .drop_duplicates()
    )
    counts = (
        poss.groupby("person", as_index=False)
            .agg(possible_issuers_count=("issuer", "nunique"))
    )
    freq = (
        counts["possible_issuers_count"]
        .value_counts()
        .sort_index()
        .rename_axis("possible_issuers_per_person")
        .reset_index(name="number_of_persons")
    )
    return freq


def compute_freq_usecase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Specific use case CSV: contains portraits + issuers.
    We take persons that appear as 'o' where p == nmo:hasPortrait,
    then count their hasPossibleIssuers.
    """
    ensure_columns(df, ["s", "p", "o"])
    portraits = (
        df.loc[df["p"] == "nmo:hasPortrait", ["s", "o"]]
          .rename(columns={"s": "coin", "o": "person"})
    )
    depicted_people = set(portraits["person"].unique())

    possible = (
        df.loc[df["p"] == "ex:hasPossibleIssuers", ["s", "o"]]
          .rename(columns={"s": "person", "o": "issuer"})
          .drop_duplicates()
    )
    possible = possible[possible["person"].isin(depicted_people)]

    counts = (
        possible.groupby("person", as_index=False)
                .agg(possible_issuers_count=("issuer", "nunique"))
    )
    freq = (
        counts["possible_issuers_count"]
        .value_counts()
        .sort_index()
        .rename_axis("possible_issuers_per_person")
        .reset_index(name="number_of_persons")
    )
    return freq


def plot_frequency(freq_df: pd.DataFrame, title: str, out_path: Path):
    """
    Professional bar chart for a thesis: neutral palette, readable typography,
    print-friendly, and saved as PNG (300dpi) + PDF (vector).
    """
    if freq_df.empty:
        print(f"[warn] Nothing to plot for {title} — frequency table is empty.")
        return

    # ---- Validate & prep ----
    required = {"possible_issuers_per_person", "number_of_persons"}
    missing = required - set(freq_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df = freq_df.copy()
    df = df.sort_values("possible_issuers_per_person")
    x = df["possible_issuers_per_person"].astype(int)
    y = df["number_of_persons"].astype(int)

    # ---- Global style (print-friendly) ----
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 13,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,     # embed fonts better in PDF
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots()

    # ---- Bars: neutral fill, light edge for print clarity ----
    bars = ax.bar(
        x, y,
        color="steelblue",       # neutral gray (good in B/W)
        edgecolor="#1e1e1e",
        linewidth=0.6
    )

    # ---- Axes & grid ----
    ax.set_title(title, pad=18, weight="bold")
    ax.set_xlabel("Possible issuers per person", labelpad=10)
    ax.set_ylabel("Number of persons", labelpad=10)

    # integer ticks & thousands separator on y
    ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))

    # y-grid only, subtle
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.grid(axis="x", visible=False)

    # remove top/right spines
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # headroom for labels
    ymax = max(y.max(), 1)
    ax.set_ylim(0, ymax * 1.12)

    # ---- Data labels with outline for print contrast ----
    for rect, yi in zip(bars, y):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height(),
            f"{yi:,}",
            ha="center",
            va="bottom",
            fontsize=11,
            # thin white outline so text stays readable on any background
            path_effects=[pe.withStroke(linewidth=2.0, foreground="white")]
        )

    fig.tight_layout()

    # ---- Save: PNG + PDF (vector) ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path.with_suffix('.png')}")


def save_csv(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved csv : {out_path}")


# ---------- main ----------
def main():
    outdir = Path("../analysis_results")

    domain_csv = Path("../triples/domain_knowledge_triples.csv")
    usecase_csv = Path(
        "../real_world_usecases/Depicted_Person_certain__Issuer_certain/Depicted_Person_certain__Issuer_certain__negative.csv")

    # Domain dataset
    df_domain = pd.read_csv(domain_csv)
    df_domain.columns = [c.strip() for c in df_domain.columns]
    freq_domain = compute_freq_domain(df_domain)
    plot_frequency(freq_domain,
                   title="Distribution of Possible Issuers per Depicted Person — Domain Knowledge Dataset",
                   out_path=outdir / "domain_knowledge_frequency.png")

    # Specific use case dataset
    df_usecase = pd.read_csv(usecase_csv)
    df_usecase.columns = [c.strip() for c in df_usecase.columns]
    freq_usecase = compute_freq_usecase(df_usecase)
    plot_frequency(freq_usecase,
                   title="Distribution of Possible Issuers per Depicted Person — Subset of 430 Coins in Negative Case (SC–SC)",
                   out_path=outdir / "domain_knowledge_SC_SC_cases_frequency.png")


if __name__ == "__main__":
    main()