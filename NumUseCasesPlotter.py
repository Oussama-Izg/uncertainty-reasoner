import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from graphviz import Digraph


def format_use_case_name(folder_name: str) -> str:
    # ... (your unchanged function)
    parts = folder_name.split('__')
    if len(parts) != 2:
        return folder_name
    dp_part = parts[0].replace('Depicted_Person_', 'Depicted Person: ')
    if 'alternative' in dp_part:
        dp_part = dp_part.replace('alternative_', 'Multiple ')
    else:
        dp_part = dp_part.replace(': ', ': Single ')
    dp_part = dp_part.replace('certain', 'Certain')
    dp_part = dp_part.replace('unCertain', 'Uncertain')

    iss_part = parts[1].replace('Issuer_', 'Issuer: ')
    if 'alternative' in iss_part:
        iss_part = iss_part.replace('alternative_', 'Multiple ')
    else:
        iss_part = iss_part.replace(': ', ': Single ')
    iss_part = iss_part.replace('certain', 'Certain')
    iss_part = iss_part.replace('unCertain', 'Uncertain')

    return f"{dp_part} / {iss_part}"


def analyze_use_cases_split_view_v2(folder_path: str):
    p = Path(folder_path)
    results = []

    # Go through each use case folder
    for folder in p.iterdir():
        if not folder.is_dir():
            continue

        total_coins = set()
        positive_coins = set()
        negative_coins = set()

        use_case_name = format_use_case_name(folder.name)

        for csv_file in folder.glob('*.csv'):
            try:
                df = pd.read_csv(csv_file)
                if 's' not in df.columns:
                    continue
                coin_ids = set(df[df['s'].str.startswith('ex:coin_', na=False)]['s'].unique())
                if csv_file.name.endswith('__positive.csv'):
                    positive_coins = coin_ids
                elif csv_file.name.endswith('__negative.csv'):
                    negative_coins = coin_ids
                # total_coins is union of both
                total_coins = positive_coins | negative_coins
            except Exception as e:
                print(f"Could not process {csv_file.name}: {e}")

        results.append({
            'use_case': use_case_name,
            'total': len(total_coins),
            'positive': len(positive_coins),
            'negative': len(negative_coins)
        })

    results_df = pd.DataFrame(results).sort_values(by='total', ascending=False).reset_index(drop=True)
    if results_df.empty:
        print("No data processed.")
        return

    # Add split columns for heatmap axes
    split_df = results_df['use_case'].str.split(' / ', expand=True)
    results_df['depicted_person'] = split_df[0]
    results_df['issuer'] = split_df[1]

    # Short code function as before
    def to_short_code(s):
        if "Multiple" in s:
            count = "M"
        else:
            count = "S"
        if "Certain" in s:
            certainty = "C "
        else:
            certainty = "U "
        return f"{count}{certainty}"

    results_df['dp_code'] = results_df['depicted_person'].apply(to_short_code)
    results_df['issuer_code'] = results_df['issuer'].apply(to_short_code)

    # --- HEATMAP 1: Total ---
    for col, title, filename in [
        ('total', 'Total', 'use_case_total_heatmap.png'),
        ('negative', 'Negative', 'use_case_negative_heatmap.png'),
        ('positive', 'Positive', 'use_case_positive_heatmap.png')
    ]:
        heatmap_df = results_df.pivot(index='dp_code', columns='issuer_code', values=col).fillna(0)
        plt.figure(figsize=(7, 6))
        ax = sns.heatmap(
            heatmap_df,
            annot=True,
            fmt='d',
            cmap='Blues',
            linewidths=0.5,
            cbar_kws={'label': f'Number of {title}'}
        )
        cbar = ax.collections[0].colorbar
        cbar.set_label(f'Number of {title}', labelpad=20)
        plt.title(
            f"{title} Usecases",
            fontsize=16, weight='bold', pad=30, loc='center'
        )
        plt.xlabel("Issuer (S=Single, M=Multiple, C=Certain, U=Uncertain)", fontsize=12, labelpad=20)
        plt.ylabel("Depicted Person (S=Single, M=Multiple, C=Certain, U=Uncertain)", fontsize=12, labelpad=20)
        plt.subplots_adjust(top=0.87, bottom=0.18, left=0.23, right=0.98)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()

if __name__ == "__main__":
    analyze_use_cases_split_view_v2('real_world_usecases')


"""
def analyze_use_cases_split_view(folder_path: str):
    p = Path(folder_path)
    csv_files = list(p.glob('*.csv'))
    results = []
    for file_path in csv_files:
        try:
            use_case_name = format_use_case_name(file_path.stem)
            df = pd.read_csv(file_path)
            if 's' not in df.columns:
                continue
            count = df[df['s'].str.startswith('ex:coin_', na=False)]['s'].nunique()
            results.append({'use_case': use_case_name, 'count': count})
        except Exception as e:
            print(f"Could not process {file_path.name}: {e}")

    if not results:
        print("No data processed.")
        return

    results_df = pd.DataFrame(results).sort_values(by='count', ascending=False).reset_index(drop=True)
    results_df['count'] = results_df['count'].fillna(0).astype(int)
    results_df['percent'] = results_df['count'] / results_df['count'].sum() * 100
    results_df['label'] = results_df.apply(lambda row: f"{row['count']} ({row['percent']:.1f}%)", axis=1)

    # --- PLOT 1: The Overall Picture ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 10))
    barplot_all = sns.barplot(x='count', y='use_case', data=results_df, hue='use_case', palette='viridis', legend=False, orient='h')
    # Robust annotation (works everywhere)
    for bar, label in zip(barplot_all.patches, results_df['label']):
        barplot_all.annotate(
            label,
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            va='center',
            ha='left',
            fontsize=11,
            xytext=(5, 0),
            textcoords='offset points'
        )

    plt.title('Overall Frequency of All Use Cases', fontsize=18, weight='bold', pad=20)
    plt.xlabel('Number of Unique Coins', fontsize=14)
    plt.ylabel('Use Case', fontsize=14)
    plt.tight_layout()
    plt.savefig('use_case_frequency_overall.png', dpi=300, bbox_inches='tight')
    plt.show()

    # --- PLOT 2: Only Uncertain Use Cases ---
    uncertain_df = results_df[
        results_df['use_case'].str.contains('Uncertain', case=False,
                                            na=False)].copy()
    if uncertain_df.empty:
        print("No uncertain use cases found, cannot create uncertain plot.")
        return

    plt.figure(figsize=(14, 7))
    barplot_uncertain = sns.barplot(x='count', y='use_case', data=uncertain_df,
                                    palette='flare', orient='h')

    for bar, label in zip(barplot_uncertain.patches, uncertain_df['label']):
        barplot_uncertain.annotate(
            label,
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            va='center',
            ha='left',
            fontsize=11,
            xytext=(5, 0),
            textcoords='offset points'
        )

    plt.title('Frequency of "Uncertain" Use Cases', fontsize=16, weight='bold',
              pad=15)
    plt.xlabel('Number of Unique Coins', fontsize=13)
    plt.ylabel('Uncertain Use Case', fontsize=13)
    plt.tight_layout()
    plt.savefig('use_case_frequency_uncertain_only.png', dpi=300,
                bbox_inches='tight')
    plt.show()

    # --- PLOT 3: heatmap ---
    # Split 'use_case' into two separate columns for heatmap axes
    split_df = results_df['use_case'].str.split(' / ', expand=True)
    results_df['depicted_person'] = split_df[0]
    results_df['issuer'] = split_df[1]

    # Optional: create shorter codes for heatmap axes (recommended for clean visuals)
    def to_short_code(s):
        if "Multiple" in s:
            count = "M "
        else:
            count = "S "
        if "Certain" in s:
            certainty = "C "
        else:
            certainty = "U "
        return f"{count}{certainty}"

    results_df['dp_code'] = results_df['depicted_person'].apply(to_short_code)
    results_df['issuer_code'] = results_df['issuer'].apply(to_short_code)

    heatmap_df = results_df.pivot(index='dp_code', columns='issuer_code',
                                  values='count').fillna(0)

    plt.figure(figsize=(7, 6))
    ax = sns.heatmap(
        heatmap_df,
        annot=True,
        fmt='d',
        cmap='Blues',
        linewidths=0.5,
        cbar_kws={'label': 'Number of Coins'})
    # Access the colorbar and add padding to its label
    cbar = ax.collections[0].colorbar
    cbar.set_label('Number of Coins',
                   labelpad=20)  # Increase labelpad as needed

    plt.title(
        "Ancient Coins Distribution by Depicted Person and Issuer",
        fontsize=16,
        weight='bold',
        pad=30,
        loc='center'# Increase this for more space above the plot
    )
    plt.xlabel(
        "Issuer (S=Single, M=Multiple, C=Certain, U=Uncertain)",
        fontsize=12,
        labelpad=20  # Increase for more space below the x-axis label
    )
    plt.ylabel(
        "Depicted Person (S=Single, M=Multiple, C=Certain, U=Uncertain)",
        fontsize=12,
        labelpad=20  # Increase for more space left of the y-axis label
    )
    plt.subplots_adjust(top=0.87, bottom=0.18, left=0.23, right=0.98)
    plt.tight_layout()
    plt.savefig("use_case_1_heatmap.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    analyze_use_cases_split_view('real_world_usecases')
"""