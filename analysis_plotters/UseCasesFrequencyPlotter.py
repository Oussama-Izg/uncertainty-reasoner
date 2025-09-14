import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from graphviz import Digraph


def format_use_case_name(folder_name: str) -> str:
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


    for col, title, filename in [
        ('total', 'Total', '../analysis_results/use_cases_total_heatmap.png'),
        ('negative', 'Negative', '../analysis_results/use_cases_negative_heatmap.png'),
        ('positive', 'Positive', '../analysis_results/use_cases_positive_heatmap.png')
    ]:
        heatmap_df = results_df.pivot(index='dp_code', columns='issuer_code', values=col).fillna(0)
        plt.figure(figsize=(7, 6))
        ax = sns.heatmap(
            heatmap_df,
            annot=True,
            fmt='d',
            cmap='Blues',
            linewidths=0.5,
            cbar_kws={'label': f'Number of Coins'}
        )
        cbar = ax.collections[0].colorbar
        cbar.set_label(f'Number of Coins', labelpad=20)
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
    analyze_use_cases_split_view_v2('../real_world_usecases')
