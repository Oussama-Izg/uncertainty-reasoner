import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def extract_categories(filename_stem: str):
    """Parse filename into clean category labels"""
    parts = filename_stem.split('__')
    if len(parts) != 2:
        return "Unknown", "Unknown"

    # Process Depicted Person category
    dp = parts[0].replace('Depicted_Person_', '')
    dp = dp.replace('alternative_',
                    'Multi ') if 'alternative' in dp else 'Single ' + dp
    dp = dp.replace('certain', 'Certain').replace('unCertain', 'Uncertain')

    # Process Issuer category
    iss = parts[1].replace('Issuer_', '')
    iss = iss.replace('alternative_',
                      'Multi ') if 'alternative' in iss else 'Single ' + iss
    iss = iss.replace('certain', 'Certain').replace('unCertain', 'Uncertain')

    return dp, iss

def analyze_use_cases_heatmap(folder_path: str):
    p = Path(folder_path)
    results = []

    for file_path in p.glob('*.csv'):
        try:
            dp_cat, iss_cat = extract_categories(file_path.stem)
            df = pd.read_csv(file_path)

            if 's' in df.columns:
                count = df[df['s'].str.startswith('ex:coin_', na=False)][
                    's'].nunique()
                results.append({
                    'dp_category': dp_cat,
                    'issuer_category': iss_cat,
                    'count': count
                })
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    if not results:
        print("No data processed.")
        return

    # Create analysis dataframe
    results_df = pd.DataFrame(results)
    total_coins = results_df['count'].sum()


    dp_order = ['Single Certain', 'Single Uncertain', 'Multi Certain',
                'Multi Uncertain']
    iss_order = ['Single Certain', 'Single Uncertain', 'Multi Certain',
                 'Multi Uncertain']

    # --- Plot 1: Heatmap of All Combinations ---
    plt.figure(figsize=(8, 7))
    heatmap_data = results_df.pivot_table(
        index='dp_category',
        columns='issuer_category',
        values='count',
        aggfunc='sum',
        fill_value=0
    )
    heatmap_data = heatmap_data.reindex(index=dp_order, columns=iss_order)

    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='d',
        annot_kws={'size': 12},
        cmap='Blues',
        linewidths=0.7,
        cbar_kws={'label': 'Number of Coins'}
    )
    cbar = ax.collections[0].colorbar
    cbar.set_label('Number of Coins', labelpad=20, fontsize=12)

    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    plt.title('Ancient Coins Distribution by Depicted Person and Issuer',
              fontsize=14, fontweight='bold', pad=14, loc='center')
    plt.xlabel('Issuer', fontsize=13, labelpad=18)
    plt.ylabel('Depicted Person', fontsize=13, labelpad=18)

    plt.tight_layout()
    plt.savefig('use_case_heatmap.png', dpi=300)
    plt.show()

    # --- Plot 2: Heatmap Only For Cases With "Uncertain" In Either ---
    uncertain_mask = (
        results_df['dp_category'].str.contains('Uncertain') |
        results_df['issuer_category'].str.contains('Uncertain')
    )
    uncertain_df = results_df[uncertain_mask].copy()

    if not uncertain_df.empty:
        plt.figure(figsize=(8, 7))
        uncertain_heatmap_data = uncertain_df.pivot_table(
            index='dp_category',
            columns='issuer_category',
            values='count',
            aggfunc='sum',
            fill_value=0
        )
        # Keep order but only for present categories
        dp_present = [cat for cat in dp_order if cat in uncertain_heatmap_data.index]
        iss_present = [cat for cat in iss_order if cat in uncertain_heatmap_data.columns]
        uncertain_heatmap_data = uncertain_heatmap_data.reindex(index=dp_present, columns=iss_present)

        ax2 = sns.heatmap(
            uncertain_heatmap_data,
            annot=True,
            fmt='d',
            annot_kws={'size': 12},
            cmap='viridis',
            linewidths=0.7,
            cbar_kws={'label': 'Number of Coins'}
        )
        cbar2 = ax2.collections[0].colorbar
        cbar2.set_label('Number of Coins', labelpad=20, fontsize=12)

        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)

        plt.title(
            'Coins with Uncertainty in Depicted Person or Issuer',
            fontsize=14, fontweight='bold', pad=14, loc='center'
        )
        plt.xlabel('Issuer', fontsize=13, labelpad=18)
        plt.ylabel('Depicted Person', fontsize=13, labelpad=18)

        plt.tight_layout()
        plt.savefig('use_case_heatmap_uncertain.png', dpi=300)
        plt.show()
    else:
        print("No uncertain cases found in the data.")

# Run the analysis
analyze_use_cases_heatmap('real_world_usecases')
