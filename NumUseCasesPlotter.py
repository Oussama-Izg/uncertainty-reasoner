import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def format_use_case_name(filename_stem: str) -> str:
    # ... (your unchanged function)
    parts = filename_stem.split('__')
    if len(parts) != 2:
        return filename_stem
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

    # --- PLOT 2: The Detailed "Focus" View ---
    focus_df = results_df.iloc[1:]
    if focus_df.empty:
        print("Only one use case found, cannot create focus plot.")
        return

    plt.figure(figsize=(14, 10))
    barplot_focus = sns.barplot(x='count', y='use_case', data=focus_df, hue='use_case', palette='mako', legend=False, orient='h')
    for bar, label in zip(barplot_focus.patches, focus_df['label']):
        barplot_focus.annotate(
            label,
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            va='center',
            ha='left',
            fontsize=11,
            xytext=(5, 0),
            textcoords='offset points'
        )
    plt.title('Detailed View: Frequency of Less Common Use Cases', fontsize=18, weight='bold', pad=20)
    plt.xlabel('Number of Unique Coins', fontsize=14)
    plt.ylabel('Use Case', fontsize=14)
    plt.tight_layout()
    plt.savefig('use_case_frequency_focus.png', dpi=300, bbox_inches='tight')
    plt.show()

    # --- PLOT 3: Only Uncertain Use Cases ---
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

# --- Run the recommended analysis ---
analyze_use_cases_split_view('real_world_usecases')
