import pandas as pd
from collections import defaultdict

# Load the CSV or define your DataFrame

df = pd.read_csv("triples/all_triples.csv")

# Only coins will be considered (start with 'ex:coin_')
df_coins = df[df['s'].str.startswith("ex:coin_")]

# Group by coin
coins = df_coins['s'].unique()

# Initialize dictionary for use cases
use_cases = defaultdict(list)


# Helper function to determine category
def categorize(coins_df, coin):
    df_coin = coins_df[coins_df["s"] == coin]

    # Issuer logic
    issuer_rows = df_coin[df_coin["p"] == "nmo:hasIssuer"]
    issuer_uncertain = issuer_rows['o'].str.contains("uncertain").any()
    issuer_clean = issuer_rows[~issuer_rows['o'].str.contains("uncertain", na=False)]
    issuer_count = len(issuer_clean)

    if issuer_count == 1 and issuer_uncertain:
        issuer_type = "Issuer uncertain"
    elif issuer_count == 1 and not issuer_uncertain:
        issuer_type = "Issuer certain"
    elif issuer_count > 1 and issuer_uncertain:
        issuer_type = "Issuer alternative uncertain"
    elif issuer_count > 1 and not issuer_uncertain:
        issuer_type = "Issuer alternative certain"
    else:
        issuer_type = "Issuer unknown Usecase"

    print(
        f"Issuers total: {len(issuer_rows)} | Certain: {issuer_count} | Uncertain found: {issuer_uncertain}")


    # Issuing logic (possible issuers)
    person_rows = df[(df["p"] == "ex:hasPossibleIssuers") & (df["o"].isin(issuer_rows['o']))]
    coin_in_person = df[(df["p"] == "ex:hasPossibleIssuers") & (df["o"].isin(issuer_rows['o']))]
    possible_rows = df[(df["p"] == "ex:hasPossibleIssuers") & (df["s"].str.startswith("ex:coin_"))]
    coin_possible = possible_rows[possible_rows["s"] == coin]
    possible_uncertain = coin_possible['o'].str.contains("uncertain").any()
    possible_count = len(coin_possible[~coin_possible['o'].str.contains("uncertain", na=False)])

    if possible_count == 1:
        issuing_type = "Issuing for certain"
    elif possible_count > 1:
        issuing_type = "Issuing for alternative certain"
    else:
        issuing_type = "Issuing for uncertain" if possible_uncertain else "Issuing unknown"

    if possible_uncertain and issuing_type != "Issuing for uncertain":
        issuing_type = "Issuing for alternative uncertain"

    # Compose key
    use_case = f"{issuing_type} / {issuer_type}"
    return use_case

# Classify each coin
for coin in coins:
    case = categorize(df_coins, coin)
    use_cases[case].append(coin)

# Export each use case group to CSV
for case, coins in use_cases.items():
    df_case = df_coins[df_coins["s"].isin(coins)]
    # Replace spaces and slashes for safe filenames
    filename = case.replace(" / ", "__").replace(" ", "_").replace("/", "_") + ".csv"
    df_case.to_csv(filename, index=False)
