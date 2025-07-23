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

    issuer_dm = get_domain_knowledge(df, issuer_clean, "ex:inPossibleIssuersOf")


    # Issuing logic (possible issuers)
    depicted_person_rows = df_coin[df_coin["p"] == "nmo:hasPortrait"]
    depicted_person_uncertain = depicted_person_rows['o'].str.contains("uncertain").any()
    depicted_person_clean = depicted_person_rows[~depicted_person_rows['o'].str.contains("uncertain", na=False)]
    depicted_person_count = len(depicted_person_clean)

    if depicted_person_count == 1 and depicted_person_uncertain:
        depicted_person_type = "Depicted Person uncertain"
    elif depicted_person_count == 1 and not depicted_person_uncertain:
        depicted_person_type = "Depicted Person certain"
    elif depicted_person_count > 1 and depicted_person_uncertain:
        depicted_person_type = "Depicted Person alternative uncertain"
    elif depicted_person_count > 1 and not depicted_person_uncertain:
        depicted_person_type = "Depicted Person alternative certain"
    else:
        depicted_person_type = "Depicted Person unknown Usecase"

    depicted_person_dm = get_domain_knowledge(df, depicted_person_clean, "ex:hasPossibleIssuers")

    case_domain_knowledge = pd.concat([depicted_person_dm, issuer_dm])

    case_data = pd.concat([df_coin, case_domain_knowledge])

    # Compose key
    use_case = f"{depicted_person_type} / {issuer_type}"
    return use_case, case_data


def get_domain_knowledge(data, targets, dm_predicate):
    domain_knowledge = pd.DataFrame()
    for target in targets["o"]:
        domain = data[data["s"] == target]
        domain = domain[domain["p"] == dm_predicate]
        domain_knowledge = pd.concat([domain_knowledge, domain], ignore_index=True)

    return domain_knowledge


# Classify each coin
for coin in coins:
    case, case_data = categorize(df_coins, coin)
    use_cases[case].append(case_data)
    print("coin: " + coin + " / case: " + case)

# Export each use case group to CSV
for case, case_all_data in use_cases.items():
    case_all_data_df = pd.concat(case_all_data)

    # Replace spaces and slashes for safe filenames
    filename = case.replace(" / ", "__").replace(" ", "_").replace("/", "_") + ".csv"
    target = "real_world_usecases/"+filename
    case_all_data_df.to_csv(target, index=False)