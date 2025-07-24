import pandas as pd
from collections import defaultdict
import os
import shutil

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
    issuer_clean = issuer_rows[
        ~issuer_rows['o'].str.contains("uncertain", na=False)]
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

    issuer_dm = get_domain_knowledge(df, issuer_clean,
                                     "ex:inPossibleIssuersOf")

    extra_dm = get_domain_knowledge_extra(df, issuer_clean)

    # Issuing logic (possible issuers)
    depicted_person_rows = df_coin[df_coin["p"] == "nmo:hasPortrait"]
    depicted_person_uncertain = depicted_person_rows['o'].str.contains(
        "uncertain").any()
    depicted_person_clean = depicted_person_rows[
        ~depicted_person_rows['o'].str.contains("uncertain", na=False)]
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

    depicted_person_dm = get_domain_knowledge(df, depicted_person_clean,
                                              "ex:hasPossibleIssuers")

    case_domain_knowledge = pd.concat([depicted_person_dm, issuer_dm, extra_dm])
    case_data = pd.concat([df_coin, case_domain_knowledge])

    has_domain_knowledge = check_domain_knowledge(case_data)
    if not has_domain_knowledge:
        case_data = pd.DataFrame()

    is_negative = check_negative_case(case_data, case_domain_knowledge)
    # Compose key
    use_case = f"{depicted_person_type} / {issuer_type}"
    use_case += " /negative" if is_negative else " /positive"
    return use_case, case_data


def get_domain_knowledge(data, targets, dm_predicate):
    domain_knowledge = pd.DataFrame()
    for target in targets["o"]:
        domain = data[data["s"] == target]
        domain = domain[domain["p"] == dm_predicate]
        domain_knowledge = pd.concat([domain_knowledge, domain],
                                     ignore_index=True)

    return domain_knowledge


def get_domain_knowledge_extra(data, issuers_data):
    issuers_dm = data[data["p"] == "ex:inPossibleIssuersOf"]
    results = pd.DataFrame()
    for issuer in issuers_data["o"]:
        issuer_dm = issuers_dm[issuers_dm["s"] == issuer]
        result_dm = get_domain_knowledge(data, issuer_dm, "ex:hasPossibleIssuers")
        results = pd.concat([results, result_dm])

    return results





def check_domain_knowledge(coin_data):
    depicted_persons_df = coin_data[coin_data["p"] == "nmo:hasPortrait"]
    depicted_persons_df = depicted_persons_df[depicted_persons_df["o"] != "ex:uncertain"]
    domain_knowledge_df = coin_data[coin_data["p"] == "ex:hasPossibleIssuers"]

    if domain_knowledge_df.empty:
        return False

    for depicted_person in depicted_persons_df["o"]:
        depicted_person_domain_knowledge = domain_knowledge_df[domain_knowledge_df["s"] == depicted_person]
        if depicted_person_domain_knowledge.empty:
            return False

    return True


def check_negative_case(coin_data, case_domain_knowledge):
    if coin_data.empty:
        return True

    dp_domain_knowledge = case_domain_knowledge[
        case_domain_knowledge["p"] == "ex:hasPossibleIssuers"]["o"].values
    coin_issuers = coin_data[coin_data["p"] == "nmo:hasIssuer"]["o"].values

    for coin_issuer in coin_issuers:
        if coin_issuer in dp_domain_knowledge:
            return False

    return True


# Classify each coin
for coin in coins:
    case, case_data = categorize(df_coins, coin)
    if not case_data.empty:
        use_cases[case].append(case_data)
    else:
        continue

# Export each use case group to CSV
for case, case_all_data in use_cases.items():
    case_all_data_df = pd.concat(case_all_data)
    case_all_data_df = case_all_data_df.drop_duplicates()

    # Replace spaces and slashes for safe filenames
    filename = case.replace(" / ", "__").replace(" ", "_").replace("/",
                                                               "_") + ".csv"
    target = "real_world_usecases/" + filename
    case_all_data_df.to_csv(target, index=False)


# --- Post-processing: move each __positive/__negative file to its own folder ---

csv_dir = "real_world_usecases"
csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]

def get_base_name(filename):
    if filename.endswith("__negative.csv"):
        return filename[:-len("__negative.csv")]
    elif filename.endswith("__positive.csv"):
        return filename[:-len("__positive.csv")]
    else:
        return None

for file in csv_files:
    base = get_base_name(file)
    if base:
        folder = os.path.join(csv_dir, base)
        os.makedirs(folder, exist_ok=True)
        src = os.path.join(csv_dir, file)
        dst = os.path.join(folder, file)
        shutil.move(src, dst)
