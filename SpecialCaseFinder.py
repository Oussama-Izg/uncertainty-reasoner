import os
import pandas as pd

# Base directory
base_dir = "real_world_usecases"

# Results
matching_coins = []

# Loop through use case directories
for usecase in os.listdir(base_dir):
    usecase_path = os.path.join(base_dir, usecase)
    if os.path.isdir(usecase_path):
        for file in os.listdir(usecase_path):
            if file.endswith("_negative.csv"):
                file_path = os.path.join(usecase_path, file)
                df = pd.read_csv(file_path, names=["s", "p", "o"])

                # Get all coins
                coins = df[df["p"] == "nmo:hasIssuer"]["s"].unique()

                for coin in coins:
                    # Get all issuers for this coin
                    issuers = \
                    df[(df["s"] == coin) & (df["p"] == "nmo:hasIssuer")][
                        "o"].unique()

                    for issuer in issuers:
                        # Check if issuer is used in ex:inPossibleIssuersOf
                        match = df[(df["s"] == issuer) & (
                                    df["p"] == "ex:inPossibleIssuersOf")]
                        if not match.empty:
                            matching_coins.append((usecase, coin))
                            break  # No need to check other issuers for this coin

# Print results
for usecase, coin in matching_coins:
    print(f"Use case: {usecase}, Coin: {coin}")
