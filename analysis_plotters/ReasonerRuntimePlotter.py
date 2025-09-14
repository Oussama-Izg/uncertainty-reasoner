import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple, List
import time

import SparqlConnector
import Reasoner


pd.set_option("display.max_rows", None)


def define_axioms():
    """Return (the different kinds of the reasoning pipeline"""
    axioms1 = [
        Reasoner.CertaintyAssignmentAxiom("nmo:hasIssuer"),
        Reasoner.CertaintyAssignmentAxiom("nmo:hasPortrait"),
        Reasoner.CertaintyAssignmentAxiom("ex:hasPossibleIssuers"),
        Reasoner.CertaintyAssignmentAxiom("ex:inPossibleIssuersOf"),
        Reasoner.AFEDempsterShaferAxiom_2(
            target_predicate="nmo:hasIssuer",
            knowledge_path_predicate="nmo:hasPortrait",
            domain_knowledge_predicate="ex:hasPossibleIssuers",
            group="1",
            target_ignorance=0.2,
            domain_knowledge_ignorance=0.2,
        ),
        Reasoner.AFEDempsterShaferAxiom_2(
            target_predicate="nmo:hasPortrait",
            knowledge_path_predicate="nmo:hasIssuer",
            domain_knowledge_predicate="ex:inPossibleIssuersOf",
            group="1",
            target_ignorance=0.2,
            domain_knowledge_ignorance=0.2,
        ),
    ]

    axioms2 = [
        Reasoner.CertaintyAssignmentAxiom("nmo:hasIssuer"),
        Reasoner.CertaintyAssignmentAxiom("nmo:hasPortrait"),
        Reasoner.CertaintyAssignmentAxiom("ex:hasPossibleIssuers"),
        Reasoner.CertaintyAssignmentAxiom("ex:inPossibleIssuersOf"),
        Reasoner.AFEDempsterShaferAxiom_2(
            target_predicate="nmo:hasIssuer",
            knowledge_path_predicate="nmo:hasPortrait",
            domain_knowledge_predicate="ex:hasPossibleIssuers",
            target_ignorance=0.2,
            domain_knowledge_ignorance=0.2,
        ),
        Reasoner.AFEDempsterShaferAxiom_2(
            target_predicate="nmo:hasPortrait",
            knowledge_path_predicate="nmo:hasIssuer",
            domain_knowledge_predicate="ex:inPossibleIssuersOf",
            target_ignorance=0.2,
            domain_knowledge_ignorance=0.2,
        ),
    ]
    return axioms1, axioms2


def slice_by_unique_coins(
    df: pd.DataFrame,
    n: int,
    col: str = "s",
    coin_marker: str = "coin_"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Return a DataFrame that contains:
      - all rows whose subject is NOT a coin, plus
      - all rows whose subject is one of the first N unique coins.

    'Coin' subjects are detected by checking whether the subject string contains `coin_marker`.
    Multiple rows for the same coin count as 1 coin.

    Returns:
      df_slice: the filtered DataFrame
      selected_coins: list of selected coin subjects (unique)
    """
    s = df[col].astype(str)
    is_coin_row = s.str.startswith(coin_marker)
    unique_coins = s[is_coin_row].drop_duplicates().tolist()

    # Take the first N coins
    selected_coins = unique_coins[:min(n, len(unique_coins))]

    keep_mask = (~is_coin_row) | (s.isin(selected_coins))
    df_slice = df.loc[keep_mask].reset_index(drop=True)

    return df_slice, selected_coins


def run_benchmark():
    QUERY_ENDPOINT = "http://localhost:3030/test/query"
    # Update endpoint
    UPDATE_ENDPOINT = "http://localhost:3030/test/update"
    # Graph Store Protocol endpoint
    GSP_ENDPOINT = "http://localhost:3030/test/data"

    prefixes = {"nmo": "http://nomisma.org/ontology#"}

    conn = SparqlConnector.ReificationSparqlConnector(QUERY_ENDPOINT,
                                                      UPDATE_ENDPOINT,
                                                      GSP_ENDPOINT,
                                                      prefixes=prefixes)

    df = pd.read_csv(
        "../real_world_usecases/Depicted_Person_certain__Issuer_certain/Depicted_Person_certain__Issuer_certain__positive.csv")

    num_coins = list(range(50, 401, 50))

    axioms1, axioms2 = define_axioms()

    results = dict()

    for n in num_coins:
        print(f"Test for {n} coins and axiom1 is running")
        results[n] = {"parallel": 0, "successive": 0}
        df_slice = slice_by_unique_coins(df, n)[0]

        conn.upload_df(df_slice)

        reasoner = Reasoner.Reasoner(axioms1)
        reasoner.load_data_from_endpoint(conn)
        t0 = time.perf_counter()
        reasoner.reason()
        t1 = time.perf_counter()
        runTime = t1 - t0
        results[n]["parallel"] = runTime


        conn.delete_query(delete_all=True)
        conn.upload_df(df_slice)
        print(f"Test for {n} coins and axiom2 is running")

        reasoner = Reasoner.Reasoner(axioms2)
        reasoner.load_data_from_endpoint(conn)
        t0 = time.perf_counter()
        reasoner.reason()
        t1 = time.perf_counter()
        runTime = t1 - t0
        results[n]["successive"] = runTime

        print(results)

    return results


def plot_results(results: dict,
                 out_png: str = "../analysis_results/reasoning_runtime_comparison.png"
                 ) -> None:
    # results: {n: {"parallel": t_axioms1, "successive": t_axioms2}, ...}

    rows = [{"coins": n,
             "axioms1_seconds": v["parallel"],
             "axioms2_seconds": v["successive"]}
            for n, v in results.items()]

    df = pd.DataFrame(rows).sort_values("coins")

    plt.figure()
    plt.plot(df["coins"], df["axioms1_seconds"], marker="o", label="parallel application")
    plt.plot(df["coins"], df["axioms2_seconds"], marker="o", label="successive application")
    plt.xlabel("Number of Coins")
    plt.ylabel("Time (seconds)")
    plt.title("Reasoning Runtime vs Number of Coins")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)


if __name__=="__main__":
    results = run_benchmark()
    plot_results(results)