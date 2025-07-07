import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame


def create_domain_knowledge_triples(filepath: str, target_path: str):
    df_domain_knowledge = pd.read_excel(filepath)

    df_domain_knowledge = df_domain_knowledge.drop_duplicates()

    df_domain_knowledge = df_domain_knowledge[['coinimage', 'issuer']]

    df_domain_knowledge["coinimage"] = "ex:issuing_for_" + df_domain_knowledge[
        "coinimage"].astype(str)
    df_domain_knowledge["issuer"] = "ex:issuer_" + df_domain_knowledge[
        "issuer"].astype(str)
    df_domain_knowledge = df_domain_knowledge.rename(columns={
        'coinimage': 's',
        'issuer': 'o'
    })
    df_domain_knowledge['p'] = 'ex:domain_knowledge'

    df_domain_knowledge.to_csv(target_path, index=False)

    return df_domain_knowledge


def create_issuer_triples(filepath: str, target_path: str):
    df = pd.read_excel(filepath)
    df = df.drop_duplicates()
    df = df[['id', 'issuers_ids', 'issuer_uncertainty']]

    # Split 'issuers_ids' by comma into lists
    df['issuers_ids'] = df['issuers_ids'].astype(str).str.split(',')
    df = df.explode('issuers_ids')
    df['issuers_ids'] = df['issuers_ids'].str.strip()

    df_issuer = df[['id', 'issuers_ids', 'issuer_uncertainty']]

    # Identify issuer selections that are uncertain
    df_issuer_uncertain = \
    df_issuer[df_issuer['issuer_uncertainty'].isin([2, 4])][
        ['id']].drop_duplicates()
    df_issuer_uncertain = df_issuer_uncertain.rename(columns={'id': 's'})
    df_issuer_uncertain['o'] = 'ex:uncertain'

    df_issuer = df_issuer.rename(columns={'id': 's', 'issuers_ids': 'o'})
    df_issuer = pd.concat(
        [df_issuer.drop(columns=['issuer_uncertainty']), df_issuer_uncertain])

    df_issuer['s'] = 'ex:coin_' + df_issuer['s'].astype(str)
    df_issuer['o'] = 'ex:issuer_' + df_issuer['o'].astype(str)
    df_issuer['p'] = 'ex:issuer'

    df_issuer.to_csv(target_path, index=False)

    return df_issuer


def create_issuing_for_triples(filepath: str, target_path: str):
    df = pd.read_excel(filepath)
    df = df.drop_duplicates()
    df = df[
        ['id', 'CoinImage', 'CoinImage2', 'CoinImage_Uncertainty_mapped']]

    df_issuing_for_1 = (
        df[['id', 'CoinImage']]
        .dropna(subset=['CoinImage'])  # remove rows where CoinImage is NaN
        .drop_duplicates()
        .rename(columns={'CoinImage': 'o', 'id': 's'}
                )
    )

    df_issuing_for_2 = (
        df[['id', 'CoinImage2']]
        .dropna(subset=['CoinImage2'])  # remove rows where CoinImage2 is NaN
        .drop_duplicates()
        .rename(columns={'CoinImage2': 'o', 'id': 's'}
                )
    )

    df_issuing_for_1['o'] = 'ex:issuing_for_' + df_issuing_for_1['o'].astype(
        int).astype(str)
    df_issuing_for_2['o'] = 'ex:issuing_for_' + df_issuing_for_2['o'].astype(
        int).astype(str)

    df_issuing_for_uncertain = \
    df[df['CoinImage_Uncertainty_mapped'].isin([2, 4])][
        ['id']].drop_duplicates()
    df_issuing_for_uncertain = df_issuing_for_uncertain.rename(
        columns={'id': 's'})
    df_issuing_for_uncertain['o'] = 'ex:uncertain'

    df_issuing_for = pd.concat(
        [df_issuing_for_1, df_issuing_for_2, df_issuing_for_uncertain])

    df_issuing_for['s'] = 'ex:coin_' + df_issuing_for['s'].astype(str)
    df_issuing_for['p'] = "ex:issuing_for"

    df_issuing_for.to_csv(target_path, index=False)

    return df_issuing_for


def create_all_triples(domain_knowledge_triples: DataFrame,
                       issuer_triples: DataFrame,
                       issuing_for_triples: DataFrame, target_path: str):
    all_triples = pd.concat([domain_knowledge_triples, issuer_triples, issuing_for_triples])
    all_triples.to_csv(target_path, index=False)
    return all_triples



if __name__ == "__main__":
    df_domain_knowledge = create_domain_knowledge_triples(filepath="data/2024_02_06_issuingFor_possibleIssuers.xlsx",
                                    target_path="triples/domain_knowledge_triples.csv")
    df_issuer = create_issuer_triples(filepath="data/2025_06_26_Issuer_IssuingFor.xlsx",
                                    target_path="triples/issuer_triples.csv")
    df_issuing_for = create_issuing_for_triples(filepath="data/2025_06_26_Issuer_IssuingFor.xlsx",
                                    target_path="triples/issuing_for_triples.csv")

    all_triples = create_all_triples(df_domain_knowledge, df_issuer, df_issuing_for, target_path="triples/all_triples")
