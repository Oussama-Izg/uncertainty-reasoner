import DataGenerator
import Reasoner
import SparqlConnector
import logging
from pathlib import Path
import time
import pandas as pd
import numpy as np
import logging


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

np.random.seed(123456)

QUERY_ENDPOINT = "http://localhost:3030/test/query"
UPDATE_ENDPOINT = "http://localhost:3030/test/update"
GSP_ENDPOINT = "http://localhost:3030/test/data"


def sparql_test_reification(n=500):
    """
    Test SPARQL endpoint with reification triples. Generates around n*n rows in internal format and thus n*n*5 triples.
    :param n: Amount of coin types.
    :return:
    """

    conn = SparqlConnector.ReificationSparqlConnector(QUERY_ENDPOINT,
                                               UPDATE_ENDPOINT,
                                               GSP_ENDPOINT)

    logger.info("Generating data")
    df1 = DataGenerator.generate_vague_similarity_data(n, 3)
    df2 = DataGenerator.generate_precise_certain_data(n)
    df = pd.concat([df1, df2]).reset_index(drop=True)

    logger.info("Deleting old data")
    conn.delete_query(delete_all=True)
    logger.info("Uploading new data")
    start = time.time()
    conn.upload_df(df)
    end = time.time()
    # 37.62
    logger.info(f"Done in {end - start} seconds. Inserted {df.shape[0]} rows.")
    logger.info("Querying data")
    start = time.time()
    df = conn.download_df()[0]
    df.to_csv('export.csv', index=False)
    count = df.shape[0]
    end = time.time()
    # 23.52 1498500
    logger.info(f"Done in {end - start} seconds. Queried {count} rows.")


def sparql_test_sparql_star(n=500):
    """
    Test SPARQL endpoint with RDF-star triples. Generates around n*n rows in internal format and thus n*n*3 triples.
    :param n: Amount of coin types.
    :return:
    """
    conn = SparqlConnector.SparqlStarConnector(QUERY_ENDPOINT,
                                               UPDATE_ENDPOINT,
                                               GSP_ENDPOINT)

    logger.info("Generating data")
    df1 = DataGenerator.generate_vague_similarity_data(n, 3)
    df2 = DataGenerator.generate_precise_certain_data(n)
    df = pd.concat([df1, df2]).reset_index(drop=True)

    logger.info("Deleting old data")
    conn.delete_query(delete_all=True)
    logger.info("Uploading new data")
    start = time.time()
    conn.upload_df(df)
    end = time.time()
    # 37.62
    logger.info(f"Done in {end - start} seconds. Inserted {df.shape[0]} rows.")
    logger.info("Querying data")
    start = time.time()
    df = conn.download_df()[0]
    df.to_csv('export.csv', index=False)
    count = df.shape[0]
    end = time.time()
    # 23.52 1498500
    logger.info(f"Done in {end - start} seconds. Queried {count} rows.")


def reasoner_test_aggregation_mean_handcrafted():
    """
    Test AggregationAxiom on handcrafted data.
    :return:
    """
    conn = SparqlConnector.SparqlStarConnector(QUERY_ENDPOINT,
                                               UPDATE_ENDPOINT,
                                               GSP_ENDPOINT)
    df = pd.read_csv('data/agg_mean_test.csv')
    conn.delete_query(delete_all=True)
    conn.upload_df(df)
    axioms = [
        Reasoner.AggregationAxiom('ex:similarTo', 'mean')
    ]
    reasoner = Reasoner.Reasoner(axioms)
    reasoner.load_data_from_endpoint(conn)
    reasoner.reason()

    logger.info("Writing output to output/agg_mean_test.csv")

    reasoner.get_triples_as_df().to_csv('output/agg_mean_test.csv', index=False)


def reasoner_test_aggregation_mean(n_coin_types: int, iterations: int, conn_type='rdf-star'):
    """
    Test AggregationAxiom with synthetic data. Creates n*n rows in internal format.
    :param n_coin_types: Amount of coin types
    :param iterations: Amount of iterations to create the mean.
    :param conn_type: rdf-star or reification
    :return:
    """
    if conn_type == 'rdf-star':
        conn = SparqlConnector.SparqlStarConnector(QUERY_ENDPOINT,
                                                   UPDATE_ENDPOINT,
                                                   GSP_ENDPOINT)
    elif conn_type == 'reification':
        conn = SparqlConnector.ReificationSparqlConnector(QUERY_ENDPOINT,
                                                          UPDATE_ENDPOINT,
                                                          GSP_ENDPOINT)
    else:
        raise ValueError(f'Undefined connection type: {conn_type}')
    df = DataGenerator.generate_vague_similarity_data(n_coin_types, 3)
    conn.delete_query(delete_all=True)
    conn.upload_df(df)
    axioms = [
        Reasoner.AggregationAxiom('ex:similarTo', 'mean')
    ]
    time_sum = 0
    data_retrieval_time_sum = 0
    for i in range(iterations):
        start = time.time()
        reasoner = Reasoner.Reasoner(axioms)
        reasoner.load_data_from_endpoint(conn)
        end = time.time()
        data_retrieval_time_sum += end - start
        reasoner.reason()
        end = time.time()
        time_sum += end - start

    return time_sum/iterations, data_retrieval_time_sum/iterations


def reasoner_test_dst_axiom_synthetic(n_coins: int, n_coin_types: int, iterations: int):
    """
    Test DempsterShaferAxiom with synthetic data.
    :param n_coins: Amount of coins
    :param n_coin_types: Amount of coin types
    :param iterations: Amount of iterations to create the mean.
    :return:
    """
    df_dst_input = DataGenerator.generate_dst_input_data(n_coins, n_coin_types, 3)

    conn = SparqlConnector.SparqlStarConnector(QUERY_ENDPOINT,
                                               UPDATE_ENDPOINT,
                                               GSP_ENDPOINT)
    conn.delete_query(delete_all=True)
    conn.upload_df(df_dst_input)
    axioms = [
        Reasoner.DempsterShaferAxiom('ex:isCoinType')
    ]

    time_sum = 0
    data_retrieval_time_sum = 0
    for i in range(iterations):
        start = time.time()
        reasoner = Reasoner.Reasoner(axioms)
        reasoner.load_data_from_endpoint(conn)
        end = time.time()
        data_retrieval_time_sum += end - start
        reasoner.reason()
        end = time.time()
        time_sum += end - start

    return time_sum / iterations, data_retrieval_time_sum / iterations, df_dst_input.shape[0]


def reasoner_test_DST_axiom_handcrafted():
    """
    Test DempsterShaferAxiom with handcrafted data.
    :return:
    """
    logger.info("Generating data")
    df_dst_input = pd.read_csv('data/dst_test.csv')

    conn = SparqlConnector.SparqlStarConnector(QUERY_ENDPOINT,
                                               UPDATE_ENDPOINT,
                                               GSP_ENDPOINT)
    logger.info("Deleting old data")
    conn.delete_query(delete_all=True)
    logger.info("Uploading data")
    conn.upload_df(df_dst_input)
    axioms = [
        Reasoner.DempsterShaferAxiom('ex:isCoinType')
    ]

    reasoner = Reasoner.Reasoner(axioms)
    reasoner.load_data_from_endpoint(conn)
    reasoner.reason()
    reasoner.get_triples_as_df()

    logger.info("Writing output to output/dst_test.csv")

    reasoner.get_triples_as_df().sort_values(by=['s', 'p', 'o']).to_csv('output/dst_test.csv', index=False)


def reasoner_test_uncertainty_assignment_axiom_sythetic(n_coins: int, n_issuer: int, n_issuing_for: int, iterations: int):
    """
    Test CertaintyAssignmentAxiom with synthetic data.
    :param n_coins: Number of coins
    :param n_issuer: Number of issuer
    :param n_issuing_for: Number of depicted persons
    :param iterations: Number of iterations
    :return:
    """
    df_afe = DataGenerator.generate_afe_dst_input_data(n_coins, n_issuer, n_issuing_for)

    conn = SparqlConnector.SparqlStarConnector(QUERY_ENDPOINT,
                                               UPDATE_ENDPOINT,
                                               GSP_ENDPOINT)
    conn.delete_query(delete_all=True)
    conn.upload_df(df_afe)
    axioms = [
        Reasoner.CertaintyAssignmentAxiom("ex:issuer"),
        Reasoner.CertaintyAssignmentAxiom("ex:issuing_for"),
        Reasoner.CertaintyAssignmentAxiom("ex:domain_knowledge"),
    ]

    time_sum = 0
    data_retrieval_time_sum = 0
    for i in range(iterations):
        start = time.time()
        reasoner = Reasoner.Reasoner(axioms)
        reasoner.load_data_from_endpoint(conn)
        end = time.time()
        data_retrieval_time_sum += end - start
        reasoner.reason()
        end = time.time()
        time_sum += end - start

    return time_sum / iterations, data_retrieval_time_sum / iterations


def reasoner_test_uncertainty_assignment_axiom_handcrafted():
    """
    Test CertaintyAssignmentAxiom with handcrafted data.
    :return:
    """
    df_afe = pd.read_csv('data/afe_test_data.csv')

    conn = SparqlConnector.SparqlStarConnector(QUERY_ENDPOINT,
                                               UPDATE_ENDPOINT,
                                               GSP_ENDPOINT)
    conn.delete_query(delete_all=True)
    conn.upload_df(df_afe)
    axioms = [
        Reasoner.CertaintyAssignmentAxiom("ex:issuer"),
        Reasoner.CertaintyAssignmentAxiom("ex:issuing_for"),
        Reasoner.CertaintyAssignmentAxiom("ex:domain_knowledge"),
    ]


    reasoner = Reasoner.Reasoner(axioms)
    reasoner.load_data_from_endpoint(conn)
    reasoner.reason()

    reasoner.get_triples_as_df().to_csv('output/afe_test_data_uncertainty_assignment.csv', index=False)


def reasoner_test_AFE_DST_synthetic(n_coins: int, n_issuer: int, n_issuing_for: int, iterations: int):
    """
    Test AFEDempsterShaferAxiom with synthetic data.
    :param n_coins: Number of coins
    :param n_issuer: Number of issuer
    :param n_issuing_for: Number of depicted persons
    :param iterations: Number of iterations
    :return:
    """
    df_afe = DataGenerator.generate_afe_dst_input_data(n_coins, n_issuer, n_issuing_for)

    conn = SparqlConnector.SparqlStarConnector(QUERY_ENDPOINT,
                                               UPDATE_ENDPOINT,
                                               GSP_ENDPOINT)
    conn.delete_query(delete_all=True)
    conn.upload_df(df_afe)
    axioms = [
        Reasoner.CertaintyAssignmentAxiom("ex:issuer"),
        Reasoner.CertaintyAssignmentAxiom("ex:issuing_for"),
        Reasoner.CertaintyAssignmentAxiom("ex:domain_knowledge"),
        Reasoner.AFEDempsterShaferAxiom('ex:issuer', 'ex:issuing_for', 'ex:domain_knowledge')
    ]

    time_sum = 0
    data_retrieval_time_sum = 0
    for i in range(iterations):
        start = time.time()
        reasoner = Reasoner.Reasoner(axioms)
        reasoner.load_data_from_endpoint(conn)
        end = time.time()
        data_retrieval_time_sum += end - start
        reasoner.reason()
        end = time.time()
        time_sum += end - start

    return time_sum / iterations, data_retrieval_time_sum / iterations


def reasoner_test_AFE_DST_handcrafted():
    """
    Test AFEDempsterShaferAxiom with handcrafted data.
    :return:
    """
    df_afe = pd.read_csv('data/afe_test_data.csv')
    conn = SparqlConnector.SparqlStarConnector(QUERY_ENDPOINT,
                                               UPDATE_ENDPOINT,
                                               GSP_ENDPOINT)
    conn.delete_query(delete_all=True)
    conn.upload_df(df_afe)
    axioms = [
        Reasoner.CertaintyAssignmentAxiom("ex:issuer"),
        Reasoner.CertaintyAssignmentAxiom("ex:issuing_for"),
        Reasoner.CertaintyAssignmentAxiom("ex:domain_knowledge"),
        Reasoner.AFEDempsterShaferAxiom('ex:issuer', 'ex:issuing_for', 'ex:domain_knowledge')
    ]

    reasoner = Reasoner.Reasoner(axioms)
    reasoner.load_data_from_endpoint(conn)
    reasoner.reason()
    reasoner.get_triples_as_df().to_csv('output/afe_test_data.csv')


def reasoner_test_AFE_DST_real_data():
    """
    Test AFEDempsterShaferAxiom with real AFE data.
    :return:
    """
    df_afe = pd.read_csv('data/afe_input.csv')
    conn = SparqlConnector.SparqlStarConnector(QUERY_ENDPOINT,
                                               UPDATE_ENDPOINT,
                                               GSP_ENDPOINT)
    conn.delete_query(delete_all=True)
    conn.upload_df(df_afe)
    axioms = [
        Reasoner.CertaintyAssignmentAxiom("ex:issuer"),
        Reasoner.CertaintyAssignmentAxiom("ex:issuing_for"),
        Reasoner.CertaintyAssignmentAxiom("ex:domain_knowledge"),
        Reasoner.AFEDempsterShaferAxiom('ex:issuer', 'ex:issuing_for', 'ex:domain_knowledge')
    ]

    reasoner = Reasoner.Reasoner(axioms)
    reasoner.load_data_from_endpoint(conn)

    reasoner.reason()
    reasoner.get_triples_as_df().to_csv('output/afe_output.csv')


def reasoner_test_chain_rule_handcrafted():
    """
    Test chain rule use case with handcrafted data.
    :return:
    """
    df_chain_rule = pd.read_csv('data/chain_rule_test.csv')
    conn = SparqlConnector.SparqlStarConnector(QUERY_ENDPOINT,
                                               UPDATE_ENDPOINT,
                                               GSP_ENDPOINT)
    conn.delete_query(delete_all=True)
    conn.upload_df(df_chain_rule)
    axioms = [
        Reasoner.DempsterShaferAxiom('ex:isCoinType'),
        Reasoner.AggregationAxiom('ex:similarTo', 'mean'),
        Reasoner.ChainRuleAxiom('ex:similarTo', 'ex:similarTo', 'ex:similarTo',
                                'lukasiewicz'),
        Reasoner.ChainRuleAxiom('ex:isCoinType', 'ex:similarTo', 'ex:similarTo',
                                'product', sum_values=True),
        Reasoner.ChainRuleAxiom('ex:isCoinType', 'ex:isCoinTypeOf', 'ex:sameCoinTypeAs',
                                'product'),
        Reasoner.DisjointAxiom('ex:isCoinType', 'ex:similarTo', throw_exception=False),
        Reasoner.DisjointAxiom('ex:isCoinTypeOf', 'ex:similarTo', throw_exception=False),
        Reasoner.SelfDisjointAxiom('ex:similarTo', throw_exception=False),
        Reasoner.InverseAxiom('ex:similarTo', 'ex:similarTo'),
        Reasoner.InverseAxiom('ex:isCoinType', 'ex:isCoinTypeOf'),
        Reasoner.InverseAxiom('ex:isCoinTypeOf', 'ex:isCoinType'),
        Reasoner.InverseAxiom('ex:sameCoinTypeAs', 'ex:sameCoinTypeAs'),
    ]

    reasoner = Reasoner.Reasoner(axioms)
    reasoner.load_data_from_endpoint(conn)

    reasoner.reason()
    reasoner.get_triples_as_df().sort_values(by=['s', 'p', 'o']).to_csv('output/chain_rule_test.csv', index=False)


def reasoner_test_chain_rule_synthetic(n_coins, n_coin_types, iterations):
    """
    Test chain rule use case with synthetic data.
    :param n_coins: Number of coins
    :param n_coin_types: Number of coin types
    :param iterations: Number of iterations
    :return:
    """
    df_dst_input = DataGenerator.generate_dst_input_data(n_coins, n_coin_types, 3)
    df_similarity_input = DataGenerator.generate_vague_similarity_data(n_coin_types, 3)
    df_types = DataGenerator.generate_type_triples(n_coins, n_coin_types)
    df_similarity_input = df_similarity_input[df_similarity_input['weight'] > 0.8]
    df_chain_rule = pd.concat([df_similarity_input, df_dst_input, df_types], ignore_index=True)

    conn = SparqlConnector.SparqlStarConnector(QUERY_ENDPOINT,
                                               UPDATE_ENDPOINT,
                                               GSP_ENDPOINT)
    conn.delete_query(delete_all=True)
    conn.upload_df(df_chain_rule)
    axioms = [
        Reasoner.DempsterShaferAxiom('ex:isCoinType'),
        Reasoner.AggregationAxiom('ex:similarTo', 'mean'),
        Reasoner.ChainRuleAxiom('ex:similarTo', 'ex:similarTo', 'ex:similarTo',
                                'lukasiewicz', output_threshold=0.8, class_1='ex:CoinType',
                                class_2='ex:CoinType', class_3='ex:CoinType'),
        Reasoner.ChainRuleAxiom('ex:similarTo', 'ex:similarTo', 'ex:similarTo',
                                'lukasiewicz', output_threshold=0.8, class_1='ex:CoinType',
                                class_2='ex:CoinType', class_3='ex:CoinType'),
        Reasoner.ChainRuleAxiom('ex:similarTo', 'ex:similarTo', 'ex:similarTo',
                                'lukasiewicz', output_threshold=0.8, class_1='ex:Coin',
                                class_2='ex:CoinType', class_3='ex:CoinType'),
        Reasoner.ChainRuleAxiom('ex:similarTo', 'ex:similarTo', 'ex:similarTo',
                                'lukasiewicz', output_threshold=0.8, class_1='ex:Coin',
                                class_2='ex:CoinType', class_3='ex:Coin'),
        Reasoner.ChainRuleAxiom('ex:isCoinType', 'ex:similarTo', 'ex:similarTo',
                                'product', sum_values=True, output_threshold=0.8, class_2='ex:CoinType',
                                class_3='ex:CoinType'),
        Reasoner.ChainRuleAxiom('ex:isCoinType', 'ex:similarTo', 'ex:similarTo',
                                'product', sum_values=True, output_threshold=0.8, class_2='ex:CoinType',
                                class_3='ex:Coin'),
        Reasoner.ChainRuleAxiom('ex:isCoinType', 'ex:isCoinTypeOf', 'ex:sameCoinTypeAs',
                                'product'),
        Reasoner.DisjointAxiom('ex:isCoinType', 'ex:similarTo', throw_exception=False),
        Reasoner.DisjointAxiom('ex:isCoinTypeOf', 'ex:similarTo', throw_exception=False),
        Reasoner.SelfDisjointAxiom('ex:similarTo', throw_exception=False),
        Reasoner.InverseAxiom('ex:similarTo', 'ex:similarTo'),
        Reasoner.InverseAxiom('ex:isCoinType', 'ex:isCoinTypeOf'),
        Reasoner.InverseAxiom('ex:isCoinTypeOf', 'ex:isCoinType'),
        Reasoner.InverseAxiom('ex:sameCoinTypeAs', 'ex:sameCoinTypeAs'),
    ]

    time_sum = 0
    data_retrieval_time_sum = 0
    for i in range(iterations):
        start = time.time()
        reasoner = Reasoner.Reasoner(axioms)
        reasoner.load_data_from_endpoint(conn)
        end = time.time()
        data_retrieval_time_sum += end - start
        reasoner.reason()
        end = time.time()
        time_sum += end - start

    return time_sum / iterations, data_retrieval_time_sum / iterations


def reasoner_test_chain_rule_part_synthetic(n_coins, n_coin_types, iterations):
    """
    Test chain rule use case with synthetic data.
    :param n_coins: Number of coins
    :param n_coin_types: Number of coin types
    :param iterations: Number of iterations
    :return:
    """
    df_dst_input = DataGenerator.generate_dst_input_data(n_coins, n_coin_types, 3)
    df_similarity_input = DataGenerator.generate_vague_similarity_data(n_coin_types, 3)
    df_types = DataGenerator.generate_type_triples(n_coins, n_coin_types)
    df_similarity_input = df_similarity_input[df_similarity_input['weight'] > 0.8]
    df_chain_rule = pd.concat([df_similarity_input, df_dst_input, df_types], ignore_index=True)

    conn = SparqlConnector.SparqlStarConnector(QUERY_ENDPOINT,
                                               UPDATE_ENDPOINT,
                                               GSP_ENDPOINT)
    conn.delete_query(delete_all=True)
    conn.upload_df(df_chain_rule)
    axioms = [
        Reasoner.DempsterShaferAxiom('ex:isCoinType'),
        Reasoner.AggregationAxiom('ex:similarTo', 'mean'),
        Reasoner.ChainRuleAxiom('ex:isCoinType', 'ex:similarTo', 'ex:similarTo',
                                'product', sum_values=True, output_threshold=0.8, class_2='ex:CoinType',
                                class_3='ex:CoinType'),
        Reasoner.ChainRuleAxiom('ex:isCoinType', 'ex:similarTo', 'ex:similarTo',
                                'product', sum_values=True, output_threshold=0.8, class_2='ex:CoinType',
                                class_3='ex:Coin'),
        Reasoner.ChainRuleAxiom('ex:isCoinType', 'ex:isCoinTypeOf', 'ex:sameCoinTypeAs',
                                'product'),
        Reasoner.DisjointAxiom('ex:isCoinType', 'ex:similarTo', throw_exception=False),
        Reasoner.DisjointAxiom('ex:isCoinTypeOf', 'ex:similarTo', throw_exception=False),
        Reasoner.SelfDisjointAxiom('ex:similarTo', throw_exception=False),
        Reasoner.InverseAxiom('ex:similarTo', 'ex:similarTo'),
        Reasoner.InverseAxiom('ex:isCoinType', 'ex:isCoinTypeOf'),
        Reasoner.InverseAxiom('ex:isCoinTypeOf', 'ex:isCoinType'),
        Reasoner.InverseAxiom('ex:sameCoinTypeAs', 'ex:sameCoinTypeAs'),
    ]

    time_sum = 0
    data_retrieval_time_sum = 0
    for i in range(iterations):
        start = time.time()
        reasoner = Reasoner.Reasoner(axioms)
        reasoner.load_data_from_endpoint(conn)
        end = time.time()
        data_retrieval_time_sum += end - start
        reasoner.reason()
        end = time.time()
        time_sum += end - start

    return time_sum / iterations, data_retrieval_time_sum / iterations

if __name__ == '__main__':
    print("Please select the test:")
    print("1: Test SPARQL endpoint with reification triples")
    print("2: Test SPARQL endpoint with RDF-star triples")
    print("3: Test Use Case 1 with similarity models from multiple models (synthetic, benchmark)")
    print("4: Test Use Case 1 with similarity models from multiple models (handcrafted)")
    print("5: Test Use Case 2 with coin types from multiple models (synthetic, benchmark)")
    print("6: Test Use Case 2 with coin types from multiple models (handcrafted)")
    print("7: Test Use Case 4 with certainty assignments to triple alternatives (synthetic, benchmark)")
    print("8: Test Use Case 4 with certainty assignments to triple alternatives (handcrafted)")
    print("9: Test Use Case 5 with AFE DST (synthetic, benchmark)")
    print("10: Test Use Case 5 with AFE DST (handcrafted)")
    print("11: Test Use Case 5 with AFE DST (real data)")
    print("12: Test Use Case 6 similarity chain rules (synthetic, benchmark)")
    print("13: Test Use Case 6 similarity chain rules while removing one chain rule (synthetic, benchmark)")
    print("14: Test Use Case 6 similarity chain rules (handcrafted)")
    Path("output").mkdir(parents=True, exist_ok=True)

    selection = -1
    while True:
        try:
            selection = int(input("Select the test by typing the number: "))
            if 0 < selection <= 14:
                break
            else:
                print("Selected number must be between 1 and 14")
        except ValueError:
            print("This is not a number. Try again.")

    if selection == 1:
        sparql_test_reification()
    elif selection == 2:
        sparql_test_sparql_star()
    elif selection == 3:
        # Disable logging messages for the reasoner
        logging.getLogger('Reasoner').setLevel(logging.WARNING)

        print("Please select the connection type:")
        print("1: RDF-star")
        print("2: Reification")

        selection = -1
        while True:
            try:
                selection = int(input("Select the test by typing the number: "))
                if 0 < selection <= 2:
                    break
                else:
                    print("Selected number must be between 1 and 2")
            except ValueError:
                print("This is not a number. Try again.")
        if selection == 1:
            conn_type = 'rdf-star'
        else:
            conn_type = 'reification'

        coin_type_numbers = [100, 200, 300, 400, 500, 600, 700, 800]
        means = []
        data_retrieval_means = []
        for coin_type_number in coin_type_numbers:
            mean_time, data_retrieval_mean_time = reasoner_test_aggregation_mean(coin_type_number, 10, conn_type)
            means.append(mean_time)
            data_retrieval_means.append(data_retrieval_mean_time)
            print(f"Reasoning took an average of {round(mean_time, 3)} seconds for {coin_type_number} coins.")
            print(f"{round(data_retrieval_mean_time, 3)} seconds of this was querying the data.")

        print(f"x = np.array({str(coin_type_numbers)})")
        print(f"y = np.array({str(means)})")
        print(f"y2 = np.array({str(data_retrieval_means)})")
    elif selection == 4:
        reasoner_test_aggregation_mean_handcrafted()
    elif selection == 5:
        # Disable logging messages for the reasoner
        logging.getLogger('Reasoner').setLevel(logging.WARNING)

        coin_numbers = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        means = []
        data_retrieval_means = []
        triple_numbers = []
        for coin_number in coin_numbers:
            mean_time, data_retrieval_mean_time, n_triples = reasoner_test_dst_axiom_synthetic(coin_number, 1000, 10)
            triple_numbers.append(n_triples)
            means.append(mean_time)
            data_retrieval_means.append(data_retrieval_mean_time)
            print(f"Reasoning took an average of {round(mean_time, 3)} seconds for {coin_number} coins.")
            print(f"{round(data_retrieval_mean_time, 3)} seconds of this was querying the data.")
        print(f"x = np.array({str(coin_numbers)})")
        print(f"y = np.array({str(means)})")
        print(f"y2 = np.array({str(data_retrieval_means)})")
    elif selection == 6:
        reasoner_test_DST_axiom_handcrafted()
    elif selection == 7:
        # Disable logging messages for the reasoner
        logging.getLogger('Reasoner').setLevel(logging.WARNING)

        coin_numbers = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
        means = []
        data_retrieval_means = []
        for coin_number in coin_numbers:
            mean_time, data_retrieval_mean_time = reasoner_test_uncertainty_assignment_axiom_sythetic(coin_number, 100, 100, 10)
            means.append(mean_time)
            data_retrieval_means.append(data_retrieval_mean_time)
            print(f"Reasoning took an average of {round(mean_time, 3)} seconds for {coin_number} coins.")
            print(f"{round(data_retrieval_mean_time, 3)} seconds of this was querying the data.")
        print(f"x = np.array({str(coin_numbers)})")
        print(f"y = np.array({str(means)})")
        print(f"y2 = np.array({str(data_retrieval_means)})")
    elif selection == 8:
        reasoner_test_uncertainty_assignment_axiom_handcrafted()
    elif selection == 9:
        # Disable logging messages for the reasoner
        logging.getLogger('Reasoner').setLevel(logging.WARNING)

        coin_numbers = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        means = []
        data_retrieval_means = []
        for coin_number in coin_numbers:
            mean_time, data_retrieval_mean_time = reasoner_test_AFE_DST_synthetic(coin_number, 100, 100, 10)
            means.append(mean_time)
            data_retrieval_means.append(data_retrieval_mean_time)
            print(f"Reasoning took an average of {round(mean_time, 3)} seconds for {coin_number} coins.")
            print(f"{round(data_retrieval_mean_time, 3)} seconds of this was querying the data.")
        print(f"x = np.array({str(coin_numbers)})")
        print(f"y = np.array({str(means)})")
        print(f"y2 = np.array({str(data_retrieval_means)})")
    elif selection == 10:
        reasoner_test_AFE_DST_handcrafted()
    elif selection == 11:
        reasoner_test_AFE_DST_real_data()
    elif selection == 12:
        # Disable logging messages for the reasoner
        logging.getLogger('Reasoner').setLevel(logging.WARNING)

        coin_numbers = [100, 200, 300, 400, 500, 600]
        means = []
        data_retrieval_means = []
        for coin_number in coin_numbers:
            mean_time, data_retrieval_mean_time = reasoner_test_chain_rule_synthetic(coin_number, 100, 10)
            means.append(mean_time)
            data_retrieval_means.append(data_retrieval_mean_time)
            print(f"Reasoning took an average of {round(mean_time, 3)} seconds for {coin_number} coins.")
            print(f"{round(data_retrieval_mean_time, 3)} seconds of this was querying the data.")
        print(f"x = np.array({str(coin_numbers)})")
        print(f"y = np.array({str(means)})")
        print(f"y2 = np.array({str(data_retrieval_means)})")
    elif selection == 13:
        # Disable logging messages for the reasoner
        logging.getLogger('Reasoner').setLevel(logging.WARNING)

        coin_numbers = [100, 200, 300, 400, 500, 600]
        means = []
        data_retrieval_means = []
        for coin_number in coin_numbers:
            mean_time, data_retrieval_mean_time = reasoner_test_chain_rule_part_synthetic(coin_number, 100, 10)
            means.append(mean_time)
            data_retrieval_means.append(data_retrieval_mean_time)
            print(f"Reasoning took an average of {round(mean_time, 3)} seconds for {coin_number} coins.")
            print(f"{round(data_retrieval_mean_time, 3)} seconds of this was querying the data.")
        print(f"x = np.array({str(coin_numbers)})")
        print(f"y = np.array({str(means)})")
        print(f"y2 = np.array({str(data_retrieval_means)})")
    elif selection == 14:
        reasoner_test_chain_rule_handcrafted()

