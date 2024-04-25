import DataGenerator
import Reasoner
import SparqlConnector
import logging
import time
import pandas as pd
import numpy as np
import logging


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

np.random.seed(123456)

def sparql_test_reification(n=1000):
    conn = SparqlConnector.ReificationSparqlConnector("http://localhost:3030/test/query",
                                               "http://localhost:3030/test/update",
                                               "http://localhost:3030/test/data")

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
    count = conn.download_df()[0].shape[0]
    end = time.time()
    # 23.52 1498500
    logger.info(f"Done in {end - start} seconds. Queried {count} rows.")


def sparql_test_sparql_star(n=1000):
    conn = SparqlConnector.SparqlStarConnector("http://localhost:3030/test/query",
                                               "http://localhost:3030/test/update",
                                               "http://localhost:3030/test/data")

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
    count = conn.download_df()[0].shape[0]
    end = time.time()
    # 23.52 1498500
    logger.info(f"Done in {end - start} seconds. Queried {count} rows.")


def reasoner_test_aggregation_mean_handcrafted():
    conn = SparqlConnector.SparqlStarConnector("http://localhost:3030/test/query",
                                               "http://localhost:3030/test/update",
                                               "http://localhost:3030/test/data")
    df = pd.read_csv('data/agg_mean_test')
    conn.delete_query(delete_all=True)
    conn.upload_df(df)
    axioms = [
        Reasoner.AggregationAxiom('ex:similarTo', 'mean')
    ]
    reasoner = Reasoner.Reasoner(axioms)
    reasoner.load_data_from_endpoint(conn)
    reasoner.reason()
    print(reasoner.get_triples_as_df())


def reasoner_test_aggregation_mean(n_coins: int, iterations: int):
    conn = SparqlConnector.SparqlStarConnector("http://localhost:3030/test/query",
                                               "http://localhost:3030/test/update",
                                               "http://localhost:3030/test/data")
    df = DataGenerator.generate_vague_similarity_data(n_coins, 3)
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

    return time_sum/iterations, data_retrieval_time_sum/iterations, df.shape[0]


def test_DST_axiom():
    logger.info("Generating data")
    df_dst_input = DataGenerator.generate_dst_input_data(10000, 1000, 3)

    conn = SparqlConnector.SparqlStarConnector("http://localhost:3030/test/query",
                                               "http://localhost:3030/test/update",
                                               "http://localhost:3030/test/data")
    logger.info("Deleting old data")
    conn.delete_query(delete_all=True)
    logger.info("Uploading data")
    conn.upload_df(df_dst_input)
    axioms = [
        Reasoner.CertaintyAssignmentAxiom('ex:isCoinType'),
        Reasoner.DempsterShaferAxiom('ex:isCoinType')
    ]

    reasoner = Reasoner.Reasoner(axioms)
    reasoner.load_data_from_endpoint(conn)
    reasoner.reason()
    print(reasoner.get_triples_as_df())


def test_DST_axiom_handcrafted_data():
    logger.info("Generating data")
    df_dst_input = pd.read_csv('data/dst_test.csv')

    conn = SparqlConnector.SparqlStarConnector("http://localhost:3030/test/query",
                                               "http://localhost:3030/test/update",
                                               "http://localhost:3030/test/data")
    logger.info("Deleting old data")
    conn.delete_query(delete_all=True)
    logger.info("Uploading data")
    conn.upload_df(df_dst_input)
    axioms = [
        Reasoner.CertaintyAssignmentAxiom('ex:isCoinType'),
        Reasoner.DempsterShaferAxiom('ex:isCoinType')
    ]

    reasoner = Reasoner.Reasoner(axioms)
    reasoner.load_data_from_endpoint(conn)
    reasoner.reason()
    print(reasoner.get_triples_as_df())


def test_AFE_DST_test_data():
    df_afe = pd.read_csv('afe_test_data.csv')
    conn = SparqlConnector.SparqlStarConnector("http://localhost:3030/test/query",
                                               "http://localhost:3030/test/update",
                                               "http://localhost:3030/test/data")
    conn.delete_query(delete_all=True)
    conn.upload_df(df_afe)
    axioms = [
        Reasoner.CertaintyAssignmentAxiom("ex:issuer"),
        Reasoner.CertaintyAssignmentAxiom("ex:issuing_for"),
        Reasoner.AFEDempsterShaferAxiom('ex:issuer', 'ex:issuing_for', 'ex:domain_knowledge')
    ]

    reasoner = Reasoner.Reasoner(axioms)
    reasoner.load_data_from_endpoint(conn)
    reasoner.reason()
    print(reasoner.get_triples_as_df())


def test_AFE_DST_synthetic_data():
    df_afe = DataGenerator.generate_afe_dst_input_data(5000, 100, 100)
    print(df_afe)
    conn = SparqlConnector.SparqlStarConnector("http://localhost:3030/test/query",
                                               "http://localhost:3030/test/update",
                                               "http://localhost:3030/test/data")
    conn.delete_query(delete_all=True)
    conn.upload_df(df_afe)
    axioms = [
        Reasoner.CertaintyAssignmentAxiom("ex:issuer"),
        Reasoner.CertaintyAssignmentAxiom("ex:issuing_for"),
        Reasoner.AFEDempsterShaferAxiom('ex:issuer', 'ex:issuing_for', 'ex:domain_knowledge')
    ]

    reasoner = Reasoner.Reasoner(axioms)
    reasoner.load_data_from_endpoint(conn)
    reasoner.reason()
    reasoner.get_triples_as_df().to_csv('output.csv', index=False)

def test_AFE_DST_real_data():
    df_afe = pd.read_csv('data/afe_input.csv')
    #df_afe = pd.read_csv('test.csv')
    conn = SparqlConnector.SparqlStarConnector("http://localhost:3030/test/query",
                                               "http://localhost:3030/test/update",
                                               "http://localhost:3030/test/data")
    conn.delete_query(delete_all=True)
    conn.upload_df(df_afe)
    axioms = [
        Reasoner.CertaintyAssignmentAxiom("ex:issuer"),
        Reasoner.CertaintyAssignmentAxiom("ex:issuing_for"),
        Reasoner.AFEDempsterShaferAxiom('ex:issuer', 'ex:issuing_for', 'ex:domain_knowledge')
    ]

    reasoner = Reasoner.Reasoner(axioms)
    reasoner.load_data_from_endpoint(conn)

    reasoner.reason()
    reasoner.get_triples_as_df().to_csv('output.csv')


def test_chain_rule_handcrafted():
    df_chain_rule = pd.read_csv('data/chain_rule_test.csv')
    conn = SparqlConnector.SparqlStarConnector("http://localhost:3030/test/query",
                                               "http://localhost:3030/test/update",
                                               "http://localhost:3030/test/data")
    conn.delete_query(delete_all=True)
    conn.upload_df(df_chain_rule)
    axioms = [
        Reasoner.ChainRuleAxiom('ex:isCoinType', 'ex:similarTo', 'ex:similarTo',
                                'product', sum_values=True),
        Reasoner.DisjointAxiom('ex:isCoinType', 'ex:similarTo', throw_exception=False)
    ]

    reasoner = Reasoner.Reasoner(axioms)
    reasoner.load_data_from_endpoint(conn)

    reasoner.reason()
    reasoner.get_triples_as_df().to_csv('output.csv')


def test_chain_rule():
    logger.info("Generating ex:coinType data")
    df_dst_input = DataGenerator.generate_dst_input_data(10000, 1000, 3)
    logger.info("Generating ex:similarTo data")
    df_similarity_input = DataGenerator.generate_vague_similarity_data(1000, 3)
    df_chain_rule = pd.concat([df_similarity_input, df_dst_input]).reset_index()

    conn = SparqlConnector.SparqlStarConnector("http://localhost:3030/test/query",
                                               "http://localhost:3030/test/update",
                                               "http://localhost:3030/test/data")
    logger.info("Deleting old data")
    conn.delete_query(delete_all=True)
    logger.info("Uploading new data")
    conn.upload_df(df_chain_rule)
    axioms = [
        Reasoner.CertaintyAssignmentAxiom('ex:isCoinType'),
        Reasoner.DempsterShaferAxiom('ex:isCoinType'),
        Reasoner.AggregationAxiom('ex:similarTo', 'mean'),
        Reasoner.ChainRuleAxiom('ex:isCoinType', 'ex:similarTo', 'ex:similarTo',
                                'product', sum_values=True, input_threshold=0.8),
        Reasoner.DisjointAxiom('ex:isCoinType', 'ex:similarTo', throw_exception=False)
    ]

    reasoner = Reasoner.Reasoner(axioms)
    reasoner.load_data_from_endpoint(conn)

    reasoner.reason()
    reasoner.get_triples_as_df().to_csv('output.csv')


if __name__ == '__main__':
    print("Please select the test:")
    print("1: Test SPARQL endpoint with reification triples")
    print("2: Test SPARQL endpoint with RDF-star triples")
    print("3: Test Use Case 1 with similarity models from multiple models")

    # Disable logging messages for the reasoner
    logging.getLogger('Reasoner').setLevel(logging.WARNING)

    selection = -1
    while True:
        try:
            selection = int(input("Select the test by typing the number: "))
            if 0 < selection <= 3:
                break
            else:
                print("Selected number must be between 1 and 3")
        except ValueError:
            print("This is not a number. Try again.")

    if selection == 1:
        sparql_test_reification()
    elif selection == 2:
        sparql_test_sparql_star()
    elif selection == 3:
        coin_numbers = [200, 400, 600, 800, 1000, 1200, 1400]
        means = []
        data_retrieval_means = []
        for coin_number in coin_numbers:
            mean_time, data_retrieval_mean_time, n_triples = reasoner_test_aggregation_mean(coin_number, 10)
            means.append(mean_time)
            data_retrieval_means.append(data_retrieval_mean_time)
            print(f"Reasoning took an average of {round(mean_time, 3)} seconds for {coin_number} coins and {n_triples} triples.")
            print(f"{round(data_retrieval_mean_time, 3)} seconds of this was querying the data.")
        print(means)
        print(data_retrieval_means)
