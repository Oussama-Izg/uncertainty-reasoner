import DataGenerator
import Reasoner
import SparqlConnector
import logging
import time
import pandas as pd


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
def sparql_test_reification():
    conn = SparqlConnector.ReificationSparqlConnector("http://localhost:3030/test/query",
                                               "http://localhost:3030/test/update",
                                               "http://localhost:3030/test/data")

    logger.info("Generating data")
    df1 = DataGenerator.generate_vague_similarity_data(1000, 3)
    df2 = DataGenerator.generate_similarity_data(1001, 2000)
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
    count = conn.read_into_df()[0].shape[0]
    end = time.time()
    # 23.52 1498500
    logger.info(f"Done in {end - start} seconds. Queried {count} rows.")

def sparql_test_sparql_star():
    conn = SparqlConnector.SparqlStarConnector("http://localhost:3030/test/query",
                                               "http://localhost:3030/test/update",
                                               "http://localhost:3030/test/data")

    logger.info("Generating data")
    df1 = DataGenerator.generate_vague_similarity_data(1000, 3)
    df2 = DataGenerator.generate_similarity_data(1001, 2000)
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
    count = conn.read_into_df()[0].shape[0]
    end = time.time()
    # 23.52 1498500
    logger.info(f"Done in {end - start} seconds. Queried {count} rows.")


def reasoner_test_aggregation_mean():
    #sparql_test_sparql_star()
    conn = SparqlConnector.SparqlStarConnector("http://localhost:3030/test/query",
                                               "http://localhost:3030/test/update",
                                               "http://localhost:3030/test/data")
    axioms = [
        Reasoner.AggregationAxiom('ex:similarTo', 'mean')
    ]

    reasoner = Reasoner.Reasoner(axioms)
    reasoner.load_data_from_endpoint(conn)
    df = reasoner.reason()

    print(df[df['model'] == 'uncertainty_reasoner'])

def test_AFE_DST_usecase_data():
    df_afe = pd.read_csv('afe_test_data.csv')
    conn = SparqlConnector.SparqlStarConnector("http://localhost:3030/test/query",
                                               "http://localhost:3030/test/update",
                                               "http://localhost:3030/test/data")
    conn.delete_query(delete_all=True)
    conn.upload_df(df_afe)
    axioms = [
        Reasoner.UncertaintyAssignmentAxiom("ex:issuer"),
        Reasoner.UncertaintyAssignmentAxiom("ex:issuing_for"),
        Reasoner.AFEDempsterShaferAxiom('ex:issuer', 'ex:issuing_for', 'ex:domain_knowledge')
    ]

    reasoner = Reasoner.Reasoner(axioms)
    reasoner.load_data_from_endpoint(conn)

    print(reasoner.reason())

def test_AFE_DST_real_data():
    df_afe = pd.read_csv('afe_input.csv')
    #df_afe = pd.read_csv('test.csv')
    conn = SparqlConnector.SparqlStarConnector("http://localhost:3030/test/query",
                                               "http://localhost:3030/test/update",
                                               "http://localhost:3030/test/data")
    conn.delete_query(delete_all=True)
    conn.upload_df(df_afe)
    axioms = [
        Reasoner.UncertaintyAssignmentAxiom("ex:issuer"),
        Reasoner.UncertaintyAssignmentAxiom("ex:issuing_for"),
        Reasoner.AFEDempsterShaferAxiom('ex:issuer', 'ex:issuing_for', 'ex:domain_knowledge')
    ]

    reasoner = Reasoner.Reasoner(axioms)
    reasoner.load_data_from_endpoint(conn)

    reasoner.reason().to_csv('output.csv')

if __name__ == '__main__':
    test_AFE_DST_real_data()