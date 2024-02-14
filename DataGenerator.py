import pandas as pd
import numpy as np
import SparqlConnector
import logging
import time

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_data(n, n_models):
    result = pd.DataFrame()
    for model_nr in range(1, n_models+1):
        cn_type_numbers = np.arange(1, n+1)
        df = pd.DataFrame({'from': cn_type_numbers}, index=np.repeat(0, n))
        df = pd.merge(df, df, left_index=True, right_index=True)
        df = df.reset_index(drop=True)
        df = df.rename(columns = {
            'from_x': 's',
            'from_y': 'o'
        })

        # A similar B -> B similar A
        # Therfore, you remove the duplicates
        df['tmp'] = df['s'].astype('string')+","+df['o'].astype('string')
        df.loc[df['s'] < df['o'], 'tmp'] = df['o'].astype('string')+","+df['s'].astype('string')
        df = df.drop_duplicates(subset=['tmp'])

        # No self references
        df = df[df['s'] != df['o']]

        df['s'] = "ex:cn_type_" + df['s'].astype('string')
        df['o'] = "ex:cn_type_" + df['o'].astype('string')
        df['p'] = "ex:similarTo"
        df['certainty'] = np.round(np.random.rand(df.shape[0]), decimals=2)
        df['model'] = 'ex:model_' + str(model_nr)
        result = pd.concat([result, df])
    return result.reset_index(drop=True)

def generate_data2(lb, ub):
    cn_type_numbers_from = np.arange(lb, ub+1)
    cn_type_numbers_to = cn_type_numbers_from.copy()
    np.random.shuffle(cn_type_numbers_to)

    df = pd.DataFrame()

    df['s'] = cn_type_numbers_from
    df['o'] = cn_type_numbers_to

    # No self references
    df.loc[df['s'] == df['o'], 's'] = df['s'] + 1

    df['s'] = "ex:cn_type_" + df['s'].astype('string')
    df['o'] = "ex:cn_type_" + df['o'].astype('string')
    df['p'] = "ex:similarTo"
    df['certainty'] = 1

    return df


if __name__ == '__main__':
    conn = SparqlConnector.ReificationSparqlConnector("http://localhost:3030/test/query",
                                                      "http://localhost:3030/test/update",
                                                      "http://localhost:3030/test/data")
    logger.info("Generating data")
    df1 = generate_data(1000, 3)
    df2 = generate_data2(1001, 2000)
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
    count = conn.read_into_df().shape[0]
    end = time.time()
    # 23.52 1498500
    logger.info(f"Done in {end - start} seconds. Queried {count} rows.")


