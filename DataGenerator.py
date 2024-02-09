import pandas as pd
import numpy as np
import SparqlConnector

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
        df['s'] = "ex:cn_type_" + df['s'].astype('string')
        df['o'] = "ex:cn_type_" + df['o'].astype('string')
        df['p'] = "ex:similarTo"
        df['certainty'] = np.round(np.random.rand(n*n), decimals=2)
        df['model'] = 'ex:model_' + str(model_nr)
        result = pd.concat([result, df])
    return result


if __name__ == '__main__':
    df = generate_data(100, 3)
    conn = SparqlConnector.ReificationSparqlConnector("http://localhost:3030/test/query",
                                                      "http://localhost:3030/test/update",
                                                      "http://localhost:3030/test/data")
    conn.delete_query(delete_all=True)
    conn.upload_df(df)

