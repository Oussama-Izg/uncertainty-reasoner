import pandas as pd
import numpy as np


def generate_vague_similarity_data(n, n_models):
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

        # A similar B <-> B similar A
        # Therfore, you remove the duplicates
        df['tmp'] = df['s'].astype('string')+","+df['o'].astype('string')
        df.loc[df['s'] < df['o'], 'tmp'] = df['o'].astype('string')+","+df['s'].astype('string')
        df = df.drop_duplicates(subset=['tmp'])

        # No self references
        df = df[df['s'] != df['o']]

        df['s'] = "ex:cn_type_" + df['s'].astype('string')
        df['o'] = "ex:cn_type_" + df['o'].astype('string')
        df['p'] = "ex:similarTo"
        df['weight'] = np.round(np.random.rand(df.shape[0]), decimals=2)
        df['model'] = 'ex:model_' + str(model_nr)
        result = pd.concat([result, df])
    return result.reset_index(drop=True)


def generate_similarity_data(lb, ub):
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
    df['weight'] = 1

    return df






