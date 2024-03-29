import pandas as pd
import numpy as np


def generate_vague_similarity_data(n, n_models):
    """
    Create synthetic vague similarity data. Like ex:cn_type_1 ex:similarTo ex:cn_type_2
    :param n: Number of coin types
    :param n_models: Number of models
    :return: Cartesian product of coin type combinations with random weight in internal dataframe format
    """
    result = pd.DataFrame()
    for model_nr in range(1, n_models+1):
        cn_type_numbers = np.arange(1, n+1)
        # Create cartesian product
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
        # Random weights
        df['weight'] = np.round(np.random.rand(df.shape[0]), decimals=2)
        df['model'] = 'ex:model_' + str(model_nr)
        result = pd.concat([result, df])
    return result.reset_index(drop=True)


def generate_precise_certain_data(n):
    """
    Generate precise/certain triples to test SPARQL connectors.
    :param n: Number of certain triples
    :return: Dataframe in internal format with certain and precise triples
    """
    cn_type_numbers_from = np.arange(1, n+1)
    cn_type_numbers_to = cn_type_numbers_from.copy()
    np.random.shuffle(cn_type_numbers_to)

    df = pd.DataFrame()

    df['s'] = cn_type_numbers_from
    df['o'] = cn_type_numbers_to

    # No self references
    df.loc[df['s'] == df['o'], 's'] = df['s'] + 1

    df['s'] = "ex:cn_type_" + df['s'].astype('string')
    df['o'] = "ex:cn_type_" + df['o'].astype('string')
    df['p'] = "ex:examplePredicate"
    df['weight'] = 1

    return df


def generate_dst_input_data(n, n_coin_types, n_models, uncertainty_object="ex:uncertain"):
    """
    Generates input data for the DempsterShaferAxiom. Each coin has 5 possible coin types that the experts could choose.
    The experts choose up to three each.

    :param n: Number of coins
    :param n_coin_types: Number of coin types
    :param n_models: Number of experts or models
    :param uncertainty_object: Uncertainty object to indicate that the expert is uncertain about the selection
    :return: DST example data
    """
    result_from = []
    result_to = []
    model = []
    cn_type_numbers = np.arange(1, n_coin_types + 1)
    for i in range(n):
        # Possible choices for the experts
        possible = np.random.choice(cn_type_numbers, size=5, replace=False)
        for m in range(n_models):
            r = np.random.rand()
            # Get expert choices
            n_choices = 0
            if r < 0.2:
                n_choices = 1
            elif r < 0.7:
                n_choices = 2
            else:
                n_choices = 3
            choices = np.random.choice(possible, size=n_choices, replace=False)
            for j in range(n_choices):
                result_from.append("ex:coin_" + str(i))
                result_to.append("ex:cn_type_" + str(choices[j]))
                model.append("ex:model_" + str(m))
            # Probability for the uncertain checkbox to indicate an uncertain checkbox selection
            if np.random.rand() < 0.2:
                result_from.append("ex:coin_" + str(i))
                result_to.append(uncertainty_object)
                model.append("ex:model_" + str(m))
    df = pd.DataFrame({'s': result_from, 'o': result_to, 'model': model})
    df['p'] = 'ex:coinType'
    df['weight'] = 1

    return df








