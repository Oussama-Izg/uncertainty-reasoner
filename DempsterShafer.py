import pandas as pd


class MassFunction:
    """
    Implements a simpler mass function for the Dempster-Shafer-Theory. It only supports subsets with one element or all
    elements ("*" subset).
    """
    def __init__(self, mass_values: dict[str, float]):
        self._mass_values = mass_values

    def get_mass_values(self) -> dict[str, float]:
        """
        Get the mass values
        :return: Mass value dict
        """
        return self._mass_values.copy()

    def join_masses(self, m: 'MassFunction') -> 'MassFunction':
        """
        Join two masses via the Dempster-Shafer combination rule. Only supports subsets with one element or all
        elements ("*" subset).
        :param m: Mass function to join with
        :return: Joint mass function
        """
        result = {}
        empty_set_value = 0
        mass_values1 = self.get_mass_values()
        mass_values2 = m.get_mass_values()
        for subset1 in mass_values1.keys():
            for subset2 in mass_values2.keys():
                if subset1 == subset2:
                    result[subset1] = result.get(subset1, 0) + mass_values1[subset1] * mass_values2[subset2]
                elif subset1 == '*':
                    result[subset2] = result.get(subset2, 0) + mass_values1[subset1] * mass_values2[subset2]
                elif subset2 == '*':
                    result[subset1] = result.get(subset1, 0) + mass_values1[subset1] * mass_values2[subset2]
                else:
                    empty_set_value += mass_values1[subset1] * mass_values2[subset2]
        for subset in result.keys():
            result[subset] /= 1 - empty_set_value

        return MassFunction(result)


def df_to_subset_dict(df: pd.DataFrame, default_ignorance: float, uncertainty_object:str) -> dict[str, float]:
    """
    Translate dataframe to a subset dict
    :param df: The dataframe to translate
    :param default_ignorance: The ignorance to use
    :param uncertainty_object: The uncertainty object that indicates ignorance
    :return: Subset as dict
    """
    result = {}
    result['*'] = default_ignorance
    n = df[df['o'] != uncertainty_object].shape[0]
    for i, x in df.iterrows():
        if x['o'] == uncertainty_object:
            result['*'] += x['weight']
        else:
            result[x['o']] = x['weight'] - default_ignorance / n
    return result


if __name__ == '__main__':
    m1 = MassFunction({'*': 0.4, 'ex:cn_type_1': 0.3, 'ex:cn_type_2': 0.3})
    m2 = MassFunction({'*': 0.2, 'ex:cn_type_3': 0.4, 'ex:cn_type_4': 0.4})
    m3 = MassFunction({'*': 0.4, 'ex:cn_type_5': 0.3, 'ex:cn_type_4': 0.3})

    print(m1.join_masses(m2).get_mass_values())