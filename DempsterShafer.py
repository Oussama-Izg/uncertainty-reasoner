from typing import Optional

import pandas as pd


class IntervalMassFunction:
    """
    Implements a simpler mass function for the Dempster-Shafer-Theory. It only supports subsets with continuous
    intervals like 2...9 or all elements ("*" subset).
    """
    def __init__(self, mass_values: dict[tuple[int, int], float]):
        self._mass_values = mass_values

    def get_mass_values(self) -> dict[tuple[int, int], float]:
        """
        Get the mass values
        :return: Mass value dict
        """
        return self._mass_values.copy()

    def get_beliefs(self) -> dict[tuple[int, int], float]:
        """
        Calculate the belief values
        :return: Belief values
        """
        result = {}
        for subset1 in self._mass_values:
            if subset1 == '*':
                result['*'] = 1.0
                continue
            for subset2 in self._mass_values:
                if subset2 == '*':
                    continue
                # Check if subset2 is included and thus a subset of subset1
                if subset2[0] >= subset1[0] and subset2[1] <= subset1[1]:
                    result[subset1] = result.get(subset1, 0) + self._mass_values[subset2]
        return result

    def get_plausibility_values(self) -> dict[tuple[int, int], float]:
        """
        Calculate the plausibility values
        :return: Plausibility values
        """
        result = {}
        for subset1 in self._mass_values:
            if subset1 == '*':
                result['*'] = 1.0
                continue
            for subset2 in self._mass_values:
                if subset2 == '*':
                    result[subset1] = result.get(subset1, 0) + self._mass_values[subset2]
                    continue
                intersecting_interval = get_intersection(subset1, subset2)
                if intersecting_interval is not None:
                    result[subset1] = result.get(subset1, 0) + self._mass_values[subset2]
        return result

    def join_masses(self, m: 'IntervalMassFunction') -> 'IntervalMassFunction':
        """
        Join two masses via the Dempster-Shafer combination rule. Only supports subsets with continuous
        intervals like 2...9 or all elements ("*" subset).
        :param m: Mass function to join with
        :return: Joint mass function
        """
        result = {}
        empty_set_value = 0
        mass_values1 = self.get_mass_values()
        mass_values2 = m.get_mass_values()
        for subset1 in mass_values1.keys():
            for subset2 in mass_values2.keys():
                if subset1 == '*' and subset2 == '*':
                    result['*'] = mass_values1[subset1] * mass_values2[subset2]
                elif subset1 == '*':
                    result[subset2] = result.get(subset2, 0) + mass_values1[subset1] * mass_values2[subset2]
                elif subset2 == '*':
                    result[subset1] = result.get(subset1, 0) + mass_values1[subset1] * mass_values2[subset2]
                else:
                    # Calculate intersecting interval
                    resulting_subset = get_intersection(subset1, subset2)
                    if resulting_subset is None:
                        empty_set_value += mass_values1[subset1] * mass_values2[subset2]
                    else:
                        result[resulting_subset] = result.get(resulting_subset, 0) + mass_values1[subset1] * mass_values2[subset2]
        for subset in result.keys():
            result[subset] /= 1 - empty_set_value

        return IntervalMassFunction(result)


def get_intersection(a: tuple[int, int], b: tuple[int, int]) -> Optional[tuple[int, int]]:
    """
    Get intersection of two intervals. Based on
    https://scicomp.stackexchange.com/questions/26258/the-easiest-way-to-find-intersection-of-two-intervals
    :param a: Interval 1
    :param b: Interval 2
    :return: Intersecting interval or None if no intersecting interval can be derived
    """
    if b[0] > a[1] or a[0] > b[1]:
        return None
    else:
        start = max(a[0], b[0])
        end = min(a[1], b[1])

        return start, end


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

    def get_beliefs(self) -> dict[str, float]:
        """
        Calculate the belief values
        :return: Belief values
        """
        result = {}
        for subset in self._mass_values:
            if subset == '*':
                result['*'] = 1.0
                continue
            else:
                result[subset] = self._mass_values[subset]
        return result

    def get_plausibility_values(self) -> dict[str, float]:
        """
        Calculate the plausibility values
        :return: Plausibility values
        """
        result = {}
        for subset in self._mass_values:
            if subset == '*':
                result['*'] = 1.0
                continue
            else:
                result[subset] = self._mass_values[subset] + self._mass_values.get('*', 0)
        return result


def interval_df_to_subset_dict(df: pd.DataFrame, default_ignorance: float, uncertainty_object: str) -> dict[tuple[int, int], float]:
    """
    Translate dataframe to a subset dict
    :param df: The dataframe to translate
    :param default_ignorance: The ignorance to use
    :param uncertainty_object: The uncertainty object that indicates ignorance
    :return: Subset as dict
    """
    result = {'*': default_ignorance}
    n = df[df['o'] != uncertainty_object].shape[0]
    for i, x in df.iterrows():
        if x['o'] == uncertainty_object:
            result['*'] += x['weight']
        else:
            result[(x['start'], x['end'])] = x['weight'] - default_ignorance / n
    return result


def df_to_subset_dict(df: pd.DataFrame, default_ignorance: float, uncertainty_object:str) -> dict[str, float]:
    """
    Translate dataframe to a subset dict
    :param df: The dataframe to translate
    :param default_ignorance: The ignorance to use
    :param uncertainty_object: The uncertainty object that indicates ignorance
    :return: Subset as dict
    """
    result = {'*': default_ignorance}
    n = df[df['o'] != uncertainty_object].shape[0]
    for i, x in df.iterrows():
        if x['o'] == uncertainty_object:
            result['*'] += x['weight']
        else:
            result[x['o']] = x['weight'] - default_ignorance / n
    return result


if __name__ == '__main__':
    m1 = IntervalMassFunction({'*': 0.2, (-73, -72): 0.4, (-85, -70): 0.4})
    m2 = IntervalMassFunction({'*': 0.2, (-73, -72): 0.8 / 3, (-85, -70): 0.8 / 3, (-100, -73): 0.8 / 3})
    m3 = IntervalMassFunction({'*': 0.4, (-80, -75): 0.6})

    combined_mass = m1.join_masses(m2).join_masses(m3)
    joint_mass = combined_mass.get_mass_values()
    plausibilities = combined_mass.get_plausibility_values()
    beliefs = combined_mass.get_beliefs()

    for k in joint_mass.keys():
        print(f"{k} = {round(joint_mass[k], 3)}, {round(beliefs[k], 3)}, {round(plausibilities[k], 3)}")
    print(sum(joint_mass.values()))

    m1 = MassFunction({'*': 0.4, 'A': 0.3, 'B': 0.3})
    m2 = MassFunction({'*': 0.1, 'C': 0.9})

    combined_mass = m1.join_masses(m2)
    joint_mass = combined_mass.get_mass_values()
    plausibilities = combined_mass.get_plausibility_values()
    beliefs = combined_mass.get_beliefs()

    for k in joint_mass.keys():
        print(f"{k} = {round(joint_mass[k], 3)}, {round(beliefs[k], 3)}, {round(plausibilities[k], 3)}")
    print(sum(joint_mass.values()))

    m1 = MassFunction({'*': 0.4, 'ex:cn_type_1': 0.3, 'ex:cn_type_2': 0.3})
    m2 = MassFunction({'*': 0.2, 'ex:cn_type_3': 0.4, 'ex:cn_type_4': 0.4})
    m3 = MassFunction({'*': 0.4, 'ex:cn_type_5': 0.3, 'ex:cn_type_4': 0.3})

    joint_mass = m1.join_masses(m2).get_mass_values()

    for k in joint_mass.keys():
        print(f"{k} = {round(joint_mass[k], 3)}")
    print(sum(joint_mass.values()))