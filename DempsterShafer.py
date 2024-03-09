class MassFunction:
    def __init__(self, mass_values: dict):
        self.mass_values = mass_values

    def get_mass_values(self):
        return self.mass_values.copy()

    def join_masses(self, m):
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


if __name__ == '__main__':
    m1 = MassFunction({'a': 0.3, 'd': 0.3, '*': 1 - 0.6})
    m2 = MassFunction({'a': 8/30, 'b': 8/30, 'c': 8/30, '*': 1 - 0.8})
    m3 = MassFunction({'d': 0.4, 'e': 0.4, '*': 1 - 0.8})

    print(m1.join_masses(m2).join_masses(m3).get_mass_values())
    print(0.237+0.327+0.103+0.103+0.154+0.077)