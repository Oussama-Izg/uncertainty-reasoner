import logging
import pandas as pd
import numpy as np
import time

from SparqlConnector import SparqlBaseConnector
from abc import ABC, abstractmethod
from Exceptions import ConstraintException
from DempsterShafer import MassFunction

logger = logging.getLogger(__name__)


class Reasoner:
    def __init__(self, axioms, max_iterations=100, reasoner_name='uncertainty_reasoner'):
        self._preprocessing_axioms = []
        self._rule_reasoning_axioms = []
        self._postprocessing_axioms = []
        self._reasoner_name = reasoner_name

        self._df_triples = pd.DataFrame()
        self._df_classes = pd.DataFrame()
        self._max_iterations = max_iterations

        for axiom in axioms:
            if axiom.get_stage() == "postprocessing":
                self._postprocessing_axioms.append(axiom)
            elif axiom.get_stage() == "preprocessing":
                self._preprocessing_axioms.append(axiom)
            elif axiom.get_stage() == "rule_based_reasoning":
                self._rule_reasoning_axioms.append(axiom)
            else:
                raise ValueError(f"Unknown axiom type given {axiom.get_stage()}")

    def load_data_from_endpoint(self, conn: SparqlBaseConnector, query=None):
        logger.info("Querying data")
        start = time.time()
        self._df_triples, self._df_classes = conn.read_into_df(query=query)
        self._df_triples['certainty'] = self._df_triples['certainty'].fillna(1.0)
        end = time.time()
        logger.info(f"Done in {round(end - start, 3)} seconds. Queried {self._df_triples.shape[0]} rows.")

    def _compare_dataframes(self, df_before, df_after):
        # Find rows with new values
        df_result = pd.concat([df_before, df_after]).drop_duplicates(subset=['s', 'p', 'o', 'certainty'], keep=False).reset_index(drop=True)
        # Set reasoner name as model
        df_result['model'] = df_result['model'].fillna(self._reasoner_name)
        df_result = pd.concat([df_before, df_result])

        # Just a quick check, that only the highest certainties are used
        df_result = df_result.sort_values(by=['certainty'], ascending=True)
        df_result = df_result.drop_duplicates(subset=['s', 'p', 'o', 'model'])

        return df_result

    def reason(self):
        logger.info(f"Starting reasoning.")
        start_reasoning = time.time()
        df_before = self._df_triples.copy()
        if len(self._preprocessing_axioms) != 0:
            logger.info(f"Starting preprocessing.")
            start_postprocessing = time.time()
            for axiom in self._preprocessing_axioms:
                self._df_triples = axiom.reason(self._df_triples, self._df_classes)
                self._df_triples = self._df_triples[['s', 'p', 'o', 'certainty', 'model']]
            end_postprocessing = time.time()

            logger.info(f"Preprocessing done in {round(end_postprocessing - start_postprocessing, 3)} seconds.")
        df_triples_with_model = self._df_triples[~self._df_triples['model'].isna()].copy()
        self._df_triples = self._df_triples[self._df_triples['model'].isna()].copy()
        if len(self._rule_reasoning_axioms) != 0:
            logger.info(f"Starting rule based reasoning.")
            start_rule_reasoning = time.time()
            counter = 0
            for i in range(self._max_iterations):
                counter += 1
                df_old = self._df_triples.copy()
                for axiom in self._rule_reasoning_axioms:
                    self._df_triples = axiom.reason(self._df_triples, self._df_classes)
                if pd.concat([df_old, self._df_triples]).drop_duplicates(subset=['s', 'p', 'o', 'certainty'], keep=False).shape[0] == 0:
                    break
            end_rule_reasoning = time.time()

            logger.info(f"Rule based reasoning done after {counter} iterations in {round(end_rule_reasoning - start_rule_reasoning, 3)} seconds.")

        if len(self._postprocessing_axioms) != 0:
            logger.info(f"Starting postprocessing.")
            start_postprocessing = time.time()
            for axiom in self._postprocessing_axioms:
                self._df_triples = axiom.reason(self._df_triples, self._df_classes)
            end_postprocessing = time.time()

            logger.info(f"Postprocessing done in {round(end_postprocessing - start_postprocessing, 3)} seconds.")
        self._df_triples = self._compare_dataframes(df_before, self._df_triples)
        self._df_triples = pd.concat([self._df_triples, df_triples_with_model])
        end_reasoning = time.time()
        logger.info(f"Reasoning done in {round(end_reasoning - start_reasoning, 3)} seconds.")

    def get_triples_as_df(self):
        return self._df_triples


    def save_data_to_file(self, file_name, conn: SparqlBaseConnector, only_new: bool = False):
        with open(file_name, "rw") as f:
            if only_new:
                f.write(conn.df_to_turtle(self._df_triples['model'] == self._reasoner_name))
            else:
                f.write(conn.df_to_turtle(self._df_triples))

    def upload_data_to_endpoint(self, conn: SparqlBaseConnector):
        conn.upload_df(self._df_triples)


class Axiom(ABC):
    def __init__(self, stage):
        self._stage = stage

    def get_stage(self):
        return self._stage

    @abstractmethod
    def reason(self, df_triples, df_classes):
        pass


class AggregationAxiom(Axiom):

    def __init__(self, predicate, aggregation_type):
        super().__init__("preprocessing")
        self.predicate = predicate

        if aggregation_type not in ['mean', 'median']:
            raise ValueError("aggregation_type must be mean or median")

        self.aggregation_type = aggregation_type

    def reason(self, df_triples: pd.DataFrame, df_classes):
        df_agg = df_triples[df_triples['p'] == self.predicate].copy()
        df_agg = df_agg.groupby(['s', 'p', 'o'])[['certainty']]

        if self.aggregation_type == 'mean':
            df_agg = df_agg.mean()
        elif self.aggregation_type == 'median':
            df_agg = df_agg.median()
        else:
            raise ValueError("aggregation_type must be mean or median")

        df_agg = df_agg.reset_index()

        return pd.concat([df_triples, df_agg]).drop_duplicates()


class UncertaintyAssignmentAxiom (Axiom):
    def __init__(self, predicate, uncertainty_object="ex:uncertain", default_uncertainty=0.2):
        super().__init__("preprocessing")
        self.predicate = predicate
        self.uncertainty_object = uncertainty_object
        self.default_uncertainty = default_uncertainty

    def reason(self, df_triples: pd.DataFrame, df_classes):
        df_selected_triples = df_triples[df_triples['p'] == self.predicate].copy()
        df_triples = df_triples[df_triples['p'] != self.predicate]
        df_agg = df_selected_triples[df_selected_triples['o'] != self.uncertainty_object].copy()
        df_agg = df_agg.groupby(['s', 'p'])[['o']]
        df_agg = df_agg.count()

        df_agg = df_agg.reset_index()
        df_agg = df_agg.rename(columns={'o': 'count'})

        df_selected_triples = pd.merge(df_selected_triples, df_agg, on=['s', 'p'])
        df_selected_triples = pd.merge(df_selected_triples, df_selected_triples[df_selected_triples['o'] == self.uncertainty_object][['s', 'p']],
                                       on=['s', 'p'], how='left', indicator=True)
        df_selected_triples['certainty'] = 1 / df_selected_triples['count']
        df_selected_triples.loc[df_selected_triples['_merge'] == 'both', 'certainty'] = (1 - self.default_uncertainty) / df_selected_triples['count']
        df_selected_triples.loc[df_selected_triples['o'] == self.uncertainty_object, 'certainty'] = self.default_uncertainty

        df_selected_triples = df_selected_triples.drop(columns=['count'])

        return pd.concat([df_selected_triples, df_triples])


class DempsterShaferAxiom(Axiom):
    def __init__(self, predicate, ignorance_object='ex:uncertain', ignorance=0.2):
        super().__init__("preprocessing")
        self.predicate = predicate
        self.ignorance = ignorance
        self.ignorance_object = ignorance_object

    def reason(self, df_triples: pd.DataFrame, df_classes):
        df_selected_triples = df_triples[(df_triples['p'] == self.predicate)].copy()
        result = pd.DataFrame()
        for i, s in df_selected_triples['s'].drop_duplicates().items():
            df_subsets = df_selected_triples[df_selected_triples['s'] == s]
            joinT_mass_function = None
            if len(df_subsets['model'].drop_duplicates().items()) == 1:
                result = pd.concat([result, df_subsets])

            for j, model in df_subsets['model'].drop_duplicates().items():
                df_model_subsets = df_selected_triples[df_selected_triples['model'] == model]

                issuer_ignorance = self.ignorance
                df_ignorance = df_model_subsets[df_model_subsets['o'] == self.ignorance_object]
                if df_ignorance.shape[0] == 1:
                    issuer_ignorance += df_ignorance['certainty'].iloc[0]
                df_model_subsets = df_model_subsets[df_model_subsets['o'] != self.ignorance_object]
                if joinT_mass_function is None:
                    joinT_mass_function = MassFunction(self._df_to_subset(df_model_subsets, issuer_ignorance))
                else:
                    joinT_mass_function = joinT_mass_function.join_masses(MassFunction(self._df_to_subset(df_model_subsets, issuer_ignorance)))

            mass_values = joinT_mass_function.get_mass_values()
            for issuer in mass_values:
                if issuer == "*":
                    result_tmp['s'].append(s)
                    result_tmp['p'].append(self.predicate)
                    result_tmp['o'].append(self.ignorance_object)
                    result_tmp['certainty'].append(mass_values[issuer])
                    continue
                result_tmp['s'].append(s)
                result_tmp['p'].append(self.predicate)
                result_tmp['o'].append(issuer)
                result_tmp['certainty'].append(mass_values[issuer])
            result_tmp = pd.DataFrame(result_tmp)
            result = pd.concat([result, result_tmp])
        result['model'] = np.nan
        df_triples = pd.concat([df_triples[df_triples['p'] != self.predicate], result])
        return df_triples

    def _df_to_subset(self, df: pd.DataFrame, ignorance=None):
        result = {}
        if ignorance is None:
            for i, x in df.iterrows():
                if x['o'] == self.ignorance_object:
                    result['*'] = x['certainty']
                else:
                    result[x['o']] = x['certainty']
        else:
            certainty = (1 - ignorance) / df.shape[0]
            result['*'] = ignorance
            for i, x in df.iterrows():
                result[x['o']] = certainty
        return result


class AFEDempsterShaferAxiom(Axiom):
    def __init__(self, issuer_predicate, issuing_for_predicate, domain_knowledge_predicate='ex:domain_knowledge', ignorance_object='ex:uncertain', ignorance=0.2):
        super().__init__("preprocessing")
        self.issuer_predicate = issuer_predicate
        self.ignorance = ignorance
        self.ignorance_object = ignorance_object
        self.domain_knowledge_predicate = domain_knowledge_predicate
        self.issuing_for_predicate = issuing_for_predicate

    def reason(self, df_triples: pd.DataFrame, df_classes):
        df_domain_knowledge = df_triples[(df_triples['p'] == self.domain_knowledge_predicate)].copy()
        df_issuer = df_triples[(df_triples['p'] == self.issuer_predicate)].copy()
        df_issuing_for = df_triples[(df_triples['p'] == self.issuing_for_predicate)].copy()
        # df_issuer = self._process_df(df_issuer)
        result = pd.DataFrame()
        for i, coin in df_issuer['s'].drop_duplicates().items():
            df_issuer_subsets = df_issuer[df_issuer['s'] == coin]
            df_issuing_for_subsets = df_issuing_for[df_issuing_for['s'] == coin]
            if df_issuing_for_subsets.shape[0] == 0:
                result = pd.concat([result, df_issuer_subsets])
                continue

            issuer_ignorance = self.ignorance
            df_issuer_ignorance = df_issuer_subsets[df_issuer_subsets['o'] == self.ignorance_object]
            if df_issuer_ignorance.shape[0] == 1:
                issuer_ignorance += df_issuer_ignorance['certainty'].iloc[0]
            df_issuer_subsets = df_issuer_subsets[df_issuer_subsets['o'] != self.ignorance_object]

            issuing_for_ignorance = self.ignorance
            df_issuing_for_ignorance = df_issuing_for_subsets[df_issuing_for_subsets['o'] == self.ignorance_object]
            if df_issuing_for_ignorance.shape[0] == 1:
                issuing_for_ignorance += df_issuing_for_ignorance['certainty'].iloc[0]
            df_issuing_for_subsets = df_issuing_for_subsets[df_issuing_for_subsets['o'] != self.ignorance_object]

            issuer_mass_function = MassFunction(self._df_to_subset(df_issuer_subsets, issuer_ignorance))
            for j, issuing_for in df_issuing_for_subsets['o'].items():
                df_domain_knowledge_subsets = df_domain_knowledge[df_domain_knowledge['s'] == issuing_for]
                domain_knowledge_mass_function = MassFunction(self._df_to_subset(df_domain_knowledge_subsets, issuing_for_ignorance))
                issuer_mass_function = issuer_mass_function.join_masses(domain_knowledge_mass_function)
            result_tmp = {
                's': [],
                'p': [],
                'o': [],
                'certainty': []
            }
            mass_values = issuer_mass_function.get_mass_values()
            for issuer in mass_values:
                if issuer == "*":
                    result_tmp['s'].append(coin)
                    result_tmp['p'].append(self.issuer_predicate)
                    result_tmp['o'].append(self.ignorance_object)
                    result_tmp['certainty'].append(mass_values[issuer])
                    continue
                result_tmp['s'].append(coin)
                result_tmp['p'].append(self.issuer_predicate)
                result_tmp['o'].append(issuer)
                result_tmp['certainty'].append(mass_values[issuer])
            result_tmp = pd.DataFrame(result_tmp)
            result = pd.concat([result, result_tmp])
        df_triples = pd.concat([df_triples[df_triples['p'] != self.issuer_predicate], result])
        return df_triples

    def _df_to_subset(self, df: pd.DataFrame, ignorance=None):
        result = {}
        if ignorance is None:
            for i, x in df.iterrows():
                if x['o'] == self.ignorance_object:
                    result['*'] = x['certainty']
                else:
                    result[x['o']] = x['certainty']
        else:
            certainty = (1 - ignorance) / df.shape[0]
            result['*'] = ignorance
            for i, x in df.iterrows():
                result[x['o']] = certainty
        return result


class NormalizationAxiom(Axiom):
    def __init__(self, predicate):
        super().__init__("postprocessing")
        self.predicate = predicate

    def reason(self, df_triples: pd.DataFrame, df_classes):
        df_agg = df_triples[df_triples['p'] == self.predicate].copy()
        df_agg = df_agg.groupby(['s', 'p', 'o'])[['certainty']]
        df_agg = df_agg.sum()

        df_agg = df_agg.reset_index()
        df_agg = df_agg.rename(columns={'certainty': 'sum'})

        df_triples = pd.merge(df_triples, df_agg, on=['s', 'p', 'o'])
        df_triples['certainty'] = 1 / df_triples['sum']
        df_triples = df_triples.drop(columns=['sum'])

        return df_triples


class InverseAxiom(Axiom):
    def __init__(self, predicate):
        super().__init__('rule_based_reasoning')
        self.predicate = predicate

    def reason(self, df_triples: pd.DataFrame, df_classes):
        df_tmp = df_triples[df_triples['p'] == self.predicate].copy()
        df_tmp = df_tmp.rename(columns={'s': 's_t'})
        df_tmp = df_tmp.rename(columns={'s_t': 'o', 'o': 's'})
        df_triples = pd.concat([df_triples, df_tmp]).sort_values(by='certainty', ascending=True)
        df_triples = df_triples.drop_duplicates(subset=['s', 'p', 'o'], keep='last')

        return df_triples


class ChainRuleAxiom(Axiom):
    def __init__(self, antecedent1, antecedent2, consequent, reasoning_logic, class_1=None, class_2=None,
                 class_3=None):
        super().__init__('rule_based_reasoning')
        self.antecedent1 = antecedent1
        self.antecedent2 = antecedent2
        self.consequent = consequent

        self.class1 = class_1
        self.class2 = class_2
        self.class3 = class_3
        if reasoning_logic not in ['product', 'goedel', 'lukasiewicz']:
            raise ValueError("Reasoning logic must be product, goedel or lukasiewicz.")
        self.reasoning_logic = reasoning_logic

    def reason(self, df_triples, df_classes):
        df_triples_left = df_triples[df_triples['p'] == self.antecedent1]
        df_triples_right = df_triples[df_triples['p'] == self.antecedent2]
        if self.class1:
            df_triples_left = pd.merge(df_triples_left, df_classes[df_classes['class'] == self.class1],
                                       right_on='s', left_on='node')
            df_triples_left = df_triples_left.drop(columns=['node'])
        if self.class1:
            df_triples_left = pd.merge(df_triples_left, df_classes[df_classes['class'] == self.class2],
                                       right_on='o', left_on='node')
            df_triples_left = df_triples_left.drop(columns=['node'])
            df_triples_right = pd.merge(df_triples_right, df_classes[df_classes['class'] == self.class2],
                                       right_on='s', left_on='node')
            df_triples_right = df_triples_right.drop(columns=['node'])
        if self.class3:
            df_triples_right = pd.merge(df_triples_right, df_classes[df_classes['class'] == self.class3],
                                        right_on='o', left_on='node')
            df_triples_right = df_triples_right.drop(columns=['node'])

        df_result = pd.merge(df_triples_left, df_triples_right, left_on='o', right_on='s')
        if self.reasoning_logic == 'product':
            df_result['certainty'] = df_result['certainty_x'] + df_result['certainty_y']
        elif self.reasoning_logic == 'goedel':
            df_result['certainty'] = df_result['certainty_x']
            df_result.loc[df_result['certainty_x'] > df_result['certainty_y'], 'certainty'] = df_result['certainty_y']
        elif self.reasoning_logic == 'lukasiewicz':
            df_result['certainty'] = df_result['certainty_x'] + df_result['certainty_x'] - 1.0
            df_result.loc[0 > df_result['certainty'], 'certainty'] = 0.0
        else:
            raise ValueError("Reasoning logic must be product, goedel or lukasiewicz.")
        rename_dict = {
            's_x': 's',
            's_class_x': 's_class',
            'o_y': 'o',
            'o_class_y': 'o_class',
        }
        df_result['p'] = self.consequent
        df_result = df_result.rename(columns=rename_dict)
        df_result = df_result[['s', 'p', 'o', 's_class', 'o_class', 'certainty']]

        df_triples = pd.concat([df_triples, df_result])

        df_triples = df_triples.sort_values(by='certainty', ascending=True)
        df_triples = df_triples.drop_duplicates(subset=['s', 'p', 'o'])

        return df_triples


class DisjointAxiom(Axiom):
    def __init__(self, predicate1, predicate2):
        super().__init__("rule_based_reasoning")
        self.predicate1 = predicate1
        self.predicate2 = predicate2

    def reason(self, df_triples: pd.DataFrame, df_classes):
        df_tmp = df_triples[(df_triples['p'] == self.predicate1) | (df_triples['p'] == self.predicate2)].groupby(['s', 'o'])['p'].count()
        df_tmp = df_tmp.reset_index()

        if (df_tmp['o'] != 1).any():
            raise ConstraintException(f"Constraint violation for predicates {self.predicate1} and {self.predicate2}")

        return df_triples


class SelfDisjointAxiom(Axiom):
    def __init__(self, predicate):
        super().__init__("rule_based_reasoning")
        self.predicate = predicate

    def reason(self, df_triples: pd.DataFrame, df_classes):
        df_tmp = df_triples[df_triples['p'] == self.predicate]
        if (df_tmp['s'] == df_tmp['o']).sum() != 0:
            raise ConstraintException(f"Constraint violation for predicate {self.predicate}")

        return df_triples

