import logging
import pandas as pd
import numpy as np
import time

import DempsterShafer
from SparqlConnector import SparqlBaseConnector
from abc import ABC, abstractmethod
from Exceptions import ConstraintException

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
        self._df_triples, self._df_classes = conn.download_df(query=query)
        self._df_triples['weight'] = self._df_triples['weight'].fillna(1.0)
        end = time.time()
        logger.info(f"Done in {round(end - start, 3)} seconds. Queried {self._df_triples.shape[0]} rows.")

    def _compare_dataframes(self, df_before, df_after):
        # Find rows with new values
        df_result = pd.concat([df_before, df_after]).drop_duplicates(subset=['s', 'p', 'o', 'weight'], keep=False).reset_index(drop=True)
        # Set reasoner name as model
        df_result['model'] = df_result['model'].fillna(self._reasoner_name)
        df_result = pd.concat([df_before, df_result])

        # Just a quick check, that only the highest certainties are used
        df_result = df_result.sort_values(by=['weight'], ascending=True)
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
                self._df_triples = axiom.reason(self._df_triples, self._df_classes).reset_index(drop=True)
                self._df_triples = self._df_triples[['s', 'p', 'o', 'weight', 'model']]
            end_postprocessing = time.time()

            logger.info(f"Preprocessing done in {round(end_postprocessing - start_postprocessing, 3)} seconds.")
        self._df_triples = self._df_triples[self._df_triples['model'].isna()].copy()
        if len(self._rule_reasoning_axioms) != 0:
            logger.info(f"Starting rule based reasoning.")
            start_rule_reasoning = time.time()
            counter = 0
            for i in range(self._max_iterations):
                logger.info(f"Iteration {i+1} of max {self._max_iterations} iteration")
                counter += 1
                df_old = self._df_triples.copy()
                for axiom in self._rule_reasoning_axioms:
                    self._df_triples = axiom.reason(self._df_triples, self._df_classes)
                if pd.concat([df_old, self._df_triples]).drop_duplicates(subset=['s', 'p', 'o', 'weight'], keep=False).shape[0] == 0:
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
        df_agg = df_agg.groupby(['s', 'p', 'o'])[['weight']]

        if self.aggregation_type == 'mean':
            df_agg = df_agg.mean()
        elif self.aggregation_type == 'median':
            df_agg = df_agg.median()
        else:
            raise ValueError("aggregation_type must be mean or median")

        df_agg = df_agg.reset_index()

        return pd.concat([df_triples, df_agg]).drop_duplicates()


class UncertaintyAssignmentAxiom (Axiom):
    def __init__(self, predicate, uncertainty_object="ex:uncertain", uncertainty_value=0.2):
        super().__init__("preprocessing")
        self.predicate = predicate
        self.uncertainty_object = uncertainty_object
        self.uncertainty_value = uncertainty_value

    def reason(self, df_triples: pd.DataFrame, df_classes):
        df_selected_triples = df_triples[df_triples['p'] == self.predicate].copy()
        df_triples = df_triples[df_triples['p'] != self.predicate]
        df_agg = df_selected_triples[df_selected_triples['o'] != self.uncertainty_object].copy()
        df_agg = df_agg.groupby(['s', 'p', 'model'])[['o']]
        df_agg = df_agg.count()

        df_agg = df_agg.reset_index()
        df_agg = df_agg.rename(columns={'o': 'count'})

        df_selected_triples = pd.merge(df_selected_triples, df_agg, on=['s', 'p', 'model'])
        df_selected_triples = pd.merge(df_selected_triples, df_selected_triples[df_selected_triples['o'] == self.uncertainty_object][['s', 'p', 'model']],
                                       on=['s', 'p', 'model'], how='left', indicator=True)
        df_selected_triples['weight'] = 1 / df_selected_triples['count']
        df_selected_triples.loc[df_selected_triples['_merge'] == 'both', 'weight'] = (1 - self.uncertainty_value) / df_selected_triples['count']
        df_selected_triples.loc[df_selected_triples['o'] == self.uncertainty_object, 'weight'] = self.uncertainty_value

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
        df_selected_subjects = df_selected_triples['s'].drop_duplicates().reset_index(drop=True)
        n = df_selected_subjects.shape[0]
        for i, s in df_selected_subjects.items():
            if (i+1) % 1000 == 0:
                logger.info(f"DempsterShaferAxiom: Processed {i+1}/{n} subjects for {self.predicate}")
            df_subsets = df_selected_triples[df_selected_triples['s'] == s]
            joint_mass_function = None
            if df_subsets['model'].drop_duplicates().shape[0] == 1:
                result = pd.concat([result, df_subsets])
                continue

            for j, model in df_subsets['model'].drop_duplicates().items():
                df_model_subsets = df_subsets[df_subsets['model'] == model]
                if joint_mass_function is None:
                    joint_mass_function = DempsterShafer.MassFunction(DempsterShafer.df_to_subset_dict(df_model_subsets, self.ignorance, self.ignorance_object))
                else:
                    joint_mass_function = joint_mass_function.join_masses(DempsterShafer.MassFunction(DempsterShafer.df_to_subset_dict(df_model_subsets, self.ignorance, self.ignorance_object)))

            mass_values = joint_mass_function.get_mass_values()
            result_tmp = {
                's': [],
                'p': [],
                'o': [],
                'weight': []
            }
            for issuer in mass_values:
                if issuer == "*":
                    result_tmp['s'].append(s)
                    result_tmp['p'].append(self.predicate)
                    result_tmp['o'].append(self.ignorance_object)
                    result_tmp['weight'].append(mass_values[issuer])
                    continue
                result_tmp['s'].append(s)
                result_tmp['p'].append(self.predicate)
                result_tmp['o'].append(issuer)
                result_tmp['weight'].append(mass_values[issuer])
            result_tmp = pd.DataFrame(result_tmp)
            result = pd.concat([result, result_tmp])
        result['model'] = np.nan
        df_triples = pd.concat([df_triples[df_triples['p'] != self.predicate], result])
        return df_triples


class AFEDempsterShaferAxiom(Axiom):
    def __init__(self, issuer_predicate='ex:issuer', issuing_for_predicate='ex:issuing_for', domain_knowledge_predicate='ex:domain_knowledge', ignorance_object='ex:uncertain', ignorance=0.2):
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
        result = pd.DataFrame()
        for i, coin in df_issuer['s'].drop_duplicates().items():
            df_issuer_subsets = df_issuer[df_issuer['s'] == coin]
            df_issuing_for_subsets = df_issuing_for[df_issuing_for['s'] == coin]
            if df_issuing_for_subsets.shape[0] == 0:
                result = pd.concat([result, df_issuer_subsets])
                continue
            issuer_mass_function = DempsterShafer.MassFunction(DempsterShafer.df_to_subset_dict(df_issuer_subsets, self.ignorance, self.ignorance_object))

            issuing_for_ignorance = self.ignorance
            df_issuing_for_ignorance = df_issuing_for_subsets[df_issuing_for_subsets['o'] == self.ignorance_object]
            if df_issuing_for_ignorance.shape[0] == 1:
                issuing_for_ignorance += df_issuing_for_ignorance['weight'].iloc[0]
            df_issuing_for_subsets = df_issuing_for_subsets[df_issuing_for_subsets['o'] != self.ignorance_object]

            for j, issuing_for in df_issuing_for_subsets['o'].items():
                df_domain_knowledge_subsets = df_domain_knowledge[df_domain_knowledge['s'] == issuing_for]
                domain_knowledge_mass_function = DempsterShafer.MassFunction(DempsterShafer.df_to_subset_dict(df_domain_knowledge_subsets, issuing_for_ignorance, self.ignorance_object))
                issuer_mass_function = issuer_mass_function.join_masses(domain_knowledge_mass_function)
            result_tmp = {
                's': [],
                'p': [],
                'o': [],
                'weight': []
            }
            mass_values = issuer_mass_function.get_mass_values()
            for issuer in mass_values:
                if issuer == "*":
                    result_tmp['s'].append(coin)
                    result_tmp['p'].append(self.issuer_predicate)
                    result_tmp['o'].append(self.ignorance_object)
                    result_tmp['weight'].append(mass_values[issuer])
                    continue
                result_tmp['s'].append(coin)
                result_tmp['p'].append(self.issuer_predicate)
                result_tmp['o'].append(issuer)
                result_tmp['weight'].append(mass_values[issuer])
            result_tmp = pd.DataFrame(result_tmp)
            result = pd.concat([result, result_tmp])
        df_triples = pd.concat([df_triples[df_triples['p'] != self.issuer_predicate], result])
        return df_triples


class NormalizationAxiom(Axiom):
    def __init__(self, predicate):
        super().__init__("postprocessing")
        self.predicate = predicate

    def reason(self, df_triples: pd.DataFrame, df_classes):
        df_agg = df_triples[df_triples['p'] == self.predicate].copy()
        df_agg = df_agg.groupby(['s', 'p', 'o'])[['weight']]
        df_agg = df_agg.sum()

        df_agg = df_agg.reset_index()
        df_agg = df_agg.rename(columns={'weight': 'sum'})

        df_triples = pd.merge(df_triples, df_agg, on=['s', 'p', 'o'])
        df_triples['weight'] = 1 / df_triples['sum']
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
        df_triples = pd.concat([df_triples, df_tmp]).sort_values(by='weight', ascending=True)
        df_triples = df_triples.drop_duplicates(subset=['s', 'p', 'o'], keep='last')

        return df_triples


class ChainRuleAxiom(Axiom):
    def __init__(self, antecedent1, antecedent2, consequent, reasoning_logic, sum_values=False, class_1=None, class_2=None,
                 class_3=None, input_threshold=None, output_threshold=None):
        super().__init__('rule_based_reasoning')
        self.antecedent1 = antecedent1
        self.antecedent2 = antecedent2
        self.consequent = consequent

        self.input_threshold = input_threshold
        self.output_threshold = output_threshold
        self.sum_values = sum_values

        self.class1 = class_1
        self.class2 = class_2
        self.class3 = class_3
        if reasoning_logic not in ['product', 'goedel', 'lukasiewicz']:
            raise ValueError("Reasoning logic must be product, goedel or lukasiewicz.")
        self.reasoning_logic = reasoning_logic

    def reason(self, df_triples, df_classes):
        df_selected_triples = df_triples[(df_triples['p'] == self.antecedent1) |
                                         (df_triples['p'] == self.antecedent2)].copy()

        # Enforce input threshold
        if self.input_threshold:
            df_selected_triples = df_selected_triples[df_selected_triples['weight'] > self.input_threshold]

        df_triples_left = df_selected_triples[df_selected_triples['p'] == self.antecedent1]
        df_triples_right = df_selected_triples[df_selected_triples['p'] == self.antecedent2]

        # Enforce classes by inner-merging with df_classes
        if self.class1:
            df_triples_left = pd.merge(df_triples_left, df_classes[df_classes['class'] == self.class1],
                                       right_on='s', left_on='node')
            df_triples_left = df_triples_left.drop(columns=['class'])
        if self.class2:
            df_triples_left = pd.merge(df_triples_left, df_classes[df_classes['class'] == self.class2],
                                       right_on='o', left_on='node')
            df_triples_left = df_triples_left.drop(columns=['class'])
            df_triples_right = pd.merge(df_triples_right, df_classes[df_classes['class'] == self.class2],
                                        right_on='s', left_on='node')
            df_triples_right = df_triples_right.drop(columns=['class'])
        if self.class3:
            df_triples_right = pd.merge(df_triples_right, df_classes[df_classes['class'] == self.class3],
                                        right_on='o', left_on='node')
            df_triples_right = df_triples_right.drop(columns=['class'])
        # Merge antecedent triples
        df_result = pd.merge(df_triples_left, df_triples_right, left_on='o', right_on='s')

        # Apply reasoning logic
        if self.reasoning_logic == 'product':
            df_result['weight'] = df_result['weight_x'] * df_result['weight_y']
        elif self.reasoning_logic == 'goedel':
            df_result['weight'] = df_result['weight_x']
            df_result.loc[df_result['certainty_x'] > df_result['weight_y'], 'weight'] = df_result['weight_y']
        elif self.reasoning_logic == 'lukasiewicz':
            df_result['weight'] = df_result['weight_x'] + df_result['weight_x'] - 1.0
            df_result.loc[0 > df_result['weight'], 'weight'] = 0.0
        else:
            raise ValueError("Reasoning logic must be product, goedel or lukasiewicz.")

        rename_dict = {
            's_x': 's',
            'o_y': 'o',
        }
        df_result['p'] = self.consequent
        df_result = df_result.rename(columns=rename_dict)
        df_result = df_result[['s', 'p', 'o', 'weight']]

        # Sum values
        if self.sum_values:
            df_result = df_result.groupby(['s', 'p', 'o']).agg({'weight': 'sum'}).reset_index()

        # Apply output threshold
        if self.output_threshold:
            df_result = df_result[df_result['weight'] > self.output_threshold]

        df_triples = pd.concat([df_triples, df_result])

        # Only keep the triple with the highest weight
        df_triples = df_triples.sort_values(by='weight', ascending=True)
        df_triples = df_triples.drop_duplicates(subset=['s', 'p', 'o'], keep='last')

        return df_triples


class DisjointAxiom(Axiom):
    def __init__(self, predicate1, predicate2, throw_exception=True, keep_predicate1=True):
        super().__init__("rule_based_reasoning")
        self.predicate1 = predicate1
        self.predicate2 = predicate2
        self.throw_exception = throw_exception
        self.keep_predicate1 = keep_predicate1

    def reason(self, df_triples: pd.DataFrame, df_classes):
        if self.throw_exception:
            # There should not be both predicates for the same s,o combination
            df_tmp = df_triples[(df_triples['p'] == self.predicate1) | (df_triples['p'] == self.predicate2)].groupby(['s', 'o'])['p'].count()
            df_tmp = df_tmp.reset_index()

            if (df_tmp['o'] != 1).any():
                raise ConstraintException(f"Constraint violation for predicates {self.predicate1} and {self.predicate2}")
        else:
            df_predicate1 = df_triples[(df_triples['p'] == self.predicate1)]
            df_predicate2 = df_triples[(df_triples['p'] == self.predicate2)]

            df_triples = df_triples[(df_triples['p'] != self.predicate1) & (df_triples['p'] != self.predicate2)]

            df_tmp = pd.concat([df_predicate1, df_predicate2])

            # Only keep one of the predicates
            if self.keep_predicate1:
                df_tmp = df_tmp.drop_duplicates(subset=['s', 'o'], keep='first')
            else:
                df_tmp = df_tmp.drop_duplicates(subset=['s', 'o'], keep='last')

            df_triples = pd.concat([df_triples, df_tmp])
        return df_triples


class SelfDisjointAxiom(Axiom):
    def __init__(self, predicate, throw_exception=True):
        super().__init__("rule_based_reasoning")
        self.predicate = predicate
        self.throw_exception = throw_exception

    def reason(self, df_triples: pd.DataFrame, df_classes):
        df_tmp = df_triples[df_triples['p'] == self.predicate]
        if (df_tmp['s'] == df_tmp['o']).sum() != 0 and self.throw_exception:
            raise ConstraintException(f"Constraint violation for predicate {self.predicate}")
        else:
            df_triples = df_triples[(df_triples['p'] == self.predicate) | (df_triples['s'] != df_triples['o'])]

        return df_triples
