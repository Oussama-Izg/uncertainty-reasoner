import logging
import pandas as pd
import time

from SparqlConnector import SparqlBaseConnector
from abc import ABC, abstractmethod
from Exceptions import ConstraintException

logger = logging.getLogger(__name__)


class Reasoner:
    def __init__(self, axioms, max_iterations=100, reasoner_name='uncertainty_reasoner'):
        self.preprocessing_axioms = []
        self.rule_reasoning_axioms = []
        self.postprocessing_axioms = []
        self.constraint_axioms = []
        self.reasoner_name = reasoner_name

        self.df_triples = pd.DataFrame()
        self.df_classes = pd.DataFrame()
        self.max_iterations = max_iterations

        for axiom in axioms:
            if axiom.get_type() == "postprocessing":
                self.postprocessing_axioms.append(axiom)
            elif axiom.get_type() == "preprocessing":
                self.preprocessing_axioms.append(axiom)
            elif axiom.get_type() == "rule_based_reasoning":
                self.rule_reasoning_axioms.append(axiom)
            elif axiom.get_type() == "constraint":
                self.constraint_axioms.append(axiom)
            else:
                raise ValueError(f"Unknown axiom type given {axiom.get_type()}")

    def load_data_from_endpoint(self, conn: SparqlBaseConnector, query=None):
        logger.info("Querying data")
        start = time.time()
        self.df_triples = conn.read_into_df(query=query)
        end = time.time()
        logger.info(f"Done in {round(end - start, 3)} seconds. Queried {self.df_triples.shape[0]} rows.")


    def _compare_dataframes(self, df_before, df_after):
        change = False
        df_result = pd.concat([df_before, df_after]).drop_duplicates(subset=['s', 'p', 'o', 'certainty'], keep=False).reset_index()
        if df_result.shape[0] != 0:
            change = True
        else:
            return df_before
        df_result['model'] = df_result['model'].fillna(self.reasoner_name)
        df_result = df_result.sort_values(by=['certainty'])
        df_result = df_result.drop_duplicates(subset=['s', 'p', 'o'])
        df_result = pd.concat([df_before, df_result])
        df_result = df_result.sort_values(by=['certainty'])
        df_result = df_result.drop_duplicates(subset=['s', 'p', 'o'])

        return df_result

    def reason(self):
        logger.info(f"Starting reasoning.")
        start_reasoning = time.time()
        df_before = self.df_triples.copy()
        if len(self.preprocessing_axioms) != 0:
            logger.info(f"Starting preprocessing.")
            start_postprocessing = time.time()
            for axiom in self.preprocessing_axioms:
                self.df_triples = axiom.reason(self.df_triples)
            end_postprocessing = time.time()

            logger.info(f"Preprocessing done in {round(end_postprocessing - start_postprocessing, 3)} seconds.")
        df_triples_with_model = self.df_triples[~self.df_triples['model'].isna()].copy()
        self.df_triples = self.df_triples[self.df_triples['model'].isna()].copy()
        if len(self.rule_reasoning_axioms) != 0:
            logger.info(f"Starting rule based reasoning.")
            start_rule_reasoning = time.time()
            counter = 0
            for i in range(self.max_iterations):
                counter += 1
                df_old = self.df_triples.copy()
                for axiom in self.rule_reasoning_axioms:
                    self.df_triples = axiom.reason(self.df_triples)
                if pd.concat([df_old, self.df_triples]).drop_duplicates(subset=['s', 'p', 'o', 'certainty'], keep=False).shape[0] == 0:
                    break
            end_rule_reasoning = time.time()

            logger.info(f"Rule based reasoning done after {counter} iterations in {round(end_rule_reasoning - start_rule_reasoning, 3)} seconds.")

        if len(self.postprocessing_axioms) != 0:
            logger.info(f"Starting postprocessing.")
            start_postprocessing = time.time()
            for axiom in self.postprocessing_axioms:
                self.df_triples = axiom.reason(self.df_triples)
            end_postprocessing = time.time()

            logger.info(f"Postprocessing done in {round(end_postprocessing - start_postprocessing, 3)} seconds.")

        self.df_triples = self._compare_dataframes(df_before, self.df_triples)
        self.df_triples = pd.concat([self.df_triples, df_triples_with_model])
        end_reasoning = time.time()
        logger.info(f"Reasoning done in {round(end_reasoning - start_reasoning, 3)} seconds.")

    def save_data_to_file(self, file_name, conn: SparqlBaseConnector, only_new: bool = False):
        with open(file_name, "rw") as f:
            if only_new:
                f.write(conn.df_to_turtle(self.df_triples['model'] == self.reasoner_name))
            else:
                f.write(conn.df_to_turtle(self.df_triples))

    def upload_data_to_endpoint(self, conn: SparqlBaseConnector):
        conn.upload_df(self.df_triples)


class Axiom(ABC):
    def __init__(self, type):
        self.type = type

    def get_type(self):
        return self.type

    @abstractmethod
    def reason(self, df_triples):
        pass


class AggregationAxiom(Axiom):

    def __init__(self, source_predicate, target_predicate, aggregation_type):
        super().__init__("preprocessing")
        self.source_predicate = source_predicate
        self.target_predicate = target_predicate

        if aggregation_type not in ['mean', 'median']:
            raise ValueError("aggregation_type must be mean or median")

        self.aggregation_type = aggregation_type

    def reason(self, df_triples: pd.DataFrame):
        df_agg = df_triples[df_triples['p'] == self.source_predicate].copy()
        df_agg = df_agg.groupby(['s', 'p', 'o', 's_class', 'o_class'])[['certainty']]

        if self.aggregation_type == 'mean':
            df_agg = df_agg.mean()
        elif self.aggregation_type == 'median':
            df_agg = df_agg.median()
        else:
            raise ValueError("aggregation_type must be mean or median")

        df_agg = df_agg.reset_index()

        return pd.concat([df_triples, df_agg])


class UncertaintyAssignmentAxiom (Axiom):
    def __init__(self, predicate):
        super().__init__("preprocessing")
        self.predicate = predicate

    def reason(self, df_triples: pd.DataFrame):
        df_agg = df_triples[df_triples['p'] == self.predicate].copy()
        df_agg = df_agg.groupby(['s', 'p'])[['o']]
        df_agg = df_agg.count()

        df_agg = df_agg.reset_index()
        df_agg = df_agg.rename(columns={'o': 'count'})

        df_triples = pd.merge(df_triples, df_agg, on=['s', 'p'])
        df_triples['certainty'] = 1 / df_triples['count']
        df_triples = df_triples.drop(columns=['count'])

        return df_triples


class AFEDempsterShaferAxiom(Axiom):
    def __init__(self, predicate, ignorance_object='ex:ignorance', ignorance=0.2):
        super().__init__("preprocessing")
        self.predicate = predicate
        self.ignorance = ignorance
        self.ignorance_object = ignorance_object

    def reason(self, df_triples: pd.DataFrame):
        df_agg = df_triples[(df_triples['p'] == self.predicate) & (df_triples['o'] != self.ignorance_object)].copy()
        df_agg = df_agg.groupby(['s', 'p'])[['o']]
        df_agg = df_agg.count()

        df_agg = df_agg.reset_index()
        df_agg = df_agg.rename(columns={'o': 'count'})

        df_triples = pd.merge(df_triples, df_agg, on=['s', 'p'])
        df_triples['certainty'] = df_triples['certainty'] - self.ignorance / df_triples['count']
        df_triples = df_triples.drop(columns=['count'])

        return df_triples


class NormalizationAxiom(Axiom):
    def __init__(self, predicate):
        super().__init__("postprocessing")
        self.predicate = predicate

    def reason(self, df_triples: pd.DataFrame):
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

    def reason(self, df_triples: pd.DataFrame):
        df_tmp = df_triples[df_triples['p'] == self.predicate].copy()
        df_tmp = df_tmp.rename(columns={'s': 's_t'})
        df_tmp = df_tmp.rename(columns={'s_t': 'o', 'o': 's'})
        df_triples = pd.concat([df_triples, df_tmp]).sort_values(by='certainty', ascending=True)
        df_triples = df_triples.drop_duplicates(subset=['s', 'p', 'o'], keep='last')

        return df_triples


class RoleChainAxiom(Axiom):
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

    def reason(self, df_triples):
        df_triples_left = df_triples[df_triples['p'] == self.antecedent1]
        df_triples_right = df_triples[df_triples['p'] == self.antecedent2]
        if self.class1:
            df_triples_left = df_triples_left[df_triples_left['s_class'] == self.class1]
        if self.class1:
            df_triples_left = df_triples_left[df_triples_left['o_class'] == self.class2]
            df_triples_right = df_triples_right[df_triples_right['s_class'] == self.class2]
        if self.class3:
            df_triples_right = df_triples_right[df_triples_right['o_class'] == self.class3]

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
        super().__init__("constraint")
        self.predicate1 = predicate1
        self.predicate2 = predicate2

    def reason(self, df_triples: pd.DataFrame):
        df_tmp = df_triples[(df_triples['p'] == self.predicate1) | (df_triples['p'] == self.predicate2)].groupby(['s', 'o'])['p'].count()
        df_tmp = df_tmp.reset_index()

        if (df_tmp['o'] != 1).any():
            raise ConstraintException(f"Constraint violation for predicates {self.predicate1} and {self.predicate2}")

        return df_triples


class SelfDisjointAxiom(Axiom):
    def __init__(self, predicate):
        super().__init__("constraint")
        self.predicate = predicate

    def reason(self, df_triples: pd.DataFrame):
        df_tmp = df_triples[df_triples['p'] == self.predicate]
        if (df_tmp['s'] == df_tmp['o']).sum() != 0:
            raise ConstraintException(f"Constraint violation for predicate {self.predicate}")

        return df_triples

