import logging
import pandas as pd
import time

from SparqlConnector import SparqlBaseConnector
from abc import ABC, abstractmethod
from Exceptions import ConstraintException

logger = logging.getLogger(__name__)


class Reasoner:
    def __init__(self, axioms, max_iterations=100, class_predicate="rdf:type"):
        self.preprocessing_axioms = []
        self.rule_reasoning_axioms = []
        self.postprocessing_axioms = []
        self.constraint_axioms = []

        self.df_triples = pd.DataFrame()
        self.df_classes = pd.DataFrame()
        self.max_iterations = max_iterations
        self.class_predicate = class_predicate

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
        self.df_classes = self.df_triples[self.df_triples['p'] == self.class_predicate].copy()
        self.df_classes = self.df_classes[['s', 'o']]
        self.df_classes = self.df_classes.rename(columns={'o': 'class'})
        end = time.time()
        # 23.52 1498500
        logger.info(f"Done in {round(end - start, 3)} seconds. Queried {self.df_triples.shape[0]} rows.")

    def reason(self):
        logger.info(f"Starting reasoning.")
        start_reasoning = time.time()
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
                if pd.concat([df_old, self.df_triples]).drop_duplicates(keep=False).shape[0] == 0:
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
        self.df_triples = pd.concat([self.df_triples, df_triples_with_model])
        end_reasoning = time.time()
        logger.info(f"Reasoning done in {round(end_reasoning - start_reasoning, 3)} seconds.")

    def save_data_to_file(self, file_name, conn: SparqlBaseConnector):
        with open(file_name, "rw") as f:
            f.write(conn.df_to_turtle(self.df_triples))

    def upload_data_to_endpoint(self, conn: SparqlBaseConnector):
        conn.upload_df(self.df_triples)


class Axiom(ABC):
    def __init__(self, type):
        self.type = type

    def get_type(self):
        return self.type

    @abstractmethod
    def reason(self, df):
        pass


class AggregationAxiom(Axiom):

    def __init__(self, source_predicate, target_predicate, aggregation_type):
        super().__init__("preprocessing")
        self.source_predicate = source_predicate
        self.target_predicate = target_predicate

        if aggregation_type not in ['mean', 'median']:
            raise ValueError("aggregation_type must be mean or median")

        self.aggregation_type = aggregation_type

    def reason(self, df: pd.DataFrame):
        df_agg = df[df['p'] == self.source_predicate].copy()
        df_agg = df_agg.groupby(['s', 'p', 'o'])[['certainty']]

        if self.aggregation_type == 'mean':
            df_agg = df_agg.mean()
        elif self.aggregation_type == 'median':
            df_agg = df_agg.median()
        else:
            raise ValueError("aggregation_type must be mean or median")

        df_agg = df_agg.reset_index()

        return pd.concat([df, df_agg])


class UncertaintyAssignmentAxiom (Axiom):
    def __init__(self, predicate):
        super().__init__("preprocessing")
        self.predicate = predicate

    def reason(self, df: pd.DataFrame):
        df_agg = df[df['p'] == self.predicate].copy()
        df_agg = df_agg.groupby(['s', 'p'])[['o']]
        df_agg = df_agg.count()

        df_agg = df_agg.reset_index()
        df_agg = df_agg.rename(columns={'o': 'count'})

        df = pd.merge(df, df_agg, on=['s', 'p'])
        df['certainty'] = 1/df['count']
        df = df.drop(columns=['count'])

        return df


class AFEDempsterShaferAxiom(Axiom):
    def __init__(self, predicate, ignorance_object='ex:ignorance', ignorance=0.2):
        super().__init__("preprocessing")
        self.predicate = predicate
        self.ignorance = ignorance
        self.ignorance_object = ignorance_object

    def reason(self, df: pd.DataFrame):
        df_agg = df[(df['p'] == self.predicate) & (df['o'] != self.ignorance_object)].copy()
        df_agg = df_agg.groupby(['s', 'p'])[['o']]
        df_agg = df_agg.count()

        df_agg = df_agg.reset_index()
        df_agg = df_agg.rename(columns={'o': 'count'})

        df = pd.merge(df, df_agg, on=['s', 'p'])
        df['certainty'] = df['certainty'] - self.ignorance / df['count']
        df = df.drop(columns=['count'])

        return df


class NormalizationAxiom(Axiom):
    def __init__(self, predicate):
        super().__init__("postprocessing")
        self.predicate = predicate

    def reason(self, df: pd.DataFrame):
        df_agg = df[df['p'] == self.predicate].copy()
        df_agg = df_agg.groupby(['s', 'p', 'o'])[['certainty']]
        df_agg = df_agg.sum()

        df_agg = df_agg.reset_index()
        df_agg = df_agg.rename(columns={'certainty': 'sum'})

        df = pd.merge(df, df_agg, on=['s', 'p', 'o'])
        df['certainty'] = 1 / df['sum']
        df = df.drop(columns=['sum'])

        return df


class InverseAxiom(Axiom):
    def __init__(self, predicate):
        super().__init__('rule_based_reasoning')
        self.predicate = predicate

    def reason(self, df: pd.DataFrame):
        df_tmp = df[df['p'] == self.predicate].copy()
        df_tmp = df_tmp.rename(columns={'s': 's_t'})
        df_tmp = df_tmp.rename(columns={'s_t': 'o', 'o': 's'})
        df = pd.concat([df, df_tmp]).sort_values(by='certainty', ascending=True)
        df = df.drop_duplicates(subset=['s', 'p', 'o'], keep='last')

        return df


class RoleChainAxiom:
    def __init__(self):
        pass


class DisjointAxiom(Axiom):
    def __init__(self, predicate1, predicate2):
        super().__init__("constraint")
        self.predicate1 = predicate1
        self.predicate2 = predicate2

    def reason(self, df: pd.DataFrame):
        df_tmp = df[(df['p'] == self.predicate1) | (df['p'] == self.predicate2)].groupby(['s', 'o'])['p'].count()
        df_tmp = df_tmp.reset_index()

        if (df_tmp['o'] != 1).any():
            raise ConstraintException(f"Constraint violation for predicates {self.predicate1} and {self.predicate2}")

        return df


class SelfDisjointAxiom(Axiom):
    def __init__(self, predicate):
        super().__init__("constraint")
        self.predicate = predicate

    def reason(self, df: pd.DataFrame):
        df_tmp = df[df['p'] == self.predicate]
        if (df_tmp['s'] == df_tmp['o']).sum() != 0:
            raise ConstraintException(f"Constraint violation for predicate {self.predicate}")

        return df

