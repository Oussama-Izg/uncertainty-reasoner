import logging
import pandas as pd
import time

from SparqlConnector import SparqlBaseConnector
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)
class Reasoner:
    def __init__(self, axioms, max_iterations=100):
        self.preprocessing_axioms = []
        self.rule_reasoning_axioms = []
        self.postprocessing_axioms = []
        self.constraint_axioms = []

        self.df = pd.DataFrame()
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
                raise Exception(f"Unknown axiom type given {type(axiom)}")

    def load_data_from_endpoint(self, conn: SparqlBaseConnector, query=None):
        logger.info("Querying data")
        start = time.time()
        self.df = conn.read_into_df(query=query)
        end = time.time()
        # 23.52 1498500
        logger.info(f"Done in {round(end - start, 3)} seconds. Queried {self.df.shape[0]} rows.")

    def reason(self):
        logger.info(f"Starting reasoning.")
        start_reasoning = time.time()
        if len(self.preprocessing_axioms) != 0:
            logger.info(f"Starting preprocessing.")
            start_postprocessing = time.time()
            for axiom in self.preprocessing_axioms:
                self.df = axiom.reason(self.df)
            end_postprocessing = time.time()

            logger.info(f"Preprocessing done in {round(end_postprocessing - start_postprocessing, 3)} seconds.")

        if len(self.rule_reasoning_axioms) != 0:
            logger.info(f"Starting rule based reasoning.")
            start_rule_reasoning = time.time()
            counter = 0
            for i in range(self.max_iterations):
                counter += 1
                df_old = self.df.copy()
                for axiom in self.rule_reasoning_axioms:
                    self.df = axiom.reason(self.df)
                if pd.concat([df_old, self.df]).drop_duplicates(keep=False).shape[0] == 0:
                    break
            end_rule_reasoning = time.time()

            logger.info(f"Rule based reasoning done after {counter} iterations in {round(end_rule_reasoning - start_rule_reasoning, 3)} seconds.")

        if len(self.postprocessing_axioms) != 0:
            logger.info(f"Starting postprocessing.")
            start_postprocessing = time.time()
            for axiom in self.postprocessing_axioms:
                self.df = axiom.reason(self.df)
            end_postprocessing = time.time()

            logger.info(f"Postprocessing done in {round(end_postprocessing - start_postprocessing, 3)} seconds.")
        end_reasoning = time.time()
        logger.info(f"Reasoning done in {round(end_reasoning - start_reasoning, 3)} seconds.")

    def save_data_to_file(self, file_name, conn: SparqlBaseConnector):
        with open(file_name, "rw") as f:
            f.write(conn.df_to_turtle(self.df))

    def upload_data_to_endpoint(self, conn: SparqlBaseConnector):
        conn.upload_df(self.df)


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

        if aggregation_type not in ['mean', 'median', 'dst']:
            raise Exception("aggregation_type must be mean, median or dst")

        self.aggregation_type = aggregation_type

    def reason(self, df: pd.DataFrame):
        df_agg = df[df['p'] == self.source_predicate].copy()
        df_agg = df_agg.groupby(['s', 'p', 'o'])[['certainty']]

        if self.aggregation_type == 'dst':
            pass
        elif self.aggregation_type == 'median':
            df_agg = df_agg.median()
        else:
            df_agg = df_agg.mean()

        df_agg = df_agg.reset_index()

        return pd.concat([df, df_agg])






class NormalizationAxiom:
    def __init__(self, property):
        self.property = property


class InverseAxiom:
    def __init__(self):
        pass


class RoleChainAxiom:
    def __init__(self):
        pass


class DisjointAxiom:
    def __init__(self):
        pass


class SelfDisjointAxiom:
    def __init__(self):
        pass
