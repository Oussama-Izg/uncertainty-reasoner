import logging
from typing import Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import math

import pandas as pd
import numpy as np
import time

import DempsterShafer
from SparqlConnector import SparqlBaseConnector
from abc import ABC, abstractmethod
from Exceptions import ConstraintException


logger = logging.getLogger(__name__)


class Reasoner:
    """
    Reasoner for uncertain and vague RDF graphs
    """
    def __init__(self, axioms: list['Axiom'], max_iterations: int = 100, reasoner_name: str = 'uncertainty_reasoner'):
        """
        :param axioms: List of axioms for reasoning
        :param max_iterations: Max iterations for chain rule reasoning
        :param reasoner_name: Reasoner name for output
        """
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

    def load_data_from_endpoint(self, conn: SparqlBaseConnector, query: str = None) -> None:
        """
        Load data from SPARQL endpoint
        :param conn: SPARQL connection
        :param query: Custom query
        :return:
        """
        logger.info("Querying data")
        start = time.time()
        self._df_triples, self._df_classes = conn.download_df(query=query)
        self._df_triples['weight'] = self._df_triples['weight'].fillna(1.0)
        end = time.time()
        logger.info(f"Done in {round(end - start, 3)} seconds. Queried {self._df_triples.shape[0]} rows.")

    def _compare_dataframes(self, df_before:pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
        """
        Compare dataframe before and after reasoning. New triples are flagged with reasoner_name as the model
        :param df_before: Dataframe before
        :param df_after: Dataframe after
        :return: The resulting dataframe
        """
        df_before['source'] = 'before'
        # Find rows with new values
        df_result = pd.concat([df_before, df_after], ignore_index=True)
        df_result = df_result.drop_duplicates(subset=['s', 'p', 'o', 'weight'], keep=False).reset_index(drop=True)
        df_result.to_csv('export.csv', index=False)
        # Set reasoner name as model
        df_result['model'] = df_result['model'].fillna(self._reasoner_name)
        df_result = pd.concat([df_before, df_result[df_result['source'] != 'before']], ignore_index=True)
        # Just a quick check, that only the highest certainties are used
        df_result = df_result.sort_values(by=['weight'], ascending=True)
        df_result = df_result.drop_duplicates(subset=['s', 'p', 'o', 'model'], keep='last')
        df_result = df_result.drop(columns=['source'])

        return df_result

    def reason(self) -> None:
        """
        Start the reasoning process. All preprocessing axioms are applied once. The chain rule axioms are apllied until
        the weights don't change or the maximum number of iterations is reached. Then the post-processing axioms are
        applied once.
        :return:
        """
        logger.info(f"Starting reasoning.")
        start_reasoning = time.time()
        df_before = self._df_triples.copy()
        if len(self._preprocessing_axioms) != 0:
            logger.info(f"Starting preprocessing.")
            start_postprocessing = time.time()

            visited = set()

            for i, axiom in enumerate(self._preprocessing_axioms):

                if i in visited:
                    continue

                if isinstance(axiom, AFEDempsterShaferAxiom_2):
                    axiom_group = axiom.get_group()

                    # Grouped AFEDempsterShaferAxiom_2 instances that should be run in parallel on same base data
                    if axiom_group != "Undefined":
                        group_axioms = []
                        for j, other_axiom in enumerate(self._preprocessing_axioms):
                            if j in visited:
                                continue
                            if not isinstance(other_axiom, AFEDempsterShaferAxiom_2):
                                continue
                            if other_axiom.get_group() == axiom_group:
                                group_axioms.append(other_axiom)
                        logger.debug("Grouped axioms: %s",
                                     [ax.__class__.__name__ for ax in
                                      group_axioms])

                        # Mark all as visited
                        visited.update(
                            j for j, other_axiom in enumerate(self._preprocessing_axioms)
                            if isinstance(other_axiom, AFEDempsterShaferAxiom_2)
                            and other_axiom.get_group() == axiom_group
                        )

                        # each axiom runs on its own deep copies of the SAME base data
                        def _run_axiom(ax):
                            df_t = self._df_triples.copy(deep=True)
                            df_c = self._df_classes.copy(deep=True)

                            df_result = ax.reason(df_t, df_c).reset_index(
                                drop=True)

                            # Strip helper predicate if present to keep the consistencies
                            if hasattr(ax, 'knowledge_path_predicate'):
                                df_result = df_result[
                                    df_result[
                                        'p'] != ax.knowledge_path_predicate]

                            # Ensure consistent column order
                            expected_cols = ['s', 'p', 'o', 'weight', 'model']
                            df_result = df_result[[c for c in expected_cols if
                                                   c in df_result.columns]]
                            return df_result

                        # Run in parallel
                        max_workers = min(len(group_axioms),
                                          os.cpu_count() or 4)
                        results = []
                        with ThreadPoolExecutor(max_workers=max_workers) as ex:
                            futures = {ex.submit(_run_axiom, ax): ax for ax in
                                       group_axioms}
                            for fut in as_completed(futures):
                                ax = futures[fut]
                                try:
                                    results.append(fut.result())
                                except Exception as e:
                                    logger.exception("Axiom %s failed: %s",
                                                     ax.__class__.__name__, e)
                                    raise

                        # Combine results and deduplicate
                        if results:
                            combined = pd.concat(results,
                                                 ignore_index=True).drop_duplicates()
                            self._df_triples = combined[
                                ['s', 'p', 'o', 'weight', 'model']]
                        else:
                            # If there's results, then keep the base data unchanged
                            logger.debug("No results produced by group '%s'.",
                                         axiom_group)
                    else:
                        self._df_triples = axiom.reason(self._df_triples, self._df_classes).reset_index(drop=True)
                        self._df_triples = self._df_triples[['s', 'p', 'o', 'weight', 'model']]
                        visited.add(i)

                else:
                    # For undefined group, apply directly as usual
                    self._df_triples = axiom.reason(self._df_triples, self._df_classes).reset_index(drop=True)
                    self._df_triples = self._df_triples[['s', 'p', 'o', 'weight', 'model']]
                    visited.add(i)

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
                if pd.concat([df_old, self._df_triples], ignore_index=True).drop_duplicates(subset=['s', 'p', 'o', 'weight'], keep=False).shape[0] == 0:
                    break
                logger.info(self._df_triples.shape[0])
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

    def get_triples_as_df(self) -> pd.DataFrame:
        """
        Get the df_triples dataframe
        :return: df_triples dataframe
        """
        return self._df_triples

    def save_data_to_file(self, file_name: str, conn: SparqlBaseConnector, only_new: bool = False) -> None:
        """
        Write triples to turtle file
        :param file_name: File name
        :param conn: Connection to translate triples into turtle triples
        :param only_new: Include only the new inferred triples or triple values
        :return:
        """
        with open(file_name, "w") as f:
            if only_new:
                result = self._df_triples[self._df_triples['model'] == self._reasoner_name]
            else:
                result = conn.combine_dataframes(self._df_triples, self._df_classes)
            f.write(conn.df_to_turtle(result))

    def upload_data_to_endpoint(self, conn: SparqlBaseConnector, only_new: bool = False) -> None:
        """
        Upload triple to endpoint
        :param conn: SPARQL connection to use
        :param only_new: Include only the new inferred triples or triple values
        :return:
        """
        if only_new:
            result = self._df_triples[self._df_triples['model'] == self._reasoner_name]
        else:
            result = conn.combine_dataframes(self._df_triples, self._df_classes)
        conn.upload_df(result)


class Axiom(ABC):
    """
    Abstract Axiom class to implement reasoning axioms
    """
    def __init__(self, stage: Literal['preprocessing', 'rule_based_reasoning', 'postprocessing']):
        """
        :param stage: Defines the reasoning stage: preprocessing, rule_based_reasoning or postprocessing
        """
        self._stage = stage

    def get_stage(self) -> str:
        """
        :return: Returns the stage of the axiom
        """
        return self._stage

    @abstractmethod
    def reason(self, df_triples: pd.DataFrame, df_classes: pd.DataFrame) -> pd.DataFrame:
        """
        Apply axiom on given triples dataframe
        :param df_triples: Triples dataframe
        :param df_classes: Classes dataframe
        :return: Triples dataframe where the axiom was applied
        """
        pass


class AggregationAxiom(Axiom):
    """
    Axiom to aggregate weights by model using simple aggregation functions
    """
    def __init__(self, predicate: str, aggregation_type: Literal["mean", "median"]):
        """
        :param predicate: Predicate to aggregate
        :param aggregation_type: Aggregation type
        """
        super().__init__("preprocessing")
        self.predicate = predicate

        if aggregation_type not in ['mean', 'median']:
            raise ValueError("aggregation_type must be mean or median")

        self.aggregation_type = aggregation_type

    def reason(self, df_triples: pd.DataFrame, df_classes: pd.DataFrame) -> pd.DataFrame:
        df_agg = df_triples[df_triples['p'] == self.predicate].copy()
        df_agg = df_agg.groupby(['s', 'p', 'o'])[['weight']]

        if self.aggregation_type == 'mean':
            df_agg = df_agg.mean()
        elif self.aggregation_type == 'median':
            df_agg = df_agg.median()
        else:
            raise ValueError("aggregation_type must be mean or median")

        df_agg = df_agg.reset_index()

        return pd.concat([df_triples, df_agg], ignore_index=True).drop_duplicates()


class CertaintyAssignmentAxiom(Axiom):
    """
    Uses a heuristic to assign certainty weights to triples.
    """
    def __init__(self, predicate: str, uncertainty_object: str = "ex:uncertain", uncertainty_value: float = 0.2):
        """
        :param predicate: The predicate to use
        :param uncertainty_object: Object indicating uncertainty about the selection
        :param uncertainty_value: Certainty value for the uncertainty object
        """
        super().__init__("preprocessing")
        self.predicate = predicate
        self.uncertainty_object = uncertainty_object
        self.uncertainty_value = uncertainty_value

    def reason(self, df_triples: pd.DataFrame, df_classes: pd.DataFrame) -> pd.DataFrame:
        df_selected_triples = df_triples[df_triples['p'] == self.predicate].copy()  # ex:issuer', 'ex:issuing_for' or 'ex:domain_knowledge'
        df_triples = df_triples[df_triples['p'] != self.predicate]
        df_agg = df_selected_triples[df_selected_triples['o'] != self.uncertainty_object].copy()  # ex:uncertain excluded
        df_agg = df_agg.groupby(['s', 'p', 'model'], dropna=False)[['o']]
        df_agg = df_agg.count()  # fo reach coin how many issuers, issuing_for or domain_knowledge are given

        df_agg = df_agg.reset_index()
        df_agg = df_agg.rename(columns={'o': 'count'})

        df_selected_triples = pd.merge(df_selected_triples, df_agg, on=['s', 'p', 'model'])
        df_selected_triples = pd.merge(df_selected_triples, df_selected_triples[df_selected_triples['o'] == self.uncertainty_object][['s', 'p', 'model']],
                                       on=['s', 'p', 'model'], how='left', indicator=True)
        df_selected_triples['weight'] = 1 / df_selected_triples['count']
        df_selected_triples.loc[df_selected_triples['_merge'] == 'both', 'weight'] = (1 - self.uncertainty_value) / df_selected_triples['count']
        df_selected_triples.loc[df_selected_triples['o'] == self.uncertainty_object, 'weight'] = self.uncertainty_value

        df_selected_triples = df_selected_triples.drop(columns=['count', '_merge'])
        df_selected_triples['weight'] = df_selected_triples['weight'].round(3)

        return pd.concat([df_selected_triples, df_triples])


class DempsterShaferAxiom(Axiom):
    """
    Implements a simple Dempster-Shafer combination rule to combine the weights (evidences) from multiple models. Should
    only be used for certainty weights.
    """
    def __init__(self, predicate: str, ignorance_object: str = 'ex:uncertain', ignorance: dict[str, float] = None,
                 default_ignorance: float = 0.2):
        """
        :param predicate: Predicate to aggregate
        :param ignorance_object: Object that increases ignorance for the mass function
        :param default_ignorance: Default ignorance
        """
        super().__init__("preprocessing")
        if ignorance is None:
            ignorance = {}
        self.predicate = predicate
        self.default_ignorance = default_ignorance
        self.ignorance = ignorance
        self.ignorance_object = ignorance_object

    def reason(self, df_triples: pd.DataFrame, df_classes: pd.DataFrame) -> pd.DataFrame:
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
                ignorance = self.ignorance.get(model, self.default_ignorance)
                df_model_subsets = df_subsets[df_subsets['model'] == model]
                if joint_mass_function is None:
                    joint_mass_function = DempsterShafer.MassFunction(DempsterShafer.df_to_subset_dict(df_model_subsets, ignorance, self.ignorance_object))
                else:
                    joint_mass_function = joint_mass_function.join_masses(DempsterShafer.MassFunction(DempsterShafer.df_to_subset_dict(df_model_subsets, ignorance, self.ignorance_object)))

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
            result = pd.concat([result, result_tmp], ignore_index=True)
        result['model'] = np.nan
        result['weight'] = result['weight'].round(3)
        df_triples = pd.concat([df_triples[df_triples['p'] != self.predicate], result], ignore_index=True)
        return df_triples


class AFEDempsterShaferAxiom(Axiom):
    """
    The old version of the use-case-specific Dempster-Shafer axiom
    for AFE data (before the new enhancements)
    """
    def __init__(self, issuer_predicate: str = 'ex:issuer', issuing_for_predicate: str = 'ex:issuing_for',
                 domain_knowledge_predicate: str = 'ex:domain_knowledge', ignorance_object: str = 'ex:uncertain',
                 ignorance: float = 0.2):
        """
        :param issuer_predicate: Issuer predicate
        :param issuing_for_predicate: Issuing for predicate
        :param domain_knowledge_predicate: Domain knowledge predicate
        :param ignorance_object: Object that increases ignorance for the mass function
        :param ignorance: Default ignorance
        """
        super().__init__("preprocessing")
        self.issuer_predicate = issuer_predicate
        self.ignorance = ignorance
        self.ignorance_object = ignorance_object
        self.domain_knowledge_predicate = domain_knowledge_predicate
        self.issuing_for_predicate = issuing_for_predicate
        self.plausibility = dict()
        self.beliefs = dict()

    def reason(self, df_triples: pd.DataFrame, df_classes: pd.DataFrame) -> pd.DataFrame:
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

            self.plausibility = issuer_mass_function.get_plausibility_values()
            self.beliefs = issuer_mass_function.get_beliefs()

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
        result['weight'] = result['weight'].round(3)
        df_triples = pd.concat([df_triples[df_triples['p'] != self.issuer_predicate], result])
        return df_triples


class AFEDempsterShaferAxiom_2(Axiom):
    """
    The new enhanced version of the use-case-specific Dempster-Shafer axiom
    for AFE data (after the new enhancements)
    """
    def __init__(self, target_predicate: str = 'nmo:hasIssuer',
                 knowledge_path_predicate: str = 'nmo:hasPortait',
                 domain_knowledge_predicate: str = 'ex:hasPossibleIssuers',
                 ignorance_object: str = 'ex:uncertain',
                 target_ignorance: float = 0.2,
                 domain_knowledge_ignorance: float = 0.2,
                 group: str = "Undefined",
                 auto_tune_domain_ignorance: bool = False):
        """
        :param target_predicate: Target predicate
        :param knowledge_path_predicate: Predicate that specifies which domain knowledge should be used
        :param domain_knowledge_predicate: Domain knowledge predicate
        :param ignorance_object: Ignorance object that increases the ignorance mass in the appropriate mass function
        :param target_ignorance: Default ignorance mass in the user mass function
        :param domain_knowledge_ignorance: Default ignorance mass in the domain knowledge mass functions
        :param group: Name of the Group to which the instance belongs to
        :param auto_tune_domain_ignorance: Flag to auto-tune the default domain knowledge ignorance mass to prioritize the domain knowledge suggestion
        """
        super().__init__("preprocessing")
        self.target_predicate = target_predicate
        self.knowledge_path_predicate = knowledge_path_predicate
        self.domain_knowledge_predicate = domain_knowledge_predicate
        self.domain_knowledge_ignorance = domain_knowledge_ignorance
        self.target_ignorance = target_ignorance
        self.ignorance_object = ignorance_object
        self.auto_tune_domain_ignorance = auto_tune_domain_ignorance
        self.plausibility = dict()
        self.beliefs = dict()
        self.group = group

    def get_group(self) -> str:
        """
        :return: Returns the group name to which the axiom belong
        """
        return self.group

    def reason(self, df_triples: pd.DataFrame, df_classes: pd.DataFrame) -> pd.DataFrame:
        df_domain_knowledge = df_triples[(df_triples['p'] == self.domain_knowledge_predicate)].copy()
        df_target = df_triples[(df_triples['p'] == self.target_predicate)].copy()
        df_knowledge_path = df_triples[(df_triples['p'] == self.knowledge_path_predicate)].copy()

        result = pd.DataFrame()

        for i, coin in df_target['s'].drop_duplicates().items():
            df_target_subsets = df_target[df_target['s'] == coin]
            df_knowledge_path_subsets = df_knowledge_path[df_knowledge_path['s'] == coin]

            if df_knowledge_path_subsets.shape[0] == 0:
                result = pd.concat([result, df_target_subsets])
                continue

            target_mass_function = DempsterShafer.MassFunction(DempsterShafer.df_to_subset_dict(df_target_subsets, self.target_ignorance, self.ignorance_object))

            knowledge_ignorance_tuning_conditions = self.check_conditions_for_tuning(df_target, df_knowledge_path, df_domain_knowledge)

            df_knowledge_path_subsets_temp = df_knowledge_path_subsets[df_knowledge_path_subsets['o'] != self.ignorance_object]

            for j, knowledge_path in df_knowledge_path_subsets_temp['o'].items():
                df_domain_knowledge_subsets = df_domain_knowledge[df_domain_knowledge['s'] == knowledge_path]

                if not df_domain_knowledge_subsets.empty:
                    if knowledge_ignorance_tuning_conditions:
                        knowledge_ignorance = self._tune_domain_ignorance(df_target, df_domain_knowledge_subsets)
                    else:
                        knowledge_ignorance = self.domain_knowledge_ignorance
                    df_knowledge_path_ignorance = df_knowledge_path_subsets[df_knowledge_path_subsets['o'] == self.ignorance_object]
                    if df_knowledge_path_ignorance.shape[0] == 1:
                        knowledge_ignorance += df_knowledge_path_ignorance['weight'].iloc[0]

                    domain_knowledge_mass_function = DempsterShafer.MassFunction(DempsterShafer.df_to_subset_dict(df_domain_knowledge_subsets, knowledge_ignorance, self.ignorance_object))
                    target_mass_function = target_mass_function.join_masses(domain_knowledge_mass_function)

            result_tmp = {
                's': [],
                'p': [],
                'o': [],
                'weight': []
            }
            mass_values = target_mass_function.get_mass_values()
            for target in mass_values:
                if target == "*":
                    result_tmp['s'].append(coin)
                    result_tmp['p'].append(self.target_predicate)
                    result_tmp['o'].append(self.ignorance_object)
                    result_tmp['weight'].append(mass_values[target])
                    continue
                result_tmp['s'].append(coin)
                result_tmp['p'].append(self.target_predicate)
                result_tmp['o'].append(target)
                result_tmp['weight'].append(mass_values[target])

            result_tmp = pd.DataFrame(result_tmp)
            result = pd.concat([result, result_tmp])

        result['weight'] = result['weight'].round(3)
        df_triples = pd.concat([df_triples[df_triples['p'] != self.target_predicate], result])

        return df_triples

    def check_conditions_for_tuning(self, df_target_subsets: pd.DataFrame,
                                    df_knowledge_path_subsets: pd.DataFrame,
                                    df_domain_knowledge: pd.DataFrame):
        """
        if the user set the auto_tune_domain_ignorance to true,
        check if the case really need the tuning of the domain knowledge ignorance
        """
        if not self.auto_tune_domain_ignorance:
            return False

        # user issuers (exclude 'uncertain')
        user_selected_targets = set(df_target_subsets.loc[df_target_subsets[
                                                     'o'] != self.ignorance_object, 'o'])

        user_selected_domain_path = set(
            df_knowledge_path_subsets.loc[df_knowledge_path_subsets[
                                              'o'] != self.ignorance_object, 'o'])
        expert_suggestions = set(df_domain_knowledge.loc[
                                     df_domain_knowledge['s'].isin(
                                         user_selected_domain_path), 'o'])

        if not user_selected_targets or not expert_suggestions or not user_selected_targets.isdisjoint(
                expert_suggestions):
            return False

        # check certainty states
        def certainty_state(df_subsets: pd.DataFrame) -> str:
            if df_subsets.empty:
                return 'empty'
            return 'uncertain' if (df_subsets['o'] == self.ignorance_object).any() else 'certain'

        # we require both to be 'certain' or both 'uncertain' to scale ignorance
        if not (certainty_state(df_target_subsets) == certainty_state(df_knowledge_path_subsets)
                and certainty_state(df_target_subsets) in ('certain', 'uncertain')
        ):
            return False

        return True

    def _tune_domain_ignorance(
            self,
            df_target: pd.DataFrame,
            df_domain_knowledge: pd.DataFrame,
    ) -> float:
        """
        Apply the rule D < (Ux)/(Ux + k(1-U)) .
        """
        # user selected target values (exclude 'uncertain')
        user_targets = set(df_target.loc[df_target['o'] != self.ignorance_object, 'o'])
        x = len(user_targets)

        k = len(df_domain_knowledge)

        # apply the rule
        U = self.target_ignorance
        if x <= 0 or k <= 0:
            return self.domain_knowledge_ignorance

        D_max = (U * x) / (U * x + k * (1 - U))
        D_max_floor = math.floor(D_max * 100) / 100.0

        domain_knowledge_ignorance_tuned = min(self.domain_knowledge_ignorance, D_max_floor)

        return max(0.0, domain_knowledge_ignorance_tuned)


class NormalizationAxiom(Axiom):
    """
    Normalizes the weights by model to sum up to one
    """
    def __init__(self, predicate: str, group: str = "Undefined"):
        """
        :param predicate: Predicate to normalize
        """
        super().__init__("postprocessing")
        self.predicate = predicate

    def reason(self, df_triples: pd.DataFrame, df_classes: pd.DataFrame) -> pd.DataFrame:
        df_agg = df_triples[df_triples['p'] == self.predicate].copy()
        df_agg = df_agg.groupby(['s', 'p', 'o', 'model'])[['weight']]
        df_agg = df_agg.sum()

        df_agg = df_agg.reset_index()
        df_agg = df_agg.rename(columns={'weight': 'sum'})

        df_triples = pd.merge(df_triples, df_agg, on=['s', 'p', 'o', 'model'])
        df_triples['weight'] = df_triples['weight'] / df_triples['sum']
        df_triples = df_triples.drop(columns=['sum'])

        return df_triples


class InverseAxiom(Axiom):
    """
    Defines the inverse of predicates. The inverse of an antecedent has the same weight as the antecedent.
    """
    def __init__(self, antecedent: str, inverse: str, group: str = "Undefined"):
        """
        :param antecedent: Antecedent predicate
        :param inverse: Inverse predicate
        """
        super().__init__('rule_based_reasoning')
        self.antecedent = antecedent
        self.inverse = inverse

    def reason(self, df_triples: pd.DataFrame, df_classes: pd.DataFrame) -> pd.DataFrame:
        df_tmp = df_triples[df_triples['p'] == self.antecedent].copy()
        df_tmp = df_tmp.rename(columns={'s': 's_t'})
        df_tmp = df_tmp.rename(columns={'s_t': 'o', 'o': 's'})
        df_tmp['p'] = self.inverse
        df_triples = pd.concat([df_triples, df_tmp]).sort_values(by='weight', ascending=True)
        df_triples = df_triples.drop_duplicates(subset=['s', 'p', 'o'], keep='last')

        return df_triples


class ChainRuleAxiom(Axiom):
    """
    Implements a modified version of the chain rule axiom from the Academic Meta Tool.
    """
    def __init__(self, antecedent1: str, antecedent2: str, consequent: str, reasoning_logic: Literal['product', 'goedel', 'lukasiewicz'], sum_values: bool = False, class_1: str = None, class_2: str = None,
                 class_3: str = None, input_threshold: float = None, output_threshold: float = None):
        """

        :param antecedent1: First antecedent predicate: A antecedent1 B
        :param antecedent2: Second antecedent predicate: B antecedent2 C
        :param consequent: Consequent predicate: A consequent C
        :param reasoning_logic: Reasoning logic to use
        :param sum_values: Sum values of viable paths
        :param class_1: Optional class constraint for node A
        :param class_2: Optional class constraint for node B
        :param class_3: Optional class constraint for node C
        :param input_threshold: Optional input threshold for weights
        :param output_threshold: Optional output threshold for weights
        """
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
                                         (df_triples['p'] == self.antecedent2)]

        # Enforce input threshold
        if self.input_threshold:
            df_selected_triples = df_selected_triples[df_selected_triples['weight'] >= self.input_threshold]

        df_triples_left = df_selected_triples[df_selected_triples['p'] == self.antecedent1]
        df_triples_right = df_selected_triples[df_selected_triples['p'] == self.antecedent2]

        # Enforce classes by inner-merging with df_classes
        if self.class1:
            df_triples_left = pd.merge(df_triples_left, df_classes[df_classes['class'] == self.class1],
                                       left_on='s', right_on='node')
            df_triples_left = df_triples_left.drop(columns=['class'])
        if self.class2:
            df_triples_left = pd.merge(df_triples_left, df_classes[df_classes['class'] == self.class2],
                                       left_on='o', right_on='node')
            df_triples_left = df_triples_left.drop(columns=['class'])
            df_triples_right = pd.merge(df_triples_right, df_classes[df_classes['class'] == self.class2],
                                        left_on='s', right_on='node')
            df_triples_right = df_triples_right.drop(columns=['class'])
        if self.class3:
            df_triples_right = pd.merge(df_triples_right, df_classes[df_classes['class'] == self.class3],
                                        left_on='o', right_on='node')
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
            df_result['weight'] = df_result['weight_x'] + df_result['weight_y'] - 1.0
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
            df_result = df_result[df_result['weight'] >= self.output_threshold]

        df_result['weight'] = df_result['weight'].round(3)
        df_triples = pd.concat([df_triples, df_result])

        # Only keep the triple with the highest weight
        df_triples = df_triples.sort_values(by='weight', ascending=True)
        df_triples = df_triples.drop_duplicates(subset=['s', 'p', 'o'], keep='last')

        return df_triples


class DisjointAxiom(Axiom):
    """
    Axiom to add a constraint that disallows two predicate to have the same subject and object.
    """
    def __init__(self, predicate1: str, predicate2: str, throw_exception: bool = True, keep_predicate1: bool = True):
        """
        :param predicate1: First predicate
        :param predicate2: Second predicate
        :param throw_exception: Throw exception or just remove one of the triples?
        :param keep_predicate1: Keep predicate1 or predicate2?
        """
        super().__init__("rule_based_reasoning")
        self.predicate1 = predicate1
        self.predicate2 = predicate2
        self.throw_exception = throw_exception
        self.keep_predicate1 = keep_predicate1

    def reason(self, df_triples: pd.DataFrame, df_classes: pd.DataFrame) -> pd.DataFrame:
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
    """
    Constraint axiom to disallow self-referencing for a given predicate
    """
    def __init__(self, predicate: str, throw_exception: bool = True):
        """

        :param predicate: Predicate to disallow self-referencing for
        :param throw_exception: Throw exception or just remove the triples?
        """
        super().__init__("rule_based_reasoning")
        self.predicate = predicate
        self.throw_exception = throw_exception

    def reason(self, df_triples: pd.DataFrame, df_classes: pd.DataFrame) -> pd.DataFrame:
        df_tmp = df_triples[df_triples['p'] == self.predicate]
        if (df_tmp['s'] == df_tmp['o']).sum() != 0 and self.throw_exception:
            raise ConstraintException(f"Constraint violation for predicate {self.predicate}")
        else:
            df_triples = df_triples[(df_triples['p'] != self.predicate) | (df_triples['s'] != df_triples['o'])]

        return df_triples
