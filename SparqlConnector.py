import requests
import io
import pandas as pd
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class SparqlBaseConnector(ABC):
    """
    Abstract class for the SPARQL communication.
    """

    def __init__(self, query_endpoint: str, update_endpoint: str, gsp_endpoint: str, prefixes: dict[str, str] = None,
                 weight_predicate: str = "ex:certaintyValue", model_predicate: str = "ex:accordingTo",
                 class_predicate: str = "rdf:type"):
        """
        :param query_endpoint: Query endpoint url
        :param update_endpoint: Update endpoint url
        :param gsp_endpoint: Graph Store Protocol url
        :param prefixes: Dictionary with the prefixes to use {abbreviation: url}
        :param weight_predicate: Predicate for the weight
        :param model_predicate: Predicate for the model
        :param class_predicate: Predicate for the class
        """
        if prefixes is None:
            prefixes = {}

        if 'rdf' not in prefixes:
            prefixes['rdf'] = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
        if 'xsd' not in prefixes:
            prefixes['xsd'] = 'http://www.w3.org/2001/XMLSchema#'
        if 'ex' not in prefixes:
            prefixes['ex'] = 'http://example.org/'

        self._query_endpoint = query_endpoint
        self._update_endpoint = update_endpoint
        self._gsp_endpoint = gsp_endpoint
        self._prefixes = prefixes
        self._weight_predicate = weight_predicate
        self._model_predicate = model_predicate
        self._class_predicate = class_predicate

    def read_query(self, query: str) -> pd.DataFrame:
        """
        Sends a query to the query endpoint and parses the csv result as a DataFrame
        :param query: The query
        :return: Result as a DataFrame
        """
        headers = {
            'Accept': 'text/csv; charset=utf-8'
        }
        data = {
            'query': query
        }
        response = requests.post(self._query_endpoint, data=data, headers=headers)
        if not response:
            raise Exception(f"Failed to query data: {response.reason}")
        return pd.read_csv(io.StringIO(response.text))

    def delete_query(self, query: str = None, delete_all: str = False) -> None:
        """
        Send a delete query to the update endpoint
        :param query: Optional custom query
        :param delete_all: True to delete all data in the endpoint
        :return:
        """
        if delete_all:
            query = """DELETE {?s ?p ?o}
                       WHERE {?s ?p ?o}"""
        if query is None and not delete_all:
            raise Exception("Specify query or use delete_all parameter to delete all triples in the graph.")
        data = {
            'update': query
        }
        response = requests.post(self._update_endpoint, data=data)
        if not response:
            raise Exception(f"Failed to delete data: {response.reason}")

    def upload_turtle(self, data: str) -> None:
        """
        Upload turtle string to Graph Store Protocol endpoint
        :param data: Turtle data
        :return:
        """
        headers = {
            'Content-Type': 'text/turtle; charset=utf-8'
        }

        response = requests.post(self._gsp_endpoint, data=data, headers=headers)

        if not response:
            raise Exception(f"Failed to upload data to SPARQL endpoint: {response.reason}")

    def upload_df(self, df_triples: pd.DataFrame) -> None:
        """
        Translate df to turtle triples and upload it to the gsp endpoint
        :param df_triples: Dataframe in the internal format
        :return:
        """
        self.upload_turtle(self.df_to_turtle(df_triples))

    def download_df(self, query: str = '') -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Query data from SPARQL endpoint and
        :param query: Custom query
        :return: Triples and class dataframe in internal format
        """
        return self.get_classes(self._apply_prefixes(self.read_into_df(query)))

    def get_classes(self, df_triples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract class declaration from triples dataframe
        :param df_triples: Dataframe in internal format
        :return: Triples and class dataframe in internal format
        """
        df_classes = df_triples[df_triples['p'] == self._class_predicate].copy()
        df_triples = df_triples[df_triples['p'] != self._class_predicate].copy()
        df_classes = df_classes[['s', 'o']]
        df_classes = df_classes.rename(columns={
            's': 'node',
            'o': 'class',
        })

        return df_triples, df_classes

    def _apply_prefixes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply prefixes on queried data
        :param df: Dataframe in internal format
        :return: Dataframe with abbreviated resources
        """
        # Reverse prefix mapping
        prefix_mapping = {value: key + ":" for key, value in self._prefixes.items()}
        # Apply prefixes
        df = df.replace(prefix_mapping, regex=True)
        return df

    @abstractmethod
    def df_to_turtle(self, df_triples: pd.DataFrame) -> str:
        """
        Translate dataframe to turtle triple string
        :param df_triples: Dataframe in internal dataframe
        :return: Turtle string
        """
        pass

    @abstractmethod
    def read_into_df(self, query: str = "") -> pd.DataFrame:
        """
        Query endpoint and translate result into dataframe
        :param query: Custom query
        :return: Dataframe in internal format
        """
        pass


class ReificationSparqlConnector(SparqlBaseConnector):
    """
    SparqlConnector that implements a generic refication approach to model weight values
    """
    def __init__(self, query_endpoint, update_endpoint, gsp_endpoint, prefixes=None):
        super().__init__(query_endpoint, update_endpoint, gsp_endpoint, prefixes)

    def read_into_df(self, query: str = None) -> pd.DataFrame:
        """
        Query endpoint and translate result into dataframe
        :param query: Custom query
        :return: Dataframe in internal format
        """
        if not query:
            query = ""
            for prefix in self._prefixes:
                query += f"PREFIX {prefix}: <{self._prefixes.get(prefix)}> \n"
            query += f"""
            SELECT ?s ?p ?o ?weight ?model WHERE {{
                {{?b rdf:subject ?s .
                ?b rdf:predicate ?p .
                ?b rdf:object ?o .
                ?b {self._weight_predicate} ?weight .
                OPTIONAL {{?b {self._model_predicate} ?model}} .}} UNION {{
                    ?s ?p ?o .
                    FILTER NOT EXISTS{{?s {self._weight_predicate} ?weight .}}
                }}
            }}"""
        return self.read_query(query)

    def df_to_turtle(self, df_triples: pd.DataFrame) -> str:
        """
        Translate dataframe to turtle triple string
        :param df_triples: Dataframe in internal dataframe
        :return: Turtle string
        """
        turtle_data = ""
        for prefix in self._prefixes:
            turtle_data += f"@prefix {prefix}: <{self._prefixes.get(prefix)}> . \n"

        # Certain triples
        mask = (df_triples['weight'] == 1) & (df_triples['model'].isna())
        df_triples['certain_triple'] = ""
        df_triples.loc[mask, 'certain_triple'] = df_triples['s'] + " " + df_triples['p'] + " " + df_triples[
            'o'] + " . \n"
        turtle_data += df_triples['certain_triple'].str.cat(sep="")
        df_triples = df_triples[~mask]

        df_triples = df_triples.reset_index()
        df_triples['subject_turtle'] = "_:" + df_triples['index'].astype('string') + " rdf:subject " + df_triples[
            's'] + " . \n"
        df_triples['predicate_turtle'] = "_:" + df_triples['index'].astype('string') + " rdf:predicate " + df_triples[
            'p'] + " . \n"
        df_triples['object_turtle'] = "_:" + df_triples['index'].astype('string') + " rdf:object " + df_triples[
            'o'] + " . \n"
        df_triples['weight_turtle'] = "_:" + df_triples['index'].astype('string') + f" {self._weight_predicate} \"" + \
                                         df_triples['weight'].astype('string') + "\"^^xsd:decimal . \n"
        # Optional model values
        df_triples['model_turtle'] = ""
        df_triples.loc[~df_triples['model'].isna(), 'model_turtle'] = "_:" + df_triples['index'].astype(
            'string') + f" {self._model_predicate} \"" + df_triples['model'] + "\" . \n"

        turtle_data += df_triples['subject_turtle'].str.cat(sep="")
        turtle_data += df_triples['predicate_turtle'].str.cat(sep="")
        turtle_data += df_triples['object_turtle'].str.cat(sep="")
        turtle_data += df_triples['weight_turtle'].str.cat(sep="")

        return turtle_data


class SparqlStarConnector(SparqlBaseConnector):
    """
    SparqlConnector that implements a RDF-star to model weight values
    """
    def __init__(self, query_endpoint, update_endpoint, gsp_endpoint, prefixes=None):
        super().__init__(query_endpoint, update_endpoint, gsp_endpoint, prefixes)

    def read_into_df(self, query: str = None) -> pd.DataFrame:
        """
        Query endpoint and translate result into dataframe
        :param query: Custom query
        :return: Dataframe in internal format
        """
        if not query:
            query = ""
            for prefix in self._prefixes:
                query += f"PREFIX {prefix}: <{self._prefixes.get(prefix)}> \n"
            query += f"""
            SELECT ?s ?p ?o ?weight ?model WHERE {{
                {{<< << ?s ?p ?o >> {self._weight_predicate} ?weight >> {self._model_predicate} ?model .}} UNION {{
                ?s ?p ?o .
                MINUS {{ 
                    ?s ?p ?o .
                    FILTER(isTRIPLE(?s)) .
                }} }} UNION {{
                    << ?s ?p ?o >> {self._weight_predicate} ?weight 
                }}
            }}"""
        return self.read_query(query)

    def df_to_turtle(self, df_triples: pd.DataFrame) -> str:
        """
        Translate dataframe to turtle triple string
        :param df_triples: Dataframe in internal dataframe
        :return: Turtle string
        """
        turtle_data = ""
        for prefix in self._prefixes:
            turtle_data += f"@prefix {prefix}: <{self._prefixes.get(prefix)}> . \n"

        # Certain triples
        mask = (df_triples['weight'] == 1) & (df_triples['model'].isna())
        df_triples['certain_triple'] = ""
        df_triples.loc[mask, 'certain_triple'] = df_triples['s'] + " " + df_triples['p'] + " " + df_triples[
            'o'] + " . \n"
        turtle_data += df_triples['certain_triple'].str.cat(sep="")
        df_triples = df_triples[~mask].copy()

        # Nested RDF-star triples
        df_triples['turtle_triple'] = "<< " + df_triples['s'] + " " + df_triples['p'] + " " + df_triples[
            'o'] + f" >> {self._weight_predicate} \"" + df_triples['weight'].astype('string') + "\"^^xsd:decimal"
        df_triples.loc[~df_triples['model'].isna(), 'turtle_triple'] = "<< " + df_triples[
            'turtle_triple'] + f" >> {self._model_predicate} \"" + df_triples['model'] + "\" . \n"
        df_triples.loc[df_triples['model'].isna(), 'turtle_triple'] = df_triples['turtle_triple'] + " . \n"
        turtle_data += df_triples['turtle_triple'].str.cat(sep="")

        return turtle_data
