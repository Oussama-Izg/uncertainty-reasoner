import requests
import io
import pandas as pd
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class SparqlBaseConnector(ABC):
    def __init__(self, query_endpoint, update_endpoint, gsp_endpoint, prefixes=None,
                 certainty_predicate="ex:certaintyValue",
                 model_predicate="ex:accordingTo", class_predicate="rdf:type"):
        if prefixes is None:
            prefixes = {}

        if 'rdf' not in prefixes:
            prefixes['rdf'] = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
        if 'xsd' not in prefixes:
            prefixes['xsd'] = 'http://www.w3.org/2001/XMLSchema#'
        if 'ex' not in prefixes:
            prefixes['ex'] = 'http://example.org/'

        self.query_endpoint = query_endpoint
        self.update_endpoint = update_endpoint
        self.gsp_endpoint = gsp_endpoint
        self.prefixes = prefixes
        self.certainty_predicate = certainty_predicate
        self.model_predicate = model_predicate
        self.class_predicate = class_predicate

    def read_query(self, query):
        headers = {
            'Accept': 'text/csv; charset=utf-8'
        }
        data = {
            'query': query
        }
        response = requests.post(self.query_endpoint, data=data, headers=headers)
        if not response:
            raise Exception(f"Failed to query data: {response.reason}")
        return pd.read_csv(io.StringIO(response.text))

    def delete_query(self, query=None, delete_all=False):
        if delete_all:
            query = """DELETE {?s ?p ?o}
                       WHERE {?s ?p ?o}"""
        if query is None and not delete_all:
            raise Exception("Specify query or use delete_all parameter to delete all triples in the graph.")
        data = {
            'update': query
        }
        response = requests.post(self.update_endpoint, data=data)
        if not response:
            raise Exception(f"Failed to delete data: {response.reason}")

    def upload_turtle(self, data):
        headers = {
            'Content-Type': 'text/turtle; charset=utf-8'
        }

        response = requests.post(self.gsp_endpoint, data=data, headers=headers)

        if not response:
            raise Exception(f"Failed to upload data to SPARQL endpoint: {response.reason}")

    def upload_df(self, df_triples):
        self.upload_turtle(self.df_to_turtle(df_triples))

    def download_df(self, query=''):
        self._add_classes(self.read_into_df(query))

    def _add_classes(self, df_triples):
        df_classes = df_triples[df_triples['p'] == self.class_predicate].copy()
        df_classes = df_classes[['s', 'o']]
        df_classes = df_classes.rename(columns={'o': 'class'})
        df_triples = pd.merge(df_triples, df_classes, left_on='s', right_on='s', how='left')
        df_triples = df_triples.rename(columns={'class': 's_concept'})
        df_triples = pd.merge(df_triples, df_classes, left_on='o', right_on='s', how='left')
        df_triples = df_triples.rename(columns={'class': 'o_concept'})

        return df_triples

    @abstractmethod
    def df_to_turtle(self, df_triples):
        pass

    @abstractmethod
    def read_into_df(self, query=""):
        pass


class ReificationSparqlConnector(SparqlBaseConnector):
    def __init__(self, query_endpoint, update_endpoint, gsp_endpoint, prefixes=None):
        super().__init__(query_endpoint, update_endpoint, gsp_endpoint, prefixes)

    def read_into_df(self, query=None):
        if not query:
            query = ""
            for prefix in self.prefixes:
                query += f"PREFIX {prefix}: <{self.prefixes.get(prefix)}> \n"
            query += f"""
            SELECT ?s ?p ?o ?certainty ?model WHERE {{
                {{?b rdf:subject ?s .
                ?b rdf:predicate ?p .
                ?b rdf:object ?o .
                ?b {self.certainty_predicate} ?certainty .
                OPTIONAL {{?b {self.model_predicate} ?model}} .}} UNION {{
                    ?s ?p ?o .
                    FILTER NOT EXISTS{{?s {self.certainty_predicate} ?certainty .}}
                }}
            }}"""
        return self._add_classes(self.read_query(query))

    def df_to_turtle(self, df_triples):
        turtle_data = ""
        for prefix in self.prefixes:
            turtle_data += f"@prefix {prefix}: <{self.prefixes.get(prefix)}> . \n"

        mask = (df_triples['certainty'] == 1) & (df_triples['model'].isna())
        df_triples['certain_triple'] = ""
        df_triples.loc[mask, 'certain_triple'] = df_triples['s'] + " " + df_triples['p'] + " " + df_triples['o'] + " . \n"
        turtle_data += df_triples['certain_triple'].str.cat(sep="")
        df_triples = df_triples[~mask]

        df_triples = df_triples.reset_index()
        df_triples['subject_turtle'] = "_:" + df_triples['index'].astype('string') + " rdf:subject " + df_triples['s'] + " . \n"
        df_triples['predicate_turtle'] = "_:" + df_triples['index'].astype('string') + " rdf:predicate " + df_triples['p'] + " . \n"
        df_triples['object_turtle'] = "_:" + df_triples['index'].astype('string') + " rdf:object " + df_triples['o'] + " . \n"
        df_triples['certainty_turtle'] = "_:" + df_triples['index'].astype('string') + f" {self.certainty_predicate} \"" + df_triples['certainty'].astype('string') + "\"^^xsd:decimal . \n"
        df_triples['model_turtle'] = ""
        df_triples.loc[~df_triples['model'].isna(), 'model_turtle'] = "_:" + df_triples['index'].astype('string') + f" {self.model_predicate} \"" + df_triples['model'] + "\" . \n"

        turtle_data += df_triples['subject_turtle'].str.cat(sep="")
        turtle_data += df_triples['predicate_turtle'].str.cat(sep="")
        turtle_data += df_triples['object_turtle'].str.cat(sep="")
        turtle_data += df_triples['certainty_turtle'].str.cat(sep="")

        return turtle_data


class SparqlStarConnector(SparqlBaseConnector):
    def __init__(self, query_endpoint, update_endpoint, gsp_endpoint, prefixes=None):
        super().__init__(query_endpoint, update_endpoint, gsp_endpoint, prefixes)

    def read_into_df(self, query=None):
        if not query:
            query = ""
            for prefix in self.prefixes:
                query += f"PREFIX {prefix}: <{self.prefixes.get(prefix)}> \n"
            query += f"""
            SELECT ?s ?p ?o ?certainty ?model WHERE {{
                {{<< << ?s ?p ?o >> {self.certainty_predicate} ?certainty >> {self.model_predicate} ?model .}} UNION {{
                ?s ?p ?o .
                MINUS {{ 
                    ?s ?p ?o .
                    FILTER(isTRIPLE(?s)) .
                }} }} UNION {{
                    << ?s ?p ?o >> {self.certainty_predicate} ?certainty 
                }}
            }}"""
        return self._add_classes(self.read_query(query))

    def df_to_turtle(self, df_triples):
        turtle_data = ""
        for prefix in self.prefixes:
            turtle_data += f"@prefix {prefix}: <{self.prefixes.get(prefix)}> . \n"
        mask = (df_triples['certainty'] == 1) & (df_triples['model'].isna())
        df_triples['certain_triple'] = ""
        df_triples.loc[mask, 'certain_triple'] = df_triples['s'] + " " + df_triples['p'] + " " + df_triples['o'] + " . \n"
        turtle_data += df_triples['certain_triple'].str.cat(sep="")
        df_triples = df_triples[~mask].copy()

        df_triples['turtle_triple'] = "<< " + df_triples['s'] + " " + df_triples['p'] + " " + df_triples['o'] + f" >> {self.certainty_predicate} \"" + df_triples['certainty'].astype('string') + "\"^^xsd:decimal"
        df_triples.loc[~df_triples['model'].isna(), 'turtle_triple'] = "<< " + df_triples['turtle_triple'] + f" >> {self.model_predicate} \"" + df_triples['model'] + "\" . \n"
        df_triples.loc[df_triples['model'].isna(), 'turtle_triple'] = df_triples['turtle_triple'] + " . \n"
        turtle_data += df_triples['turtle_triple'].str.cat(sep="")

        return turtle_data
