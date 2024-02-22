import requests
import io
import pandas as pd
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class SparqlBaseConnector(ABC):
    def __init__(self, query_endpoint, update_endpoint, gsp_endpoint, prefixes=None,
                 certainty_predicate="ex:certaintyValue",
                 model_predicate="ex:accordingTo", concept_predicate="rdf:type"):
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
        self.concept_predicate = concept_predicate

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

    def upload_df(self, df):
        self.upload_turtle(self.df_to_turtle(df))

    @abstractmethod
    def df_to_turtle(self, df):
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
        return self.read_query(query)

    def df_to_turtle(self, df):
        turtle_data = ""
        for prefix in self.prefixes:
            turtle_data += f"@prefix {prefix}: <{self.prefixes.get(prefix)}> . \n"

        mask = (df['certainty'] == 1) & (df['model'].isna())
        df['certain_triple'] = ""
        df.loc[mask, 'certain_triple'] = df['s'] + " " + df['p'] + " " + df['o'] + " . \n"
        turtle_data += df['certain_triple'].str.cat(sep="")
        df = df[~mask]

        df = df.reset_index()
        df['subject_turtle'] = "_:" + df['index'].astype('string') + " rdf:subject " + df['s'] + " . \n"
        df['predicate_turtle'] = "_:" + df['index'].astype('string') + " rdf:predicate " + df['p'] + " . \n"
        df['object_turtle'] = "_:" + df['index'].astype('string') + " rdf:object " + df['o'] + " . \n"
        df['certainty_turtle'] = "_:" + df['index'].astype('string') + f" {self.certainty_predicate} \"" + df['certainty'].astype('string') + "\"^^xsd:decimal . \n"
        df['model_turtle'] = ""
        df.loc[~df['model'].isna(), 'model_turtle'] = "_:" + df['index'].astype('string') + f" {self.model_predicate} \"" + df['model'] + "\" . \n"

        turtle_data += df['subject_turtle'].str.cat(sep="")
        turtle_data += df['predicate_turtle'].str.cat(sep="")
        turtle_data += df['object_turtle'].str.cat(sep="")
        turtle_data += df['certainty_turtle'].str.cat(sep="")

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
        return self.read_query(query)

    def df_to_turtle(self, df):
        turtle_data = ""
        for prefix in self.prefixes:
            turtle_data += f"@prefix {prefix}: <{self.prefixes.get(prefix)}> . \n"
        mask = (df['certainty'] == 1) & (df['model'].isna())
        df['certain_triple'] = ""
        df.loc[mask, 'certain_triple'] = df['s'] + " " + df['p'] + " " + df['o'] + " . \n"
        turtle_data += df['certain_triple'].str.cat(sep="")
        df = df[~mask].copy()

        df['turtle_triple'] = "<< " + df['s'] + " " + df['p'] + " " + df['o'] + f" >> {self.certainty_predicate} \"" + df['certainty'].astype('string') + "\"^^xsd:decimal"
        df.loc[~df['model'].isna(), 'turtle_triple'] = "<< " + df['turtle_triple'] + f" >> {self.model_predicate} \"" + df['model'] + "\" . \n"
        df.loc[df['model'].isna(), 'turtle_triple'] = df['turtle_triple'] + " . \n"
        turtle_data += df['turtle_triple'].str.cat(sep="")

        return turtle_data
