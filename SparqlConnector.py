import requests
import io
import pandas as pd
from abc import ABC, abstractmethod


class SparqlBaseConnector(ABC):
    def __init__(self, query_endpoint, update_endpoint, gsp_endpoint, prefixes=None, certainty_predicate="ex:certaintyValue",
                 model_predicate="ex:accordingTo"):
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

    def read_query(self, query):
        headers = {
            'Accept': 'text/csv; charset=utf-8'
        }
        data = {
            'query': query
        }
        response = requests.get(self.query_endpoint, data=data, headers=headers)
        return pd.read_csv(io.StringIO(response.text))

    def delete_query(self, query=None, delete_all=False):
        if delete_all:
            query = """DELETE {?s ?p ?o}
                       WHERE {?s ?p ?o}"""
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

    @abstractmethod
    def upload_df(self, df):
        pass

    @abstractmethod
    def read_into_df(self, query=""):
        pass


class ReificationSparqlConnector(SparqlBaseConnector):
    def __init__(self, query_endpoint, update_endpoint, gsp_endpoint, prefixes=None):
        super().__init__(query_endpoint, update_endpoint, gsp_endpoint, prefixes)

    def read_into_df(self, query=""):
        if not query:
            for prefix in self.prefixes:
                query += f"PREFIX {prefix}: <{self.prefixes.get(prefix)}> \n"
            query += f"""
            SELECT ?s ?p ?o ?certainty ?model WHERE {{
                _:1 rdf:subject ?s .
                _:1 rdf:predicate ?p .
                _:1 rdf:object ?o .
                OPTIONAL {{_:1 {self.certainty_predicate} ?certainty}} .
                OPTIONAL {{_:1 {self.model_predicate} ?model}} .
            }}"""
        self.read_query(query)

    def upload_df(self, df):
        turtle_data = ""
        for prefix in self.prefixes:
            turtle_data += f"@prefix {prefix}: <{self.prefixes.get(prefix)}> . \n"
        counter = 1
        for index, row in df.iterrows():
            turtle_data += f"_:{counter} rdf:subject {row['s']} . \n"
            turtle_data += f"_:{counter} rdf:predicate {row['p']} . \n"
            turtle_data += f"_:{counter} rdf:object {row['o']} . \n"
            turtle_data += f"_:{counter} {self.certainty_predicate} \"{row['certainty']}\"^^xsd:decimal . \n"
            counter += 1
        with open('text.txt', 'w') as f:
            f.write(turtle_data)
        self.upload_turtle(turtle_data)


class SparqlStarConnector(SparqlBaseConnector):
    def __init__(self, query_endpoint, update_endpoint, gsp_endpoint, prefixes=None):
        super().__init__(query_endpoint, update_endpoint, gsp_endpoint, prefixes)

    def read_into_df(self, query=""):
        if not query:
            for prefix in self.prefixes:
                query += f"PREFIX {prefix}: <{self.prefixes.get(prefix)}> \n"
            query += f"""
            SELECT ?s ?p ?o ?certainty ?model WHERE {{
                << ?s ?p ?o >> {self.certainty_predicate} ?certainty .
                OPTIONAL{{<< ?s ?p ?o >> {self.model_predicate} ?model}} .
            }}"""
        self.read_query(query)

    def upload_df(self, df):
        turtle_data = ""
        for prefix in self.prefixes:
            turtle_data += f"@prefix {prefix}: <{self.prefixes.get(prefix)}> . \n"
        for index, row in df.iterrows():
            turtle_data += f"<< {row['s']} {row['p']} {row['o']} >> {self.certainty_predicate} \"{row['certainty']}\"^^xsd:decimal . \n"

        self.upload_turtle(turtle_data)


if __name__ == '__main__':
    conn = ReificationSparqlConnector("http://localhost:3030/test")
    query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT * WHERE {
        ?sub ?pred ?obj .
    } LIMIT 10
    """
    print(conn.read_query(query))
