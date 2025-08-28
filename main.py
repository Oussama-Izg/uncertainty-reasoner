import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame

import SparqlConnector
import Reasoner

# Query endpoint
QUERY_ENDPOINT = "http://localhost:3030/input/query"
# Update endpoint
UPDATE_ENDPOINT = "http://localhost:3030/input/update"
# Graph Store Protocol endpoint
GSP_ENDPOINT = "http://localhost:3030/input/data"

conn = SparqlConnector.ReificationSparqlConnector(QUERY_ENDPOINT, UPDATE_ENDPOINT, GSP_ENDPOINT)

df = pd.read_csv("usecases/data/afe_test_data.csv")
df_2 = pd.read_csv("usecases/data/afe_input.csv")
df_3 = pd.read_csv("usecase_2_4/issuingFor_alternative_uncertain_issuers__alternative_uncertain/usecase_34.csv")

# upload the data from the triple store
conn.upload_df(df_3)

# define the axiom's pipeline for the Reasoner
axioms = [
        Reasoner.CertaintyAssignmentAxiom("nmo:hasIssuer"),
        Reasoner.CertaintyAssignmentAxiom("nmo:hasPortrait"),
        Reasoner.CertaintyAssignmentAxiom("ex:hasPossibleIssuers"),
        Reasoner.CertaintyAssignmentAxiom("ex:inPossibleIssuersOf"),
        Reasoner.AFEDempsterShaferAxiom_2("nmo:hasPortrait",
                                          "nmo:hasIssuer",
                                          "ex:inPossibleIssuersOf",
                                          group="issuingForAndIssuer"),
        Reasoner.AFEDempsterShaferAxiom_2('nmo:hasIssuer',
                                          'nmo:hasPortrait',
                                        'ex:hasPossibleIssuers',
                                          group="issuingForAndIssuer")
    ]

# Instantiate the Reasoner
reasoner = Reasoner.Reasoner(axioms, reasoner_name="UR")
reasoner.load_data_from_endpoint(conn)

# Start the reasoning process
reasoner.reason()

df_dempster_result = reasoner.get_triples_as_df()


df_dempster_result.to_csv("result_usecase_34.csv", index=False)



# Upload result to the second endpoint
#reasoner.upload_data_to_endpoint(conn_2)