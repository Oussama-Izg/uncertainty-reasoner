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

conn.upload_df(df_3)

axioms = [Reasoner.CertaintyAssignmentAxiom("ex:issuer"),
          Reasoner.CertaintyAssignmentAxiom("ex:issuing_for"),
          Reasoner.CertaintyAssignmentAxiom("ex:domain_knowledge"),
          Reasoner.AFEDempsterShaferAxiom('ex:issuer', 'ex:issuing_for',
                                          'ex:domain_knowledge')]

#reasoner = Reasoner.Reasoner(axioms, reasoner_name="Reasoner1")
#reasoner.load_data_from_endpoint(conn)
#reasoner.reason()

#df_result = reasoner.get_triples_as_df()

dempster_reasoner = Reasoner.Reasoner(axioms, reasoner_name="DempsterReasoner")
dempster_reasoner.load_data_from_endpoint(conn)
print(dempster_reasoner.get_triples_as_df())
dempster_reasoner.reason()

df_dempster_result = dempster_reasoner.get_triples_as_df()
df_dempster_result.to_csv("result_usecase_34.csv", index=False)

#print("########### Before using Dempster Shafer Theory ############")
#print(df_result)

print()

print("########### After using Dempster Shafer Theory ############")
print(df_dempster_result.sort_index(inplace=False))


# Save result as Turtle file
#reasoner.save_data_to_file('result.ttl', conn)


# Query endpoint
QUERY_ENDPOINT_2 = "http://localhost:3030/output/query"
# Update endpoint
UPDATE_ENDPOINT_2 = "http://localhost:3030/output/update"
# Graph Store Protocol endpoint
GSP_ENDPOINT_2 = "http://localhost:3030/output/data"

conn_2 = SparqlConnector.ReificationSparqlConnector(QUERY_ENDPOINT_2, UPDATE_ENDPOINT_2, GSP_ENDPOINT_2)

# Upload result to the second endpoint
#reasoner.upload_data_to_endpoint(conn_2)