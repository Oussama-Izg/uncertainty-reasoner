import Reasoner
import SparqlConnector

import pandas as pd
import os


def run_use_cases():
    # Query endpoint
    QUERY_ENDPOINT = "http://localhost:3030/test/query"
    # Update endpoint
    UPDATE_ENDPOINT = "http://localhost:3030/test/update"
    # Graph Store Protocol endpoint
    GSP_ENDPOINT = "http://localhost:3030/test/data"

    conn = SparqlConnector.ReificationSparqlConnector(QUERY_ENDPOINT,
                                                      UPDATE_ENDPOINT,
                                                      GSP_ENDPOINT)

    axioms = [
        Reasoner.CertaintyAssignmentAxiom("ex:issuer"),
        Reasoner.CertaintyAssignmentAxiom("ex:issuing_for"),
        Reasoner.CertaintyAssignmentAxiom("ex:domain_knowledge",
                                          uncertainty_value=0.0),
        Reasoner.AFEDempsterShaferAxiom("ex:issuer", "ex:issuing_for",
                                        "ex:domain_knowledge")
    ]

    # Path to the root directory containing all test case folders
    root_dir = "usecase_2_4"

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)

        if os.path.isdir(folder_path):
            # Loop through all CSV files in the folder
            for filename in os.listdir(folder_path):
                if filename.endswith(".csv"):
                    # if filename.endswith(".csv") and "result" not in filename:
                    csv_path = os.path.join(folder_path, filename)
                    df = pd.read_csv(csv_path)
                    print(
                        f"Uploading {csv_path} to Apache Jena Fuseki Triple store")
                    conn.upload_df(df)

                    # upload the data from the triple store and reason upon it
                    reasoner = Reasoner.Reasoner(axioms)
                    reasoner.load_data_from_endpoint(conn)
                    reasoner.reason()

                    # Get the result as a dataframe and save it in csv file
                    results_df = reasoner.get_triples_as_df()
                    results_file_name = f"result_{filename}"
                    result_path = os.path.join(folder_path, results_file_name)
                    results_df.to_csv(result_path, index=False)

                    reasoner.save_data_to_file(
                        file_name="result_usecase_1.ttl", conn=conn)

                    print(
                        f"Reasoning Results from Reasoner1 are stored in {result_path}")

                    # Delete the data of the previous use case from the triple store
                    conn.delete_query(delete_all=True)


def delete_results():
    # Walk through all directories and files
    for folder_path, _, files in os.walk("usecase_2_4"):
        for filename in files:
            if "result_" in filename:
                file_path = os.path.join(folder_path, filename)
                os.remove(file_path)
                print(f"Deleted: {file_path}")


if __name__ == "__main__":
    run_use_cases()
    #delete_results()