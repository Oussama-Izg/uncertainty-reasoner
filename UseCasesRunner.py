import Reasoner
import SparqlConnector

import pandas as pd
import os


def run_use_cases(root_folder_name):
    # Query endpoint
    QUERY_ENDPOINT = "http://localhost:3030/test/query"
    # Update endpoint
    UPDATE_ENDPOINT = "http://localhost:3030/test/update"
    # Graph Store Protocol endpoint
    GSP_ENDPOINT = "http://localhost:3030/test/data"

    prefixes = {"nmo": "http://nomisma.org/ontology#"}

    conn = SparqlConnector.ReificationSparqlConnector(QUERY_ENDPOINT,
                                                      UPDATE_ENDPOINT,
                                                      GSP_ENDPOINT,
                                                      prefixes=prefixes)

    axioms = [
        Reasoner.CertaintyAssignmentAxiom("nmo:hasIssuer"),
        Reasoner.CertaintyAssignmentAxiom("nmo:hasPortrait"),
        Reasoner.CertaintyAssignmentAxiom("ex:hasPossibleIssuers"),
        Reasoner.AFEDempsterShaferAxiom_2('nmo:hasIssuer', 'nmo:hasPortrait',
                                        'ex:hasPossibleIssuers')
    ]

    axioms1 = [
        Reasoner.CertaintyAssignmentAxiom("nmo:hasIssuer"),
        Reasoner.CertaintyAssignmentAxiom("nmo:hasPortrait"),
        Reasoner.CertaintyAssignmentAxiom("ex:hasPossibleIssuers"),
        Reasoner.CertaintyAssignmentAxiom("ex:inPossibleIssuersOf"),
        Reasoner.AFEDempsterShaferAxiom_2(target_predicate="nmo:hasIssuer",
                                          knowledge_path_predicate="nmo:hasPortrait",
                                          domain_knowledge_predicate="ex:hasPossibleIssuers",
                                          group="1",
                                          target_ignorance=0.2,
                                          domain_knowledge_ignorance=0.2),
        Reasoner.AFEDempsterShaferAxiom_2(target_predicate="nmo:hasPortrait",
                                          knowledge_path_predicate="nmo:hasIssuer",
                                          domain_knowledge_predicate="ex:inPossibleIssuersOf",
                                          group="1",
                                          target_ignorance=0.2,
                                          domain_knowledge_ignorance=0.2)

    ]

    axioms2 = [
        Reasoner.CertaintyAssignmentAxiom("nmo:hasIssuer"),
        Reasoner.CertaintyAssignmentAxiom("nmo:hasPortrait"),
        Reasoner.CertaintyAssignmentAxiom("ex:hasPossibleIssuers"),
        Reasoner.CertaintyAssignmentAxiom("ex:inPossibleIssuersOf"),
        Reasoner.AFEDempsterShaferAxiom_2(target_predicate="nmo:hasIssuer",
                                          knowledge_path_predicate="nmo:hasPortrait",
                                          domain_knowledge_predicate="ex:hasPossibleIssuers",
                                          target_ignorance=0.2,
                                          domain_knowledge_ignorance=0.2),
        Reasoner.AFEDempsterShaferAxiom_2(target_predicate="nmo:hasPortrait",
                                          knowledge_path_predicate="nmo:hasIssuer",
                                          domain_knowledge_predicate="ex:inPossibleIssuersOf",
                                          target_ignorance=0.2,
                                          domain_knowledge_ignorance=0.2)
    ]

    axioms3 = [
        Reasoner.CertaintyAssignmentAxiom("nmo:hasIssuer"),
        Reasoner.CertaintyAssignmentAxiom("nmo:hasPortrait"),
        Reasoner.CertaintyAssignmentAxiom("ex:hasPossibleIssuers"),
        Reasoner.CertaintyAssignmentAxiom("ex:inPossibleIssuersOf"),
        Reasoner.AFEDempsterShaferAxiom_2(target_predicate="nmo:hasPortrait",
                                          knowledge_path_predicate="nmo:hasIssuer",
                                          domain_knowledge_predicate="ex:inPossibleIssuersOf",
                                          target_ignorance=0.2,
                                          domain_knowledge_ignorance=0.2),
        Reasoner.AFEDempsterShaferAxiom_2(target_predicate="nmo:hasIssuer",
                                          knowledge_path_predicate="nmo:hasPortrait",
                                          domain_knowledge_predicate="ex:hasPossibleIssuers",
                                          target_ignorance=0.2,
                                          domain_knowledge_ignorance=0.2)

    ]

    # Path to the root directory containing all test case folders
    root_dir = root_folder_name

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)

        if os.path.isdir(folder_path):
            # Loop through all CSV files in the folder
            for filename in os.listdir(folder_path):
                if filename.endswith(".csv") and "Depicted_Person_uncertain__Issuer_alternative_certain__negative.csv" in filename:
                    # if filename.endswith(".csv") and "result" not in filename:
                    csv_path = os.path.join(folder_path, filename)
                    df = pd.read_csv(csv_path)
                    print(
                        f"Uploading {csv_path} to Apache Jena Fuseki Triple store")
                    conn.upload_df(df)

                    # upload the data from the triple store and reason upon it
                    reasoner = Reasoner.Reasoner(axioms3)
                    reasoner.load_data_from_endpoint(conn)
                    reasoner.reason()

                    # Get the result as a dataframe and save it in csv file
                    results_df = reasoner.get_triples_as_df()
                    results_file_name = f"result_{filename}"
                    result_path = os.path.join(folder_path, results_file_name)
                    results_df.to_csv(result_path, index=False)
                    reasoner.save_data_to_file("result.ttl", conn)

                    print(
                        f"Reasoning Results from Reasoner1 are stored in {result_path}")

                    # Delete the data of the previous use case from the triple store
                    conn.delete_query(delete_all=True)


def delete_results(root_folder_name):
    # Walk through all directories and files
    for folder_path, _, files in os.walk(root_folder_name):
        for filename in files:
            if "result_" in filename:
                file_path = os.path.join(folder_path, filename)
                os.remove(file_path)
                print(f"Deleted: {file_path}")


if __name__ == "__main__":
    delete_results("thesis_cases")
    run_use_cases("thesis_cases")