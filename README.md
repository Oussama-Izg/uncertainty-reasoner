# Uncertainty Reasoner

## Setting up the Reasoner
1. Install required packages by importing the provided Anaconda environment in ```conda_env.yaml```.
2. Install a triple store, that is accessible via HTTP, like [Apache Jena Fuseki](https://jena.apache.org/documentation/fuseki2/index.html).

## Getting started
1. Define the SPARQL endpoints and uncertainty modeling method using either the ```ReificationSparqlConnector``` or the ```SparqlStarConnector```.
   ```python
   import SparqlConnector
   import Reasoner
   
   # Query endpoint
   QUERY_ENDPOINT = "http://localhost:3030/test/query"
   # Update endpoint
   UPDATE_ENDPOINT = "http://localhost:3030/test/update"
   # Graph Store Protocol endpoint
   GSP_ENDPOINT = "http://localhost:3030/test/data"
   
   conn = SparqlConnector.SparqlStarConnector(QUERY_ENDPOINT, UPDATE_ENDPOINT, GSP_ENDPOINT)
   ```
2. Optionally: Upload data to your endpoint. The data must be a dataframe in the internal format.
   ```python
   conn.upload_df(df)
   ```
3. Define the axioms used during reasoning:
   ```python
   axioms = [
        Reasoner.CertaintyAssignmentAxiom("ex:issuer"),
        Reasoner.CertaintyAssignmentAxiom("ex:issuing_for"),
        Reasoner.CertaintyAssignmentAxiom("ex:domain_knowledge"),
        Reasoner.AFEDempsterShaferAxiom('ex:issuer', 'ex:issuing_for', 'ex:domain_knowledge')
   ]
   ```
4. Create the reasoner and start the reasoning:
   ```
   reasoner = Reasoner.Reasoner(axioms)
   reasoner.load_data_from_endpoint(conn)
   reasoner.reason()
   ```
5. Get the results:
   ```python
   # As dataframe
   df_result = reasoner.get_triples_as_df()
   # Save result as Turtle file
   reasoner.save_data_to_file('result.ttl', conn)
   # Upload result to endpoint
   reasoner.upload_data_to_endpoint(conn2)
   ```