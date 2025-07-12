import os
import pandas as pd

# Path to the root directory containing all test case folders
root_dir = "usecase_2_4"

for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)

    if os.path.isdir(folder_path):
        # Loop through all CSV files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                csv_path = os.path.join(folder_path, filename)
                df = pd.read_csv(csv_path)
                """
                # Replace predicate values
                df['p'] = df['p'].replace({
                    'ex:issuing_for': 'nmo:hasPortrait',
                    'ex:issuer': 'nmo:hasIssuer',
                    'ex:domain_knowledge': 'ex:hasPossibleIssuers'
                })

                # Modify object values conditionally
                def transform_object(row):
                    o = row['o']
                    p = row['p']
                    if o == 'ex:uncertain':
                        return o
                    if p == 'nmo:hasPortrait' and o.startswith('ex:issuer'):
                        return o.replace('ex:issuer', 'ex:person_')
                    if p == 'nmo:hasIssuer' and o.startswith('ex:issuer'):
                        return o.replace('ex:issuer', 'ex:issuer_')
                    if p == 'ex:hasPossibleIssuers' and o.startswith('ex:issuer'):
                        return o.replace('ex:issuer', 'ex:issuer_')

                    return o

                df['o'] = df.apply(transform_object, axis=1)

                # Modify subject values conditionally
                def transform_subject(row):
                    s = row['s']
                    p = row['p']
                    if p == 'ex:hasPossibleIssuers' and s.startswith('ex:issuer'):
                        return s.replace('ex:issuer', 'ex:person_')
                    return s

                df['s'] = df.apply(transform_subject, axis=1)
                """
                # Find rows with p == "ex:hasPossibleIssuers"
                mask = df['p'] == 'ex:hasPossibleIssuers'
                new_rows = df[mask].copy()

                # Create the inverse rows
                new_rows['p'] = 'ex:inPossibleIssuersOf'
                new_rows[['s', 'o']] = new_rows[['o', 's']]

                # Append the new rows to the DataFrame
                df = pd.concat([df, new_rows], ignore_index=True)

                # Save the modified DataFrame
                df.to_csv(csv_path, index=False)