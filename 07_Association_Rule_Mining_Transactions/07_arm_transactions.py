

import os
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(SCRIPT_DIR, "transactions.csv")

if os.path.exists(csv_path):
    transaction_df = pd.read_csv(csv_path)
    transactions = transaction_df["Items"].str.split(",").apply(
        lambda items: [item.strip() for item in items]
    ).tolist()
else:
    transactions = [
        ['Milk', 'Bread', 'Butter'],
        ['Bread', 'Butter'],
        ['Milk', 'Bread'],
        ['Milk', 'Butter'],
        ['Bread', 'Butter'],
        ['Milk', 'Bread', 'Butter'],
        ['Milk', 'Bread'],
        ['Butter'],
        ['Milk', 'Butter'],
        ['Bread', 'Butter']
    ]

te = TransactionEncoder()

te_data = te.fit(transactions).transform(transactions)

df = pd.DataFrame(te_data, columns=te.columns_)

print("Transactional Dataset:")
print(df.head())

frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)

print("\nFrequent Itemsets:")
print(frequent_itemsets)

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.6
)

print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support',
             'confidence', 'lift']])
