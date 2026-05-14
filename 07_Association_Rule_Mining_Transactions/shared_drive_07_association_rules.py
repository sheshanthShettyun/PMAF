#pip install pandas mlxtend
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

transactions = [
    ['milk','bread','butter'],
    ['bread','eggs'],
    ['milk','bread'],
    ['milk','eggs'],
    ['bread','butter'],
    ['milk','bread','eggs'],
    ['milk','butter'],
    ['bread','eggs'],
    ['milk','bread','butter','eggs'],
    ['butter','eggs']
]

# Convert to one-hot encoding
te = TransactionEncoder()
te_data = te.fit(transactions).transform(transactions)

df = pd.DataFrame(te_data, columns=te.columns_)

# Apply Apriori
frequent_items = apriori(df, min_support=0.2, use_colnames=True)

# Generate rules
rules = association_rules(frequent_items, metric="confidence", min_threshold=0.5)

# Output
print("Frequent Itemsets:\n", frequent_items)
print("\nAssociation Rules:\n", rules)