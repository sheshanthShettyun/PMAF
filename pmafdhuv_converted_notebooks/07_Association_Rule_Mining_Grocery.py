# Converted from PMAF_7/Assignment 7.ipynb
# Original notebook code cells exported as a normal Python script.

from mlxtend.frequent_patterns import apriori, association_rules
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

grocery_items = set()
with open("grocery_dataset.txt") as f:
    reader = csv.reader(f, delimiter=",")
    for i, line in enumerate(reader):
        grocery_items.update(line)

output_list = list()
with open("grocery_dataset.txt") as f:
    reader = csv.reader(f, delimiter=",")
    for i, line in enumerate(reader):
        row_val = {item:0 for item in grocery_items}
        row_val.update({item:1 for item in line})
        output_list.append(row_val)

grocery = pd.DataFrame(output_list)
grocery.head()

grocery.columns.tolist()

total_item_count = sum(grocery.sum())
print("Total Item : ", total_item_count)
print()
item_summary = grocery.sum().sort_values(ascending = False).reset_index()
item_summary.rename(columns={item_summary.columns[0]:'item_name',item_summary.columns[1]:'item_count'}, inplace=True)
print("Item Summary:")
print(item_summary.head())

objects = (list(item_summary['item_name'].head(n=20)))
y_pos = np.arange(len(objects))
performance = list(item_summary['item_count'].head(n=20))

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects, rotation='vertical')
plt.ylabel('Item count')
plt.title('Item sales distribution')
plt.show()

frequent_itemsets = apriori(grocery, min_support=0.047, use_colnames=True)

my_rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

my_rules.head(100)
