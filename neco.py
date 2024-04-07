import pandas as pd 

df: pd.DataFrame = pd.read_pickle('data.pkl')

# Filter out only columns defined as important from .tsv file
important_features = [ col.split(' ')[0] 
    for col 
    in pd.read_csv('important.tsv', sep='\t').columns
]
df = df[['biome', *important_features[1:]]]

# Replace all NA values with 0.0
df.replace(pd.NA, 0.0, inplace=False)

# Add a new derived column sum_all - sum of all numeric columns (feature values)
df['sum_all'] = df.sum(axis=1, only_numeric=True)

# Filter out rows with sum_all <= 0.0 - no feature was found in the sample
df = df[df['sum_all'] > 0.0]

