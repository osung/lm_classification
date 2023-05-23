import pandas as pd

df1 = pd.read_csv('summary_test_0.tsv', sep='\t')
df2 = pd.read_csv('summary_test_1.tsv', sep='\t')

df = pd.concat([df1, df2])

print(df)

df.to_csv('test_mid_summary.tsv', index=False, sep='\t')

