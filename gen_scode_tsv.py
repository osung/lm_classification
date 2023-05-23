import pandas as pd

#df = pd.read_csv('test_mid_summary_psyche-kot5.tsv', sep='\t')
df = pd.read_csv('test_mid2.tsv', sep='\t')

df['CODE'] = df['KSIC'].apply(lambda x: f"{x:05}")
df['scode'] = df['CODE'].str[:-2]

df['mcode'] = df['mcode'].apply(lambda x: f"{x:02}")

#df.to_csv('test_summary_psyche-kot5.tsv', sep='\t', index=False)
df.to_csv('test_mid3.tsv', sep='\t', index=False)

