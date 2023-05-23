import pandas as pd

#df = pd.read_csv('test_mid_summary_psyche-kot5.tsv', sep='\t')
df = pd.read_csv('test_mid_summary_eenz-t5.tsv', sep='\t')

df['text'] = df['summary']
df = df.drop('summary', axis=1)
#df = df.drop('code', axis=1)
#df = df.drop('bleu_score', axis=1)

df['CODE'] = df['KSIC'].apply(lambda x: f"{x:05}")
df['scode'] = df['CODE'].str[:-2]

df['mcode'] = df['mcode'].apply(lambda x: f"{x:02}")

#df.to_csv('test_summary_psyche-kot5.tsv', sep='\t', index=False)
df.to_csv('test_summary_eenz-t5.tsv', sep='\t', index=False)

