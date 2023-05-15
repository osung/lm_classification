import pandas as pd

df1 = pd.read_csv('summary_data_0.tsv', sep='\t')
df2 = pd.read_csv('summary_data_1.tsv', sep='\t')
df3 = pd.read_csv('summary_data_2.tsv', sep='\t')
df4 = pd.read_csv('summary_data_3.tsv', sep='\t')

df = pd.concat([df1, df2, df3, df4])
        
print(df)
        
df.to_csv('train_mid_summary_eenz-t5.tsv', index=False, sep='\t')

