import pandas as pd

path = '/home/osung/data/korean/patent/test_ms_manufact.tsv'
#path = '/home/osung/data/korean/patent/train_summary_psyche-kot5_manufact.tsv'

df = pd.read_csv(path, sep='\t')

#df = df.drop('Unnamed: 0', axis=1)
#df = df.drop('CODE', axis=1)
#df = df.drop('code', axis=1)

print(df)

# generate midx

mcodes = {}    # dict to store pairs of KSIC and int code to learn
targets = []  # list

for idx, row in df.iterrows() :
    if not row['mcode'] in mcodes.keys() :
        mcodes[row['mcode']] = len(mcodes)

    targets.append(mcodes[row['mcode']])

df['midx'] = targets


scodes = {}    # dict to store pairs of KSIC and int code to learn
targets = []  # list

for idx, row in df.iterrows() :
    if not row['scode'] in scodes.keys() :
        scodes[row['scode']] = len(scodes)

    targets.append(scodes[row['scode']])

df['sidx'] = targets


scodes_dict = {}
targets = []  # list

for mcode in mcodes.keys() :
    scodes_dict[mcode] = {} 

for idx, row in df.iterrows() :
    scodes = scodes_dict[row['mcode']]
    if not row['scode'] in scodes.keys() :
        scodes[row['scode']] = len(scodes)

    targets.append(scodes[row['scode']])

df['msidx'] = targets

print(df)

df.to_csv('/home/osung/data/korean/patent/test_ms_manufact_msidx.tsv', 
          sep='\t', index=False)


