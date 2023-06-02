import pandas as pd

path = '/home/osung/data/korean/patent/test_ms_manufact.tsv'

df = pd.read_csv(path, sep='\t')

print(df)

# generate midx

mcodes = {}    # dict to store pairs of KSIC and int code to learn
targets = []  # list

for idx, row in df.iterrows() :
    if not row['mcode'] in mcodes.keys() :
        mcodes[row['mcode']] = len(mcodes)

    targets.append(mcodes[row['mcode']])

df['midx'] = targets


scodes_dict = {}
targets = []  # list

for mcode in mcodes.keys() :
    scodes_dict[mcode] = {} 

for idx, row in df.iterrows() :
    scodes = scodes_dict[row['mcode']]
    if not row['scode'] in scodes.keys() :
        #scodes_dict[row['mcode']][row['scode']] = len(scodes)
        scodes[row['scode']] = len(scodes)

    #targets.append(scodes_dict[row['mcode']])
    targets.append(scodes[row['scode']])

df['sidx'] = targets

print(df)

df.to_csv('/home/osung/data/korean/patent/test_ms_manufact_msidx.tsv', 
          sep='\t', index=False)


