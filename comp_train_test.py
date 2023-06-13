import pandas as pd

train_path = '/home/osung/data/korean/patent/train_summary_manufact_msidx.tsv'
test_path = '/home/osung/data/korean/patent/test_ms_manufact_msidx.tsv'

train_df = pd.read_csv(train_path, sep='\t')
test_df = pd.read_csv(test_path, sep='\t')

train_mcodes = {}    # dict to store pairs of KSIC and int code to learn

for idx, row in train_df.iterrows() :
    if not row['mcode'] in train_mcodes.keys() :
        train_mcodes[row['mcode']] = len(train_mcodes)

test_mcodes = {}    # dict to store pairs of KSIC and int code to learn

for idx, row in test_df.iterrows() :
    if not row['mcode'] in test_mcodes.keys() :
        test_mcodes[row['mcode']] = len(test_mcodes)

comp_mcodes = train_mcodes.keys() == test_mcodes.keys()

print(comp_mcodes)

train_scodes_dict = {}

for mcode in train_mcodes.keys() :
    train_scodes_dict[mcode] = {} 

for idx, row in train_df.iterrows() :
    scodes = train_scodes_dict[row['mcode']]
    if not row['scode'] in scodes.keys() :
        scodes[row['scode']] = len(scodes)

test_scodes_dict = {}

for mcode in test_mcodes.keys() :
    test_scodes_dict[mcode] = {} 

for idx, row in test_df.iterrows() :
    scodes = test_scodes_dict[row['mcode']]
    if not row['scode'] in scodes.keys() :
        scodes[row['scode']] = len(scodes)

#print(train_scodes_dict)

#print(test_scodes_dict)

for key in train_scodes_dict.keys() :
    comp_scodes = train_scodes_dict[key].keys() == test_scodes_dict[key].keys()
    print(key, comp_scodes)

for key in test_scodes_dict.keys() :
    comp_scodes = train_scodes_dict[key].keys() == test_scodes_dict[key].keys()
    print(key, comp_scodes)


