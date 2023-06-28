import sys
import pandas as pd

if len(sys.argv) < 2 :
    print("Usage: python", sys.argv[0], "tsv file")
    quit()

df = pd.read_csv(sys.argv[1], sep='\t')

grouped = df.groupby('mcode')

for name, group in grouped :
    mid = "%02d" % name 
    group.to_csv(mid + '_' + sys.argv[1], index=False, sep='\t')




