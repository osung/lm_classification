import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu


def calculate_bleu_scores(row):
    reference = row['text']
    summary = row['summary']
    reference_tokens = reference.split()
    summary_tokens = summary.split()
    bleu_score = sentence_bleu([reference_tokens], summary_tokens)
    return bleu_score


df = pd.read_csv('train_mid_summary_psyche-kot5.tsv', sep='\t')
#df = pd.read_csv('train_mid_summary_eenz-t5.tsv', sep='\t')
print(df)

df['bleu_score'] = df.apply(calculate_bleu_scores, axis=1)

# save the bleu scores 
df.to_csv('train_mid_summary_bleu_psyche-kot5.tsv', sep='\t', index=False)

# remove short sentences which did not generated summary
# because the summary is exactly same as text, the score is 1.0
df = df[df['bleu_score'] < 1.0]

print("mean bleu score:", df['bleu_score'].mean())


