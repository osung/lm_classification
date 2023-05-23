import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu


def calculate_bleu_scores(row):
    reference = row['원문']
    summary = row['요약문']
    reference_tokens = reference.split()
    summary_tokens = summary.split()
    bleu_score = sentence_bleu([reference_tokens], summary_tokens)
    return bleu_score

#df = pd.read_csv('train_mid_summary_psyche-kot5.tsv', sep='\t')
df = pd.read_csv('train_mid_summary_eenz-t5.tsv', sep='\t')
print(df)

#df['bleu_score'] = df.apply(calculate_bleu_scores, axis=1)

average_bleu_score = corpus_bleu([[ref.split()] for ref in df['text']], [hyp.split() for hyp in df['summary']])

print("average bleu score:", average_bleu_score)

