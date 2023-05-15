import pandas as pd
from rouge import Rouge


# ROUGE 스코어 계산 함수
def calculate_rouge_scores(reference, summary):
    rouge = Rouge()
    scores = rouge.get_scores(summary, reference)

    print(reference)
    print(summary)

    print(scores)

    return scores[0]['rouge-1']['f']  # ROUGE-1 F1 스코어 반환


df = pd.read_csv('train_mid_summary_eenz-t5.tsv', sep='\t')
print(df)

summaries = df['summary']
references = df['text']

#print(summaries)
#print(references)

rouge_scores = [calculate_rouge_scores(ref, summ) for ref, summ in zip(references, summaries)]  # ROUGE 스코어

average_rouge_score = sum(rouge_scores) / len(rouge_scores)

# 결과 출력
print("전체 ROUGE 스코어 평균:", average_rouge_score)

