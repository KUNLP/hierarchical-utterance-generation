import pandas as pd
from scipy.stats import spearmanr, pearsonr

# MRR 값 입력
import pandas as pd
from scipy.stats import spearmanr

# 모든 열/행 출력 설정
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)  # 폭 제한 없애기


# MRR 값
data = {
    'Evaluator1': [0.4333, 0.5542, 0.6042, 0.7000],
    'Evaluator2': [0.4292, 0.5542, 0.5917, 0.6833],
    'Evaluator3': [0.4250, 0.6250, 0.5208, 0.6875],
    'Evaluator4': [0.4708, 0.6083, 0.5375, 0.7042],
    'GPT-4o API': [0.4167, 0.4875, 0.4542, 0.7542]
}

methods = ['self-consistency', 'self-verification', 'self-reflect', '계층적 발화 생성 접근법']

df = pd.DataFrame(data, index=methods)

# Spearman 상관계수
spearman_matrix, _ = spearmanr(df, axis=0)
spearman_df = pd.DataFrame(spearman_matrix, index=df.columns, columns=df.columns)

# Pearson 상관계수
pearson_df = df.corr(method='pearson')

# 두 결과 합치기
combined_df = spearman_df.round(4).astype(str) + ' / ' + pearson_df.round(4).astype(str)

# 출력
print("Spearman / Pearson 상관계수 합친 행렬:")
print(combined_df)
