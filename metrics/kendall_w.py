import numpy as np

# Evaluator1~4 순위 데이터
samples_text = [
    {'Evaluator1': ['D','C','B','A'], 'Evaluator2': ['D','C','A','B'], 'Evaluator3': ['D','C','B','A'], 'Evaluator4': ['D','C','B','A']},
    {'Evaluator1': ['D','C','A','B'], 'Evaluator2': ['D','B','A','C'], 'Evaluator3': ['D','B','C','A'], 'Evaluator4': ['D','B','C','A']},
    {'Evaluator1': ['D','B','C','A'], 'Evaluator2': ['B','D','C','A'], 'Evaluator3': ['D','B','C','A'], 'Evaluator4': ['D','B','C','A']},
    {'Evaluator1': ['D','B','C','A'], 'Evaluator2': ['C','D','B','A'], 'Evaluator3': ['D','C','B','A'], 'Evaluator4': ['C','D','B','A']},
    {'Evaluator1': ['D','C','B','A'], 'Evaluator2': ['C','D','B','A'], 'Evaluator3': ['D','B','C','A'], 'Evaluator4': ['D','B','C','A']},
    {'Evaluator1': ['C','D','B','A'], 'Evaluator2': ['D','C','B','A'], 'Evaluator3': ['C','D','B','A'], 'Evaluator4': ['C','D','B','A']},
    {'Evaluator1': ['B','A','C','D'], 'Evaluator2': ['A','C','B','D'], 'Evaluator3': ['B','A','C','D'], 'Evaluator4': ['A','B','C','D']},
    {'Evaluator1': ['C','D','B','A'], 'Evaluator2': ['C','B','D','A'], 'Evaluator3': ['C','B','D','A'], 'Evaluator4': ['C','B','D','A']},
    {'Evaluator1': ['C','D','B','A'], 'Evaluator2': ['C','D','A','B'], 'Evaluator3': ['C','D','A','B'], 'Evaluator4': ['C','D','A','B']},
    {'Evaluator1': ['D','B','A','C'], 'Evaluator2': ['D','A','C','B'], 'Evaluator3': ['D','A','B','C'], 'Evaluator4': ['D','A','B','C']},
    {'Evaluator1': ['B','D','A','C'], 'Evaluator2': ['D','B','C','A'], 'Evaluator3': ['B','D','A','C'], 'Evaluator4': ['B','D','A','C']},
    {'Evaluator1': ['D','C','B','A'], 'Evaluator2': ['D','C','B','A'], 'Evaluator3': ['D','C','B','A'], 'Evaluator4': ['D','C','B','A']},
    {'Evaluator1': ['A','B','D','C'], 'Evaluator2': ['B','D','C','A'], 'Evaluator3': ['A','B','D','C'], 'Evaluator4': ['A','B','D','C']},
    {'Evaluator1': ['D','B','A','C'], 'Evaluator2': ['D','C','B','A'], 'Evaluator3': ['D','C','B','A'], 'Evaluator4': ['D','C','B','A']},
    {'Evaluator1': ['D','C','A','B'], 'Evaluator2': ['D','B','C','A'], 'Evaluator3': ['D','B','C','A'], 'Evaluator4': ['D','B','C','A']},
    {'Evaluator1': ['D','B','A','C'], 'Evaluator2': ['D','B','C','A'], 'Evaluator3': ['D','A','B','C'], 'Evaluator4': ['D','A','B','C']},
    {'Evaluator1': ['A','B','D','C'], 'Evaluator2': ['A','C','B','D'], 'Evaluator3': ['A','B','D','C'], 'Evaluator4': ['A','B','D','C']},
    {'Evaluator1': ['C','D','B','A'], 'Evaluator2': ['C','D','A','B'], 'Evaluator3': ['C','D','A','B'], 'Evaluator4': ['C','D','A','B']},
    {'Evaluator1': ['D','B','A','C'], 'Evaluator2': ['B','D','A','C'], 'Evaluator3': ['B','D','A','C'], 'Evaluator4': ['B','D','A','C']},
    {'Evaluator1': ['D','C','B','A'], 'Evaluator2': ['D','B','A','C'], 'Evaluator3': ['B','D','C','A'], 'Evaluator4': ['B','D','C','A']},
]

# Kendall's W 계산 함수
def kendalls_w(rank_matrix):
    n, m = rank_matrix.shape
    rank_sums = np.sum(rank_matrix, axis=1)
    S = np.sum((rank_sums - np.mean(rank_sums))**2)
    W = 12 * S / (m**2 * (n**3 - n))
    return W

# 인간 평가자끼리의 Kendall's W 계산
kendall_ws_human = []
for sample in samples_text:
    rank_matrix = np.array([
        [sample[evaluator].index(model) + 1 for model in ['A', 'B', 'C', 'D']]
        for evaluator in sample
    ]).T
    W = kendalls_w(rank_matrix)
    kendall_ws_human.append(W)

average_w_human = np.mean(kendall_ws_human)

# 결과 출력
print("인간 평가자 Kendall's W 평균:", round(average_w_human, 4))
