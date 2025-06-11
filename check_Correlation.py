
#* ======================================================================
#* 相関関係の確認
#* ======================================================================

import seaborn as sns
import matplotlib.pyplot as plt
from MultiplepleRegressionAnalysis import ScalingIndicator

def Check_CorrelationCoefficient():

    scaledMerged_df = ScalingIndicator()

    # seabornの日本語文字化け防止にフォントを指定する
    sns.set(font= 'Yu Gothic')

    # 相関行列を計算
    correlation_matrix = scaledMerged_df.corr()

    # 相関行列を表示
    print('相関行列:')
    print(correlation_matrix)

    # ヒートマップで可視化
    plt.figure(figsize=(10,8))
    sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',fmt='.2f')
    plt.title('相関行列のヒートマップ')
    plt.show()

Check_CorrelationCoefficient()