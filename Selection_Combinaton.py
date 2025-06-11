from MultiplepleRegressionAnalysis import ScalingIndicator,exhaustive_vif_search
from constants import Constants as const

# 標準化された後の説明変数
Scaled0 = const.ScaledItem[0]
Scaled1 = const.ScaledItem[1]
Scaled2 = const.ScaledItem[2]
Scaled3 = const.ScaledItem[3]
Scaled4 = const.ScaledItem[4]
Scaled5 = const.ScaledItem[5]
Scaled6 = const.ScaledItem[6]
Scaled7 = const.ScaledItem[7]

X = ScalingIndicator()[[Scaled0,Scaled1,Scaled2,Scaled3,Scaled4,Scaled5,Scaled6,Scaled7]].copy()
y = ScalingIndicator()['pd_standardized'].copy()
vif_threshold=5     # VIF値
corr_threshold=0.5  # 相関係数
num_variables=2      # 説明変数の数

# X: 説明変数のデータフレーム
# y: 目的変数（シリーズ形式）
valid_combinations = exhaustive_vif_search(X, y, vif_threshold, corr_threshold, num_variables)

# 結果を表示
print("基準を満たすすべての組み合わせ:")
for combo in valid_combinations:
    print(f"変数: {combo['変数']}, 最大VIF: {combo['最大VIF']}")
    print(f"目的変数との相関係数: {combo['目的変数との相関係数']}")