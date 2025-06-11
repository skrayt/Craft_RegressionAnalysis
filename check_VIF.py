'''
相関関係とVIF値の確認
'''
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from MultiplepleRegressionAnalysis import ScalingIndicator
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

#* ======================================================================
#* step2から標準化した経済指標とPDを結合したデータフレームを呼び出し
#* ======================================================================

def Check_VIF():

    scaledMerged_df = ScalingIndicator()

    # 説明変数を選択
    X = scaledMerged_df[[Scaled0,Scaled1,Scaled2,Scaled3,Scaled4,Scaled5,Scaled6,Scaled7]].copy()

    # VIFを計算
    vifData = pd.DataFrame()
    vifData['変数']= X.columns
    vifData['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]

    # VIFを表示
    print ('VIFの結果：')
    print(vifData)

Check_VIF()