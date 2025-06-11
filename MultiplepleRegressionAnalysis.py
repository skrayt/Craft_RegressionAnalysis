import os
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
from itertools import combinations

#* ======================================================================
#*説明変数とその標準化されたカラムを定数として扱う
#* ======================================================================
@dataclass(frozen=True)
class Constants:
    IndicatorItem = (
        "完全失業率（男女計）【%】",
        "日経平均株価【円】",
        "有効求人倍率【倍】",
        "新発10年国債利回り（月末終値）【%】",
        "東証株価指数（TOPIX）",
        "消費者物価指数（総合）2020年基準",
        "国内企業物価指数（総平均）2020年基準",
        "新設住宅着工戸数（総戸数）【戸】",
    )  # タプルにして変更不可にする

    ScaledItem = (
        "完全失業率",
        "日経平均株価",
        "有効求人倍率",
        "国債利回り",
        "東証株価指数",
        "消費者物価指数",
        "国内企業物価指数",
        "新設住宅着工戸数",
    )



# 説明変数
Indicator0 = Constants.IndicatorItem[0]
Indicator1 = Constants.IndicatorItem[1]
Indicator2 = Constants.IndicatorItem[2]
Indicator3 = Constants.IndicatorItem[3]
Indicator4 = Constants.IndicatorItem[4]
Indicator5 = Constants.IndicatorItem[5]
Indicator6 = Constants.IndicatorItem[6]
Indicator7 = Constants.IndicatorItem[7]

# 標準化された後の説明変数
Scaled0 = Constants.ScaledItem[0]
Scaled1 = Constants.ScaledItem[1]
Scaled2 = Constants.ScaledItem[2]
Scaled3 = Constants.ScaledItem[3]
Scaled4 = Constants.ScaledItem[4]
Scaled5 = Constants.ScaledItem[5]
Scaled6 = Constants.ScaledItem[6]
Scaled7 = Constants.ScaledItem[7]


#* ======================================================================
#* step1:説明変数：経済指標と目的変数（PD/LGD）をデータフレーム(DF)に変換
#* ======================================================================
# 説明変数：経済指標
def ExplanatoryVariable(region):
    EconomicIndicatorFolder = 'Indicator'
    EconomicIndicatorFile = '各種経済指標.csv'
    EconomicIndicatorFilePath = os.path.join(EconomicIndicatorFolder,EconomicIndicatorFile)

    EconomicIndicator_df = pd.read_csv(EconomicIndicatorFilePath)

    # 指標のうち、特定地域の指標のみを抽出
    ExplanatoryVarDF = EconomicIndicator_df[EconomicIndicator_df['地域'] == region]

    # kijyunnengetuのカラムを追加し、日付型に変換したのちyyyymmの文字列にフォーマット)
    ExplanatoryVarDF['kijyunnengetu'] = pd.to_datetime(ExplanatoryVarDF["時点"], format="%Y年%m月").dt.strftime("%Y%m")

    return ExplanatoryVarDF

# 目的変数
def ResponseVariable():
    pdFolder = 'Indicator'
    pdFile = 'v_pd.csv'
    pdFilePath = os.path.join(pdFolder,pdFile)

    ResponseVarDf = pd.read_csv(pdFilePath,dtype={'kijyunnengetu':str})
    ResponseVarDf = ResponseVarDf[['kijyunnengetu','pd']]

    return ResponseVarDf

#* ======================================================================
#* step2：経済指標を標準化し、標準化後の説明変数と目的変数を結合したDFを返す
#* ======================================================================
def ScalingIndicator():

    # 説明変数のDFから必要な指標のみを抽出
    Exp_df = ExplanatoryVariable('全国')[['時点','kijyunnengetu',Indicator0,Indicator1,Indicator2,Indicator3,Indicator4,Indicator5,Indicator6,Indicator7]].copy()

    #* ======================================================================
    #* 欠損値の確認 今回はスキップ
    #* ======================================================================

    #@ 必要に応じて欠損値の行を削除、欠損値を[中央値、平均値等]で埋める作業を行う。
    #@ ※今回はスキップ
    '''
    # パターン1:欠損値の行を削除
    Exp_df = Exp_df.dropna()

    # パターン2:欠損値を中央値で埋める
    Exp_df.fillna(Exp_df.median(),inplace=True)

    # パターン3:欠損値を平均値で埋める
    Exp_df.fillna(Exp_df.mean(),inplace=True)

    # 欠損値を埋めた列の統計情報を確認し、埋めた値がデータの分布に適しているか確認
    print(Exp_df.describe())
    '''

    # kijyunnengetuでデータをinner結合
    Merged_df = pd.merge(Exp_df, ResponseVariable(), on="kijyunnengetu", how="inner") # ResponseVariable(): 目的変数

    # StandardScalerを初期化
    scaler = StandardScaler()

    # 説明変数を標準化
    scaledFeatures = scaler.fit_transform(Merged_df[[Indicator0,Indicator1,Indicator2,Indicator3,Indicator4,Indicator5,Indicator6,Indicator7]])

    # 標準化したデータをデータフレームに変換
    scaledData = pd.DataFrame(scaledFeatures,columns=[Scaled0,Scaled1,Scaled2,Scaled3,Scaled4,Scaled5,Scaled6,Scaled7])

    # 目的変数を標準化
    Merged_df['pd_standardized'] = scaler.fit_transform(Merged_df[['pd']])

    # 元のデータフレームに結合
    scaledMerged_df = pd.concat([Merged_df['pd_standardized'],scaledData],axis=1) # axis：結合方向。0:'index'(縦)　1：'columns'(横)

    #@ 必要に応じてdescribeを確認し、説明変数のmean(平均値)がほぼ0、std(標準偏差)がほぼ1であれば正しく標準化できており、OK
    # print(scaledMerged_df.describe())

    return scaledMerged_df

#* ======================================================================
#* step3：VIF値を計算して多重共線性の可能性を排除した相関係数の高いペアを割り出す
#* ======================================================================

X = ScalingIndicator()[[Scaled0,Scaled1,Scaled2,Scaled3,Scaled4,Scaled5,Scaled6,Scaled7]].copy()
y = ScalingIndicator()['pd_standardized']
vif_threshold=5     # VIF値
corr_threshold=0.5  # 相関係数
num_variables=2      # 説明変数の数

# VIF計算関数
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["変数"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# 全組み合わせ探索 + 修正基準の適用
def exhaustive_vif_search(X, y, vif_threshold, corr_threshold, num_variables):
    variables = X.columns.tolist()
    valid_combinations = []

    # 指定された数の変数の組み合わせを試す
    for combination in combinations(variables, num_variables):
        X_subset = X[list(combination)]

        # VIFを計算
        vif_data = calculate_vif(X_subset)

        # VIFがすべて閾値未満であるか確認
        if all(vif_data["VIF"] < vif_threshold):
            # 各説明変数と目的変数との相関係数を計算
            corr_with_target = {var: round(float(y.corr(X[var])), 2) for var in combination}

            # 各説明変数と目的変数間の相関係数が閾値以上であるか確認
            if all(abs(corr) >= corr_threshold for corr in corr_with_target.values()):
                valid_combinations.append({
                    "変数": combination,
                    "最大VIF": round(vif_data["VIF"].max(), 2),
                    "目的変数との相関係数": corr_with_target
                })

    return valid_combinations

#todo Selection_Combinaton.pyを実行して、モデルを作成する説明変数の組合せを決定する

#* ======================================================================
#* step4：VIF値を計算して多重共線性の可能性を排除した相関係数の高いペアを割り出す
#* ======================================================================

def MultipleRegression():

    # 選ばれた変数の組み合わせ（例: '日経平均株価', '国内企業物価指数'）
    selected_variables = ['日経平均株価', '国内企業物価指数']

    # 説明変数と目的変数を設定
    X_selected = X[selected_variables]
    y_target = y

    # 定数項を追加（Statsmodels用）
    X_selected_with_const = sm.add_constant(X_selected)

    # Statsmodelsで線形回帰モデルを構築
    model = sm.OLS(y_target, X_selected_with_const).fit()

    # モデルの概要を表示
    print(model.summary())

    # Scikit-learnで交差検証を実施
    lr = LinearRegression()
    cross_val_scores = cross_val_score(lr, X_selected, y_target, cv=5, scoring='neg_mean_squared_error')

    # 平均二乗誤差（MSE）の平均値を表示
    print(f"交差検証の平均MSE: {-cross_val_scores.mean():.2f}")

    # モデルの予測値を計算
    y_pred = model.predict(X_selected_with_const)

    # 平均二乗誤差（MSE）を計算
    mse = mean_squared_error(y_target, y_pred)
    print(f"モデルのMSE: {mse:.2f}")

    return model

def residuals_search():
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf

    model = MultipleRegression()

    plt.rcParams['font.family'] = 'MS Gothic'

    # 残差を取得
    residuals = model.resid

    # 自己相関プロットを作成
    plot_acf(residuals, lags=20)
    plt.title("残差の自己相関プロット")
    plt.xlabel("ラグ")
    plt.ylabel("自己相関")
    plt.show()

residuals_search()
