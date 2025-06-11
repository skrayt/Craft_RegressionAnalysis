import os
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler

#* ======================================================================
#*説明変数とその標準化されたカラムを定数として扱う
#* ======================================================================
@dataclass(frozen=True)
class Constants:
    IndicatorItem = [
        "完全失業率（男女計）【%】",
        "日経平均株価【円】",
        "有効求人倍率【倍】",
        "新発10年国債利回り（月末終値）【%】",
        "東証株価指数（TOPIX）",
        "消費者物価指数（総合）2020年基準",
        "国内企業物価指数（総平均）2020年基準",
        "新設住宅着工戸数（総戸数）【戸】",
    ]  # タプルにして変更不可にする

    ScaledItem = [
        "完全失業率",
        "日経平均株価",
        "有効求人倍率",
        "国債利回り",
        "東証株価指数",
        "消費者物価指数",
        "国内企業物価指数",
        "新設住宅着工戸数",
    ]

# 説明変数
Indicator = Constants.IndicatorItem
# 標準化された後の説明変数
Scaled = Constants.ScaledItem

# 説明変数：経済指標
def explanatory_variable():
    EconomicIndicatorFolder = 'Indicator'
    EconomicIndicatorFile = '各種経済指標.csv'
    EconomicIndicatorFilePath = os.path.join(EconomicIndicatorFolder,EconomicIndicatorFile)

    EconomicIndicator_df = pd.read_csv(EconomicIndicatorFilePath)

    # 指標のうち、特定地域の指標のみを抽出
    ExplanatoryVarDF = EconomicIndicator_df[EconomicIndicator_df['地域'] == '全国']

    # kijyunnengetuのカラムを追加し、日付型に変換したのちyyyymmの文字列にフォーマット)
    ExplanatoryVarDF['kijyunnengetu'] = pd.to_datetime(ExplanatoryVarDF["時点"], format="%Y年%m月").dt.strftime("%Y%m")

    ExplanatoryVarDF = ExplanatoryVarDF[['kijyunnengetu'] + Indicator]

    return ExplanatoryVarDF

# 目的変数
def response_variable():
    pdFolder = 'Indicator'
    pdFile = 'v_pd.csv'
    pdFilePath = os.path.join(pdFolder,pdFile)

    ResponseVarDf = pd.read_csv(pdFilePath,dtype={'kijyunnengetu':str})
    ResponseVarDf = ResponseVarDf[['kijyunnengetu','pd']]

    return ResponseVarDf

# 説明変数と目的変数を結合
def merged_variable():

    # kijyunnengetuでデータをinner結合
    MergedDf = pd.merge(explanatory_variable(), response_variable(), on="kijyunnengetu", how="inner")

    #@ 必要に応じて欠損値の行を削除、欠損値を[中央値、平均値等]で埋める作業を行う。
    # print(MergedDf.describe())
    #@ ※今回はスキップ
    # # パターン1:欠損値の行を削除
    # Exp_df = Exp_df.dropna()

    # # パターン2:欠損値を中央値で埋める
    # Exp_df.fillna(Exp_df.median(),inplace=True)

    # # パターン3:欠損値を平均値で埋める
    # Exp_df.fillna(Exp_df.mean(),inplace=True)

    return MergedDf

# 欠損値補正後の統計情報を確認し、埋めた値がデータの分布に適しているか確認
# print(merged_variable().describe())

# 説明変数と目的変数を標準化
def standardized_variable():

    # StandardScalerを初期化
    scaler = StandardScaler()

    scalerColumns = Indicator + ['pd']

    ScaledFeatures = scaler.fit_transform(merged_variable()[scalerColumns])

    StandardizedMergedDf = pd.DataFrame(ScaledFeatures,columns=scalerColumns)

    StandardizedMergedDf = pd.concat([merged_variable()['kijyunnengetu'],StandardizedMergedDf],axis=1)

    return StandardizedMergedDf
