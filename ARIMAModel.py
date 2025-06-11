import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from constants import Constants as const
import numpy as np

# 説明変数
Indicator0 = const.IndicatorItem[0]
Indicator1 = const.IndicatorItem[1]
Indicator2 = const.IndicatorItem[2]
Indicator3 = const.IndicatorItem[3]
Indicator4 = const.IndicatorItem[4]
Indicator5 = const.IndicatorItem[5]
Indicator6 = const.IndicatorItem[6]
Indicator7 = const.IndicatorItem[7]

# 標準化された後の説明変数
Scaled0 = const.ScaledItem[0]
Scaled1 = const.ScaledItem[1]
Scaled2 = const.ScaledItem[2]
Scaled3 = const.ScaledItem[3]
Scaled4 = const.ScaledItem[4]
Scaled5 = const.ScaledItem[5]
Scaled6 = const.ScaledItem[6]
Scaled7 = const.ScaledItem[7]

plt.rcParams['font.family'] = 'MS Gothic'

#* ======================================================================
#* step1:説明変数：経済指標と目的変数（PD/LGD）をデータフレーム(DF)に変換
#* ======================================================================
# 説明変数：経済指標
def ExplanatoryVariable():
    EconomicIndicatorFolder = 'Indicator'
    EconomicIndicatorFile = '各種経済指標.csv'
    EconomicIndicatorFilePath = os.path.join(EconomicIndicatorFolder,EconomicIndicatorFile)

    EconomicIndicator_df = pd.read_csv(EconomicIndicatorFilePath)

    # 指標のうち、特定地域の指標のみを抽出
    ExplanatoryVarDF = EconomicIndicator_df[EconomicIndicator_df['地域'] == '全国']

    # kijyunnengetuのカラムを追加し、日付型に変換したのちyyyymmの文字列にフォーマット)
    ExplanatoryVarDF['kijyunnengetu'] = pd.to_datetime(ExplanatoryVarDF["時点"], format="%Y年%m月").dt.strftime("%Y%m")
    ExplanatoryVarDF['kijyunnengetu'] = pd.to_datetime(ExplanatoryVarDF['kijyunnengetu'],format='%Y%m') + pd.offsets.MonthEnd(0)

    # ExplanatoryVarDF.set_index('kijyunnengetu',inplace=True)

    return ExplanatoryVarDF

# 目的変数
def ResponseVariable():
    pdFolder = 'Indicator'
    pdFile = 'v_pd.csv'
    pdFilePath = os.path.join(pdFolder,pdFile)

    ResponseVarDf = pd.read_csv(pdFilePath)
    ResponseVarDf = ResponseVarDf[['kijyunnengetu','pd']]
    ResponseVarDf['kijyunnengetu'] = pd.to_datetime(ResponseVarDf["kijyunnengetu"],format='%Y%m') + pd.offsets.MonthEnd(0)

    # ResponseVarDf.set_index('kijyunnengetu',inplace=True)

    return ResponseVarDf

#* ======================================================================
#* step2：経済指標を標準化し、標準化後の説明変数と目的変数を結合したDFを返す
#* ======================================================================
def ScalingIndicator():

    # 説明変数のDFから必要な指標のみを抽出
    Exp_df = ExplanatoryVariable()[['kijyunnengetu',Indicator0,Indicator1,Indicator2,Indicator3,Indicator4,Indicator5,Indicator6,Indicator7]].copy()

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

    scaledMerged_df.index = Merged_df['kijyunnengetu']

    #@ 必要に応じてdescribeを確認し、説明変数のmean(平均値)がほぼ0、std(標準偏差)がほぼ1であれば正しく標準化できており、OK
    # print(scaledMerged_df.describe())


    return scaledMerged_df


scaledMerged_df = ScalingIndicator()

def SinglePlot():

    PlotDf = scaledMerged_df[['pd_standardized',Scaled0]]

    # 時系列データのプロット
    plt.figure(figsize=(12,6))
    # 各列をプロット
    for column in PlotDf.columns:
        plt.plot(PlotDf.index,PlotDf[column],label=column)

    # 具裸婦のタイトルとラベルを設定
    plt.title('Scaled Data Over Time')
    plt.xlabel('Date')
    plt.ylabel('ScaledValues')

    # 凡例を表示
    plt.legend(loc='upper left')

    # グリッドを追加
    plt.grid(True)

    # グラフを表示
    plt.show()

def MultiPlot():

    # figure()でグラフを描画する領域をつくり、figというオブジェクトにする
    fig = plt.figure(figsize=(10,6))

    # add_subplot()でグラフを描画する領域を追加する。引数は行、列、場所
    ax1 = fig.add_subplot(3,4,1)
    ax2 = fig.add_subplot(3,4,5)
    ax3 = fig.add_subplot(3,4,6)
    ax4 = fig.add_subplot(3,4,7)
    ax5 = fig.add_subplot(3,4,8)
    ax6 = fig.add_subplot(3,4,9)
    ax7 = fig.add_subplot(3,4,10)
    ax8 = fig.add_subplot(3,4,11)
    ax9 = fig.add_subplot(3,4,12)

    y1 = scaledMerged_df['pd_standardized']
    y2 = scaledMerged_df[Scaled0]
    y3 = scaledMerged_df[Scaled1]
    y4 = scaledMerged_df[Scaled2]
    y5 = scaledMerged_df[Scaled3]
    y6 = scaledMerged_df[Scaled4]
    y7 = scaledMerged_df[Scaled5]
    y8 = scaledMerged_df[Scaled6]
    y9 = scaledMerged_df[Scaled7]

    # 各プロットの色
    c1,c2,c3,c4,c5,c6,c7,c8,c9 = ['blue','green','red','black','purple','orange','grey','pink','brown']
    # 各ラベル
    l1,l2,l3,l4,l5,l6,l7,l8,l9 = scaledMerged_df.columns.values


    ax1.plot(scaledMerged_df.index,y1,color=c1,label=l1)
    ax2.plot(scaledMerged_df.index,y2,color=c2,label=l2)
    ax3.plot(scaledMerged_df.index,y3,color=c3,label=l3)
    ax4.plot(scaledMerged_df.index,y4,color=c4,label=l4)
    ax5.plot(scaledMerged_df.index,y5,color=c5,label=l5)
    ax6.plot(scaledMerged_df.index,y6,color=c6,label=l6)
    ax7.plot(scaledMerged_df.index,y7,color=c7,label=l7)
    ax8.plot(scaledMerged_df.index,y8,color=c8,label=l8)
    ax9.plot(scaledMerged_df.index,y9,color=c9,label=l9)
    # 凡例
    ax1.legend(loc= 'upper left')
    ax2.legend(loc= 'upper left')
    ax3.legend(loc= 'upper left')
    ax4.legend(loc= 'upper left')
    ax5.legend(loc= 'upper left')
    ax6.legend(loc= 'upper left')
    ax7.legend(loc= 'upper left')
    ax8.legend(loc= 'upper left')
    ax9.legend(loc= 'upper left')

    fig.tight_layout()
    plt.show()

pd_series = scaledMerged_df['pd_standardized']

plt.figure(figsize=(10,6))
plt.plot(pd_series,label = 'PD')
plt.title('Time Series of PD')
plt.xlabel('kijyunnengetu')
plt.ylabel('Value')
plt.legend()
plt.show()