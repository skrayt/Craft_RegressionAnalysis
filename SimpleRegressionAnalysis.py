import os
import pandas as pd
import statsmodels.api as sm

outerIndicatorFolder = 'Indicator'
outerIndicatorFile = '日経平均東証株価国債利回.csv'
outerIndicatorFilePath = os.path.join(outerIndicatorFolder,outerIndicatorFile)

pdFolder = 'Indicator'
pdFile = 'v_pd.csv'
pdFilePath = os.path.join(pdFolder,pdFile)

outerIndicator_df = pd.read_csv(outerIndicatorFilePath)

nikkei_df = outerIndicator_df[['時点','日経平均株価【円】']]

# 日経平均株価【円】の年月列を統一 (nikkei_dfにkijyunnengetuのカラムを追加し、日付型に変換したのちyyyymmの文字列にフォーマット)
# .locを使用し、元のデータフレームのスライスに直接値を設定することを防ぐ
nikkei_df.loc[:,"kijyunnengetu"] = pd.to_datetime(nikkei_df["時点"], format="%Y年%m月").dt.strftime("%Y%m")

pd_df = pd.read_csv(pdFilePath,dtype={'kijyunnengetu':str})
pd_df = pd_df[['kijyunnengetu','pd']]

# 基準の年月でデータを結合（inner結合）
merged_df = pd.merge(nikkei_df, pd_df, on="kijyunnengetu", how="inner")

# 欠損値を除外
merged_df = merged_df.dropna()

# 説明変数（独立変数）と目的変数（従属変数）を設定
X = merged_df["日経平均株価【円】"]
y = merged_df["pd"]

# 定数項を追加
X = sm.add_constant(X)

# 回帰モデルの作成とフィッティング
model = sm.OLS(y, X).fit()

# 結果を表示
print(model.summary())

