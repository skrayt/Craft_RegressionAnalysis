import matplotlib.pyplot as plt
import pandas as pd

from components.data_cleansing import standardized_variable

standardized_data = standardized_variable()["日経平均株価【円】"]

standardized_df = pd.DataFrame(standardized_data)


# # 時系列データのプロット
# plt.plot(standardized_df)
# plt.title('Time Series Data')
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.show()

# 移動平均・移動分散の確認
# 移動平均・移動分散の計算
standardized_df["rooling_mean"] = standardized_df["日経平均株価【円】"].rolling(window=12).mean()
standardized_df["rolling_variance"] = (
    standardized_df["日経平均株価【円】"].rolling(window=12).var()
)

# # プロット
# plt.plot(standardized_df["日経平均株価【円】"], label="Origial Data")
# plt.plot(standardized_df["rooling_mean"], label="Rolling Mean", linestyle="--")
# plt.plot(standardized_df["rolling_variance"], label="Rolling Variance", linestyle=":")
# plt.show()

# ADF検定
