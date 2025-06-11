import matplotlib.pyplot as plt
import seaborn as sns

from db.database import read_dataframe_from_sqlite

df = read_dataframe_from_sqlite('standardized_data')



xlabel = '国内企業物価指数（総平均）2020年基準'
ylabel = 'pd'

plt.figure(figsize=(8,6))
sns.scatterplot(x = xlabel, y = ylabel,data=df)
plt.title('散布図')
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.show()