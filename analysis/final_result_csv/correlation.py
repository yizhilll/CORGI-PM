import pandas as pd 
import seaborn as sns
import numpy as np





data = pd.read_csv("final_result.csv")


data1 = pd.DataFrame({'x':data['score_ernie'],'y':data["score_cbert"]})
print(data1)

print(data1.corr(method='pearson'))
print(data1.corr(method='spearman'))
print(data1.corr(method='kendall'))

print(sns.heatmap(data1.corr(),linewidths=0.1,vmax=1.0, square=True,linecolor='white', annot=True))