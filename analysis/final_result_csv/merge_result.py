import pandas as pd 
# data1 = pd.read_csv("test_result_ernie_cbert.csv")
# data2 = pd.read_csv("Score.csv")

# data = pd.merge(data1,data2,on="形容词")

# data.to_csv("final_result.csv",encoding="utf-8-sig")
data1 = pd.read_csv("final_result.csv")
data2 = pd.read_csv("/Users/a511/Desktop/zhangge/Coling_selftest/CBERT & ERNIE/result_ele-dis——adj.csv")
data3 = pd.read_csv("/Users/a511/Desktop/zhangge/Coling_selftest/CBERT & ERNIE/result_xlnet-base_adj.csv")


data = pd.merge(data1,data2,on="形容词")
data = pd.merge(data,data3,on="形容词")

data.to_csv("final_result_adj5LM.csv",encoding="utf-8-sig")