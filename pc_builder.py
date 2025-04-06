import pandas as pd
gpu_df = pd.read_csv("./Data/GPU Data.csv")
budget = 1500

def find_gpu():
    for i in gpu_df.iloc:
        if i["Price"] < budget/2:
            return i["GPU Name"]

gpu = find_gpu()
print(gpu)