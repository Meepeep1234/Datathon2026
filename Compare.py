import pandas as pd
import numpy as np
Number_correct = 0
Total = 0
Standard_file = pd.read_csv("val.csv")
Our_File = pd.read_csv("predictions.csv")
for index, row in Standard_file.iterrows():
    x = Standard_file.loc[index,'label']
    y = Our_File.loc[index,'label']
    print(x,y)
    if x == y:
        Number_correct += 1
    Total += 1
print((Number_correct/Total)*100,"%")
