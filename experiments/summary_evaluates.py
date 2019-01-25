import glob

import pandas as pd

files = glob.glob('other_models/results/*.csv', recursive=True)


frames = [pd.read_csv(file) for file in files]
data = pd.concat(frames)

values_columns = [f"{i}" for i in range(0, 6)]

data_grouped = data.groupby(['model', 'evaluate_method', 'evaluation']).mean()

print("-----------------------")
print("Full data")
print("-----------------------")
print(data_grouped[values_columns])
print()

print("-----------------------")
print("Mean")
print("-----------------------")
print(data_grouped[values_columns].T.mean())
print()


#print("-----------------------")
#print("Only test")
#print("-----------------------")
#print(data_grouped[values_columns].T.mean())
#print()
