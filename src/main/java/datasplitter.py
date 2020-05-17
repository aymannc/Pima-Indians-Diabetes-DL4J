import pandas as pd
from numpy.random import RandomState

path = '../resources/'
df = pd.read_csv(path + "diabetes.csv")
rng = RandomState()

train = df.sample(frac=0.7, random_state=rng)
test = df.loc[~df.index.isin(train.index)]
train.to_csv(path + "diabetes-train.csv", index=False)
test.to_csv(path + "diabetes-test.csv", index=False)
print(df)
print(train)
