
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid




df = pd.read_csv(r"C:\Users\dadbc\Desktop\Phy\Repositorios\Riot_Test\UseData\Datos_User.csv")

m = {True: 1, False: 0}
df["Win"] = df["Win"].map(m)

df = pd.get_dummies(df, columns=["Position"])

df = pd.get_dummies(df, columns=["Champion_Name"])

df = df.drop(columns=["Username"])


from sklearn.model_selection import train_test_split

X = df.drop(columns=["Win"])
Y = df["Win"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state= 42)

X_train = torch.FloatTensor(X_train.values)
X_test = torch.FloatTensor(X_test.values)
Y_train = torch.FloatTensor(Y_train.values)
Y_test = torch.FloatTensor(Y_test.values)

print(df)

class GranAmigo(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(39, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)


        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.out(x)
        return x

Neural = GranAmigo()

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(Neural.parameters(), lr= 0.001)

epochs = 1000
for i in range(epochs):
    optimizer.zero_grad()
    pred = Neural.forward(X_train)
    pred = torch.reshape(pred, (51,))
    loss = criterion(pred, Y_train)

    if i % 10 == 0:
        print("Iteration", i, "Loss", loss)

    loss.backward()
    optimizer.step()


y_pred = []

with torch.no_grad():
    for data in X_test:
        y = (Neural.forward(data))
        prediction = torch.sigmoid(y)
        prediction = torch.round(prediction)
        y_pred.append(prediction)

fallos = 0
for i in range(len(y_pred)):
    if y_pred[i] != Y_test[i]:
        fallos = fallos + 1



Y_test = list(Y_test)
dev = pd.DataFrame(Y_test, columns=["Real"])
dev["Prediction"] = y_pred
print(dev)
print(fallos)