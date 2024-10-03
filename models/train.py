import numpy as np
import torch
import pandas as pd
from torch import nn
from sklearn.model_selection import train_test_split
from random import shuffle

data = pd.read_csv('../code/dataset/apple_quality.csv')
data = data.dropna()
data = data[['Size', 'Weight', 'Juiciness']]
train, test = train_test_split(data, test_size=0.2)

X_train, y_train = train[['Size', 'Weight']], train[['Juiciness']]
X_test, y_test = test[['Size', 'Weight']], test[['Juiciness']]

X_train, y_train, X_test, y_test = (torch.tensor(np.array(i), dtype=torch.float32) for i in
                                    (X_train, y_train, X_test, y_test))


def get_batches(size=32, test=False):
    X, y = (X_test, y_test) if test else (X_train, y_train)
    i = list(range(len(X)))
    shuffle(i)
    X, y = X[i], y[i]
    X, y = X[:len(X) // 32 * 32], y[:len(y) // 32 * 32]
    X, y = X.reshape((-1, 32, 2)), y.reshape((-1, 32, 1))
    return X, y


model = nn.Sequential(
    nn.Linear(2, 4),
    nn.Sigmoid(),
    nn.Linear(4, 4),
    nn.Sigmoid(),
    nn.Linear(4, 1)
)

loss_fn = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), 3e-5)

best = float('inf')
for epoch in range(3000):
    print(f'Epoch {epoch + 1}...')
    Loss = []
    for input, label in zip(*get_batches()):
        output = model(input)
        loss = loss_fn(output, label)
        Loss.append(loss.item())

        model.zero_grad()
        loss.backward()
        optim.step()
    Loss = sum(Loss) / len(Loss)
    print(f'Train loss: {Loss}')
    Loss = []
    for input, label in zip(*get_batches(test=True)):
        output = model(input)
        loss = loss_fn(output, label)
        Loss.append(loss.item())
    Loss = sum(Loss) / len(Loss)
    print(f'Train loss: {Loss}')
    if Loss < best:
        best = Loss
        torch.save(model, 'model.pt')
        print('saved')
