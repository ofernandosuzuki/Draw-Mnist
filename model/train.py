from torch.utils.data import TensorDataset, DataLoader
import torch

from model_training_script.model import LeNeT

from sklearn.model_selection import train_test_split
import pandas as pd 

print('Importing data...\n')
data = pd.read_csv('../data/mnist.csv', low_memory=False).values
labels = pd.read_csv('../data/mnist_labels.csv', low_memory=False).values.reshape(-1)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=True)

X_train_tensor = torch.tensor(data=X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(data=X_test, dtype=torch.float32)

y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=64)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=64)

model = LeNeT()
model.fit(train_dataloader=train_dataloader, test_dataloader=test_dataloader, epochs=10)

print('Saving model...\n')
#torch.save(model.state_dict(), "./trained_models/LeNet.pth")