# Imports
import torch.optim as optim
from torch import nn
import torch

from tqdm import tqdm

class LeNeT(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    self.layer1 = nn.Sequential(
      nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0), # in: 1*28*28 | out: 6*24*24
      nn.BatchNorm2d(6),
      nn.LeakyReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2) # in: 6*24*24 | out: 6*12*12
    )
    self.layer2 = nn.Sequential(
      nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0), # in: 6*12*12 | out: 16*8*8
      nn.BatchNorm2d(16),
      nn.LeakyReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2) # in: 16*8*8 | out: 16*4*4
    )
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(16*4*4, 128),
      nn.LeakyReLU(),
      nn.Linear(128, 64),
      nn.Linear(64, 10)
    )
    self.device = (
      'cuda' if torch.cuda.is_available()
      else 'mps' if torch.backends.mps.is_available()
      else 'cpu'
    )
    print(f'Using device: {self.device}\n')
    
  def forward(self, X):
    X = X.view(-1, 1, 28, 28)
    X = self.layer1(X)
    X = self.layer2(X)
    X = self.flatten(X)
    X = self.linear_relu_stack(X)
    return X
  
  def _train(self, train_dataloader, criterion, optimizer):
    self.train()

    for batch, (data, label) in enumerate(train_dataloader):
        data, label = data.to(self.device), label.to(self.device)

        predict = self.forward(data)
        loss = criterion(predict, label)

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
          loss, current = loss.item(), (batch+1) * len(data)
          print(f'Loss: {loss:>7f}, [{current:>5d}/{len(train_dataloader.dataset)}]\n')

  def _test(self, test_dataloader, criterion):
    self.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():
      for data, label in test_dataloader:
        data, label = data.to(self.device), label.to(self.device)

        predict = self.forward(data)
        
        test_loss += criterion(predict, label).item()
        correct += (predict.argmax(1) == label).type(torch.float).sum().item()
    test_loss /= len(test_dataloader)
    correct /= len(test_dataloader.dataset)
    print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n')

  def fit(self, train_dataloader, test_dataloader, epochs=5, lr=1e-3, weight_decay=1e-5, ):
    print('Training model...\n')
    self.to(self.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    p_bar = tqdm(range(epochs))
    
    for epoch in range(epochs):
      p_bar.set_postfix_str(f'Epoch: {epoch+1}')
      self._train(train_dataloader=train_dataloader, criterion=criterion, optimizer=optimizer)
      self._test(test_dataloader=test_dataloader, criterion=criterion)
