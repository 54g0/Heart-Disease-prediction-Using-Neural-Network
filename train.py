import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from model import HeartDiseaseNN
#Data-Preprocessing

data = pd.read_csv("/content/Downloads")

#Data-Splitting

x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
X_train, X_test,y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#Standardization of Data

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Converting data into Tensors for Efficient Neural Network Processing and Creating Dataset Using TensorDataset and DataLoader

X_train_tensor = torch.tensor(X_train, dtype = torch.float32)
y_train_tensor = torch.tensor(y_train, dtype = torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype = torch.float32)
y_test_tensor = torch.tensor(y_test, dtype = torch.float32).unsqueeze(1)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = True)

#Model
model = HeartDiseaseNN(input_size=X_train.shape[1])

# Training loop
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
n = 30
for epoch in range(n):
    model.train()
    for X, y in train_loader:
        y_pred = model(X)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{n}, Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), 'model.pth')
