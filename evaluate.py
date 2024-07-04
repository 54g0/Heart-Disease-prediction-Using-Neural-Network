import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from model import HeartDiseaseNN
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from model import HeartDiseaseNN

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

model = HeartDiseaseNN(input_size=X_test.shape[1])
model.load_state_dict(torch.load('model.pth'))

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        # print(outputs)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Accuracy on test set: {accuracy * 100:.2f}%')
