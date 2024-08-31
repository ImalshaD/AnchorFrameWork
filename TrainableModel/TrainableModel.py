import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score
from abc import ABC, abstractmethod


import torch.nn as nn
import torch.optim as optim

class TrainableModel(nn.Module, ABC):

    def __init__(self, criterion : nn.Module, input_dim : int) -> None:
        super(TrainableModel, self).__init__()
        self.optimizer = None
        self.criterion = criterion
        self.input_dim = input_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def compile(self, **kwrds):

        lr = kwrds.get("lr", 1e-4)
        optimizer = kwrds.get("optimizer", None)
        if (optimizer == None):
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        else:
            self.optimizer = optimizer(self.parameters(),lr=lr)
    
    def trainModel(self, train_loader : DataLoader, val_loader : DataLoader, epochs : int):
        if (self.optimizer == None):
            raise ValueError("Model not compiled")
        for epoch in tqdm(range(epochs), desc="Training Model"):
            self.trainSelf(train_loader,self.criterion,self.optimizer,self.device)
            avg_loss, acc , f1 = self.evaluate(val_loader,self.criterion,self.device)
            print(f"Val Results (epoch : {epoch}) : accuracy {acc} , f1_score {f1}, avg_loss {avg_loss}")

    def evalModel(self, test_loader : DataLoader):
        return self.evaluate(test_loader,self.criterion,self.device)

    def trainSelf(self, train_loader : DataLoader, criterion : nn.Module, optimizer : optim.Optimizer, device, **kwrds):
        self.to(device)
        self.train()
        total_loss = 0

        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = self(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def evaluate(self, test_loader : DataLoader, criterion : nn.Module, device, **kwrds):
        self.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for embeddings, labels in test_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = self(embeddings)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = self.getPrediction(outputs)   #torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # Collect all labels and predictions for F1 score calculation
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        return avg_loss, accuracy, f1
    
    @abstractmethod
    def getPrediction(self, outputs):
        pass

    @abstractmethod
    def getCopy(self) -> 'TrainableModel':
        pass
