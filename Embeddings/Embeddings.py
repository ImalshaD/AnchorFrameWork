import numpy as np
import torch
from Dsets import NLPDataset
from torch.utils.data import DataLoader, Dataset as TDT , TensorDataset
from .EmbeddingsModel import EmbeddingsModel

class Embeddings:    
    def __init__(self, embeddings, target) -> None:
        self.embeddings: np.ndarray = embeddings
        self.target: np.ndarray= target

    @staticmethod
    def generateEmbeddings(dataset : NLPDataset, model : EmbeddingsModel, batch_size = 8):
        embeddings : np.ndarray = model.generateEmbeddings(dataset,batch_size)
        target : np.ndarray = dataset.getYasTensor()
        return Embeddings(embeddings, target)
    def __getitem__(self, index):
        return self.embeddings[index], self.target[index]
    
    def getDataLoader(self, batch_size = 8, shuffle = False) -> DataLoader:
        embeddings = torch.tensor(self.embeddings, dtype=torch.float32)
        dataset = TensorDataset(embeddings,self.target)
        return DataLoader(dataset,batch_size=batch_size, shuffle=shuffle)
