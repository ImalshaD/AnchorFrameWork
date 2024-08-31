from Languages import XLMRLanguages
import torch
import numpy as np
from abc import ABC, abstractmethod
from Dsets import NLPDataset

class EmbeddingsModel(ABC):

    def __init__(self, languages : list[XLMRLanguages]) -> None:
        self.languages : list[XLMRLanguages] = languages
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Model intialised with : {self.device}")
    @abstractmethod
    def getEmbeddings(self, dataset : NLPDataset, batch_size = 8, **kwrds)-> np.ndarray:
        pass

    def generateEmbeddings(self, dataset : NLPDataset, batc_size = 8, **kwrds) -> np.array:
        if dataset.lang not in self.languages:
            raise ValueError(f"Dataset Language ({dataset.lang}) and Model languages mismatch.")
        else:
            return self.getEmbeddings(dataset,batc_size,**kwrds)