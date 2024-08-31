from abc import ABC
from typing import Any
from Dsets import SplitDataset
from Embeddings import EmbeddingsModel
from TrainableModel import TrainableModel
from Dsets import NLPDataset
from Embeddings import Embeddings
from tqdm import tqdm


class NLPTask(ABC):
    def __init__(self, dataset : SplitDataset, embeddingModel : EmbeddingsModel, trainableModel) -> None:
        self.dataset : SplitDataset = dataset
        self.embeddingsModel : EmbeddingsModel = embeddingModel
        self.trainableModel : TrainableModel = trainableModel
        self.train = self.__getEmbeddings(self.dataset.getTrainData())
        self.test = self.__getEmbeddings(self.dataset.getTestData())
        self.val = self.__getEmbeddings(self.dataset.getValData())

    def __getEmbeddings(self,dataset: NLPDataset,):
        return Embeddings.generateEmbeddings(dataset,self.embeddingsModel)
    
    def runTask(self,epochs):
        trainingModel = self.trainableModel.getCopy()
        trainingModel.compile()
        trainingModel.trainModel(self.train.getDataLoader(),self.val.getDataLoader(),epochs)
        return self.trainableModel.evalModel(self.test.getDataLoader())
    
    def experiment(self,seeds=1,epochs =10):
        avg_acc, avg_f1, avg_loss = 0,0,0
        for seed in tqdm(range(seeds), desc="Running Seeds"):
            loss, acc, f1 = self.runTask(epochs)
            avg_acc += acc/seeds
            avg_f1 += f1/seeds
            avg_loss += loss/seeds
        return avg_loss,avg_acc,avg_f1
