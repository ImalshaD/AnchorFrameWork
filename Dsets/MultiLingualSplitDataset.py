from typing import TypeVar, Generic
from Languages import XLMRLanguages
from .SplitDataset import SplitDataset

G = TypeVar("G", bound=SplitDataset)
class MultilingualSplitDataset(Generic(G)):
    def __init__(self) -> None:
        self.trainingDatasets : dict[XLMRLanguages, G] = dict()
        self.testingDatasets : dict[XLMRLanguages, G] = dict()
        self.trainingLangs : list[XLMRLanguages]  = list()
        self.testingLangs : list[XLMRLanguages]  = list()

    def addTraingSet(self, dataset : G ):
        lang = dataset.lang
        self.trainingDatasets[lang] = dataset
        if lang not in self.trainingLangs:
            self.trainingLangs.append(lang)

    def addTestingSet(self, dataset: G):
        lang = dataset.lang
        self.testingDatasets[lang] = dataset
        if lang not in self.testingLangs:
            self.testingLangs.append(lang)
    def getTrainingSet(self, lang: XLMRLanguages) -> G:
        if lang in self.trainingDatasets:
            return self.trainingDatasets[lang]
        else:
            raise ValueError(f"No training dataset found for language: {lang}")
    def getTestingSet(self, lang: XLMRLanguages) -> G:
        if lang in self.testingDatasets:
            return self.testingDatasets[lang]
        else:
            raise ValueError(f"No testing dataset found for language: {lang}")
    
    def getTrainingLangs(self) -> list[XLMRLanguages]:
        return self.trainingLangs
    
    def getTestingLangs(self) -> list[XLMRLanguages]:
        return self.testingLangs
