from .EmbeddingsModel import EmbeddingsModel
from .RobertaAsEmbeddingsModel import RoBERTaAsEmbeddingsModel
from Dsets import NLPDataset


class MutliRoberta(EmbeddingsModel):
    def __init__(self) -> None:
        super().__init__([])
        self.models : dict[str,RoBERTaAsEmbeddingsModel] = dict()
        self.languages : list[str] = list()
    
    def addModel(self,model : RoBERTaAsEmbeddingsModel):
        lang = model.getLang().value
        self.models[lang] = model
        self.languages.append(lang)
    
    def getEmbeddings(self, dataset: NLPDataset, batch_size = 8, **kwrds):
        if dataset.lang.value in self.languages:
            return self.models[dataset.lang.value].getEmbeddings(dataset, batch_size, **kwrds)