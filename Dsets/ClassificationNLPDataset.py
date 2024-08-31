from .NLPDataset import NLPDataset
import pandas as pd
from Languages import XLMRLanguages

class ClassificationNLPDataset(NLPDataset):
    def __init__(self, data: pd.DataFrame, Xnames: list[str], yname: str, lang: XLMRLanguages):
        super().__init__(data, Xnames, yname, lang)
        self.label_dict = None
        self.categorial = False
    
    def yToCategorial(self, lable_dict  = None):
        if not(self.categorial):
            if lable_dict is None:
                y_labels = self.data[self.yname].unique()
                lable_dict = {label: i for i, label in enumerate(y_labels)}
            self.label_dict = lable_dict
            self.data.loc[:, self.yname] = self.data[self.yname].map(self.label_dict)
            self.data[self.yname] = self.data[self.yname].astype(int)
            
            self.categorial = True
        return self.label_dict
    
    def getLabelDict(self) -> dict:
        return self.label_dict
    
    def inspectClasses(self)-> pd.Series:
        class_counts = self.data[self.yname].value_counts()
        return class_counts