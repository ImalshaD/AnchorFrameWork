from .ClassificationNLPDataset import ClassificationNLPDataset
from .SplitDataset import SplitDataset
from Languages import XLMRLanguages
import pandas as pd

class ClassificationSplitDataset(SplitDataset[ClassificationNLPDataset]):
    def __init__(self, train_data: ClassificationNLPDataset, test_data: ClassificationNLPDataset, val_data: ClassificationNLPDataset, lang : XLMRLanguages) -> None:
        super().__init__(train_data, test_data, val_data, lang)
        self.lable_dict = None
        self.embeddings = None

    def yToCategorial(self, lable_dict  = None):
        lable_dict = self.train_data.yToCategorial(lable_dict)
        self.test_data.yToCategorial(lable_dict)
        self.val_data.yToCategorial(lable_dict)
        self.lable_dict = lable_dict
        return self.lable_dict

    def inspectAllClasses(self) -> pd.DataFrame:
        # Get class distributions
        train_classes = self.train_data.inspectClasses()
        test_classes = self.test_data.inspectClasses()
        val_classes = self.val_data.inspectClasses()
        
        # Combine into a DataFrame
        class_df = pd.DataFrame({
            'train': train_classes,
            'test': test_classes,
            'val': val_classes
        })
        
        # Fill NaN values with 0 (in case a class is missing in one of the splits)
        class_df = class_df.fillna(0).astype(int)
        
        return class_df
    
    def getLableDict(self) -> dict:
        return self.lable_dict
    
    @staticmethod
    def fromNLPClassification(dataset: ClassificationNLPDataset, test_ratio : float, val_ratio : float) -> 'ClassificationSplitDataset':
        train, test, val = dataset.getSplits(test_ratio, val_ratio)
        return ClassificationSplitDataset.from_dfs(train,test,val, dataset.Xnames, dataset.yname, dataset.lang)
    
    @staticmethod
    def from_df(data: pd.DataFrame, Xnames: list[str], yname: str, lang: str, test_ratio: float, val_ratio: float) -> 'ClassificationSplitDataset':
        # Create a ClassificationNLPDataset instance
        ClassificationSplitDataset.printMsg("Passed Dataframe", len(data))
        dataset = ClassificationNLPDataset(data, Xnames, yname, lang)
        
        return ClassificationSplitDataset.fromNLPClassification(dataset,test_ratio,val_ratio)
    
    @staticmethod
    def from_dfs(train_data: pd.DataFrame, test_data: pd.DataFrame, val_data: pd.DataFrame, Xnames: list[str], yname: str, lang: str) -> 'ClassificationSplitDataset':
        # Create ClassificationNLPDataset instances for each split
        ClassificationSplitDataset.printMsg("Train Dataset", len(train_data))
        train_dataset = ClassificationNLPDataset(train_data, Xnames, yname, lang)
        ClassificationSplitDataset.printMsg("Test Dataset", len(test_data))
        test_dataset = ClassificationNLPDataset(test_data, Xnames, yname, lang)
        ClassificationSplitDataset.printMsg("Validation Dataset", len(val_data))
        val_dataset = ClassificationNLPDataset(val_data, Xnames, yname, lang)
        
        # Create and return the ClassificationSplitDataset instance
        return ClassificationSplitDataset(train_dataset, test_dataset, val_dataset, lang)
    @staticmethod
    def printMsg(datasetDesription, num_rows):
        print(f" Processing : {datasetDesription} rows: {num_rows}")