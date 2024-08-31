from typing import TypeVar, Generic
from .NLPDataset import NLPDataset

T = TypeVar('T', bound=NLPDataset)
class SplitDataset(Generic[T]):
    def __init__(self, train_data: T, test_data: T, val_data: T, lang : str) -> None:
        self.train_data: T = train_data
        self.test_data: T = test_data
        self.val_data: T = val_data
        self.lang: str = lang
    
    def getTrainData(self) -> T:
        return self.train_data
    
    def getTestData(self) -> T:
        return self.test_data
    
    def getValData(self) -> T:
        return self.val_data