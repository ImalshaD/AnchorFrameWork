import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from Languages import XLMRLanguages

class NLPDataset:
    def __init__(self, data : pd.DataFrame, Xnames : list[str], yname : str ,lang : XLMRLanguages):
        features = Xnames + [yname]
        self.data : pd.DataFrame = data[features]
        self.lang : XLMRLanguages = lang
        self.Xnames : str = Xnames
        self.yname : str = yname
        self.drop_duplicates_and_print()
        self.drop_na_and_print()
    
    def getData(self)-> pd.DataFrame:
        return self.data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # TODO: implement support for multiple columns
        return list(self.data[self.Xnames].iloc[index])[0]
    
    def getY(self):
        return self.data[self.yname].values
    
    def getYasTensor(self):
        return torch.tensor(self.getY(), dtype=torch.long)
    
    def getSplits(self,test_ratio : float, val_ratio : float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Split the dataset into train and remaining data
        train_data, test_val_data = train_test_split(self.data, test_size=(test_ratio + val_ratio), random_state=42)
        
        # Split the remaining data into test and validation sets
        test_data, val_data = train_test_split(test_val_data, test_size=(val_ratio / (test_ratio + val_ratio)), random_state=42)

        train_data = shuffle(train_data, random_state=42)
        
        return train_data, test_data, val_data
    
    def drop_duplicates_and_print(self):
        initial_count = len(self.data)
        self.data = self.data.drop_duplicates()
        dropped_count = initial_count - len(self.data)
        self.__printMsg("Dropped", dropped_count, "Duplicate Rows")

    def drop_na_and_print(self):
        initial_count = len(self.data)
        self.data = self.data.dropna()
        dropped_count = initial_count - len(self.data)
        self.__printMsg("Dropped", dropped_count, "Missing values")
    def __printMsg(self , action , count , reason):
        if (count):
            print(f" {action} {count} (rows) reason: {reason}.")
    def getLang(self):
        return self.lang