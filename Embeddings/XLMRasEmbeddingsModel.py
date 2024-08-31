import torch
from torch.utils.data import DataLoader
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from tqdm import tqdm
import numpy as np
from .EmbeddingsModel import EmbeddingsModel
from Dsets import NLPDataset
from Languages import XLMRLanguages

class XLMRasEmbeddingsModel(EmbeddingsModel):
    def __init__(self, languages: list[XLMRLanguages] ,path : str= "FacebookAI/xlm-roberta-base") -> None:
        super().__init__(languages)
        self.model : XLMRobertaModel = XLMRobertaModel.from_pretrained(path)
        self.tokenizer : XLMRobertaTokenizer = XLMRobertaTokenizer.from_pretrained(path)
        
    
    def tokenize(self,batch,padding=True,truncation=True,return_tensors = "pt", max_len = 512):
        return self.tokenizer(batch,padding=padding, truncation=truncation, return_tensors=return_tensors, max_length=max_len)
    
    def getDataLoader(self, dataset, batch_size = 8):
        dataloader = DataLoader(dataset ,batch_size=batch_size, collate_fn=lambda x: self.tokenize(x))
        return dataloader
    
    def getEmbeddings(self, dataset : NLPDataset, batch_size = 8 ,**kwrds):

        dataloader = self.getDataLoader(dataset, batch_size)
        self.model.to(self.device)
        embeddings = []

        use_pooler = kwrds.get('pooler', False)

        for batch in tqdm(dataloader, desc="Embedding Batches"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Get the model outputs
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                if use_pooler:
                    # Use the pooler output if 'pooler' is True
                    embedding = outputs.pooler_output.squeeze().cpu().numpy()
                else:
                    # Otherwise, use the last hidden state (CLS token)
                    embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
                
            embeddings.append(embedding)

        cls_embeddings = np.concatenate(embeddings, axis = 0)
        return cls_embeddings
