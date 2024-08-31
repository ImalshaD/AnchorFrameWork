from .NLPTask import NLPTask
from Dsets import ClassificationSplitDataset
from Embeddings import EmbeddingsModel
from TrainableModel import TrainableClassificationModel


class ClassificationNLPTask(NLPTask):

    def __init__(self, dataset: ClassificationSplitDataset, embeddingModel: EmbeddingsModel, trainableModel: TrainableClassificationModel) -> None:
        super().__init__(dataset, embeddingModel, trainableModel)