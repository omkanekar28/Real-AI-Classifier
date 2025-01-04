from typing import List, Tuple, Dict, Any, Union
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from torch import nn

DATASET_PATH = "/home/om/code/Real-AI-Classifier/data/AI_Human_Refined.csv"
NUMBER_OF_EPOCHS = 3
LEARNING_RATE = 1e-3


class DatasetHandler:
    """
    Handles all the operations related to the dataset of Real/AI text classifier.
    """

    def __init__(self, csv_dataset_path: str, train_split: 0.8) -> None:
        """
        Initialises the X and y variables of the dataset.
        """
        self.dataset = pd.read_csv(csv_dataset_path)[:100]
        self.train_test_validation_split(train_split)
    
    def split_dataframe(self, dataframe: pd.DataFrame, split_value: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the dataframe in two sub-sets based on split_index.
        """
        real_dataframe = dataframe[dataframe['generated'] == 0]
        ai_dataframe = dataframe[dataframe['generated'] == 1]
        
        real_split_index = int(len(real_dataframe) * split_value)
        ai_split_index = int(len(ai_dataframe) * split_value)

        real_dataframe_first = real_dataframe.iloc[:real_split_index, :]
        real_dataframe_second = real_dataframe.iloc[real_split_index:, :]

        ai_dataframe_first = ai_dataframe.iloc[:ai_split_index, :]
        ai_dataframe_second = ai_dataframe.iloc[ai_split_index:, :]

        first_split = pd.concat([real_dataframe_first, ai_dataframe_first])
        second_split = pd.concat([real_dataframe_second, ai_dataframe_second])

        return first_split, second_split
    
    def train_test_validation_split(self, train_split_value: float) -> None:
        """
        Prepares the 3 splits.
        """
        train, test_val_combined = self.split_dataframe(self.dataset, split_value=train_split_value)
        test, val = self.split_dataframe(test_val_combined, split_value=0.5)
        self.X_train, self.y_train = torch.from_numpy(train['text'].values()), torch.from_numpy(train['generated'].values())
        self.X_test, self.y_test = torch.from_numpy(test['text'].values()), torch.from_numpy(test['generated'].values())
        self.X_val, self.y_val = torch.from_numpy(val['text'].values()), torch.from_numpy(val['generated'].values())


class DistilbertModelHandler:
    """
    Handles all the operations related to the distilbert-base-uncased model 
    (and tokenizer) being used for Real/AI text classification.
    """
    
    def __init__(self) -> None:
        """
        Initialises the distilbert-base-uncased model and tokenizer.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(self.device)
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        if torch.cuda.is_available():
            print(f"distilbert-base-uncased model and tokenizer loaded successfully on GPU")
        else:
            print(f"distilbert-base-uncased model and tokenizer loaded successfully on CPU")

    def __call__(self, batch: List[str]) -> torch.Tensor:
        """
        Processes input text and runs it through the model.
        """
        tokenized_inputs = self.tokenize_batch(batch)
        outputs = self.model(**tokenized_inputs)
        print(outputs.last_hidden_state.shape)
        return outputs.last_hidden_state

    def tokenize_batch(self, batch: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Tokenizes the batch of input text.
        """
        return self.tokenizer(
            batch,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
    

class DistilbertWithClassifier(nn.Module):
    """
    distilbert-base-uncased model with a binary-classification head. To be trained 
    for classifying text as real or AI-generated.
    """

    def __init__(self, distilbert_handler: DistilbertModelHandler, num_labels=2) -> None:
        """
        Loads the distilbert-base-uncased model and adds a binary 
        classification head (a linear layer).
        """
        super(DistilbertWithClassifier, self).__init__()
        self.distilbert_handler = distilbert_handler
        self.classifier = nn.Linear(in_features=self.distilbert_handler.model.config.hidden_size, out_features=num_labels)

    def forward(self, batch: List[str]) -> torch.Tensor:
        """Forward pass through the model for binary classification."""
        outputs = self.distilbert_handler(batch)
        cls_embedding = outputs[:, 0, :]    # ONLY TAKING CLS TOKEN
        probability = self.classifier(cls_embedding)
        return probability
    
    def freeze_distilbert_weights(self) -> None:
        """Freezes entire distilbert architecture."""
        for param in self.distilbert_handler.model.parameters():
            param.requires_grad = False


class ClassifierTrainer:
    """
    Handles the training of the classifier.
    """

    def __init__(self, classifier: DistilbertWithClassifier, dataset: DatasetHandler):
        """
        Initialises the classifier, dataset and other important training parameters.
        """
        self.classifier = classifier
        self.number_of_epochs = NUMBER_OF_EPOCHS
        self.learning_rate = LEARNING_RATE
                        
        # OPTIMIZER
        pass

        # LOSS FUNCTION
        pass

    def train_classifier(self):
        """
        """


if __name__ == '__main__':
    dataset_handler = DatasetHandler(DATASET_PATH)
    distilbert_model_handler = DistilbertModelHandler()
    classifier = DistilbertWithClassifier(distilbert_model_handler)
    classifier.freeze_distilbert_weights()
    








