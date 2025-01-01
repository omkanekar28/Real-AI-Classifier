import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


class EmbeddingAddition:
    """
    Class that deals with conversion of raw paragraph text to vector 
    embeddings that also take semantic information into account.
    """
    
    def __init__(self, dataset_path: str, dataset_store_path: str) -> None:
        """
        Loads the model and dataset to which the embeddings 
        column has to be added.
        """
        self.dataset = pd.read_csv(dataset_path)
        self.dataset_store_path = dataset_store_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L12-v1',
            device=self.device
        )
        self.dataset['embedding'] = ''
        print(f"Sentence Transformer Model loaded successfully on {self.device}")

    def compute_embeddings(self) -> None:
        """
        Iterates through each row in the dataset, computes the 
        text embedding and finally stores the finished dataset.
        """
        for index, row in self.dataset.iterrows():
            print(f"Processing row {index+1}/{len(self.dataset)}...")
            embedding = self.model.encode(row['text'])
            self.dataset['embedding'] = str(embedding.tolist())
        print(f"\nEmbeddings have been calculated for all {len(self.dataset)} rows\n")
    
    def save_dataset(self) -> None:
        """
        Saves the dataset after adding the calculated embeddings as a new column.
        """
        self.dataset.to_csv(self.dataset_store_path, index=False)
        print(f"Dataset saved successfully at {self.dataset_store_path}")

if __name__ == '__main__':
    embedding_addition = EmbeddingAddition(
        dataset_path='/home/om/code/Real-AI-Classifier/data/AI_Human_Refined.csv',
        dataset_store_path='dataset.csv'
    )
    embedding_addition.compute_embeddings()
    embedding_addition.save_dataset()
    