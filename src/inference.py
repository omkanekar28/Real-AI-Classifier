import torch
from transformers import DistilBertTokenizer, DistilBertModel
from torch import nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "/home/om/code/Real-AI-Classifier/models/RealAIClassifier-05-1-2025.pt"
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


class DistilbertModelHandler:
    def __init__(self):
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(DEVICE)
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        
    def __call__(self, batch):
        tokenized_inputs = self.tokenize_batch(batch)
        outputs = self.model(**tokenized_inputs)
        return outputs.last_hidden_state
        
    def tokenize_batch(self, batch):
        return self.tokenizer(
            batch,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(DEVICE)


class DistilbertWithClassifier(nn.Module):
    def __init__(self, distilbert_handler, num_labels=2):
        super(DistilbertWithClassifier, self).__init__()
        self.distilbert_handler = distilbert_handler
        self.classifier = nn.Linear(
            in_features=self.distilbert_handler.model.config.hidden_size, 
            out_features=num_labels, 
            device=DEVICE
        )
        
    def forward(self, batch):
        outputs = self.distilbert_handler(batch)
        cls_embedding = outputs[:, 0, :]
        probability = self.classifier(cls_embedding)
        return probability


def load_classifier(model_path):
    classifier = torch.load(model_path)
    classifier.eval()
    return classifier

def predict_text(classifier, text):
    with torch.no_grad():
        logits = classifier([text])
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        return {
            'prediction': 'AI-generated' if prediction.item() == 1 else 'Human-written',
            'confidence': probabilities[0][prediction.item()].item()
        }

if __name__ == '__main__':
    classifier = load_classifier(MODEL_PATH)
    sample_text = "The rain fell in gentle sheets, creating a rhythmic patter against the windowpane. Inside, the cozy room was filled with the scent of burning wood from the fireplace and the soft glow of lamplight. A steaming cup of tea sat on the table, untouched, as she lost herself in the pages of a book. Each word seemed to transport her to another world, far removed from the mundane realities of everyday life. In that moment, she found solace in the quiet magic of solitude."
    result = predict_text(classifier, sample_text)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")