import torch
from transformers import BertTokenizer, BertForSequenceClassification

class IntentClassifier:
    def __init__(self, model_path):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model.eval()

    def predict_intent(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        intents = ["order_issue", "small_talk", "delivery_change"]
        return intents[predicted_class]

# Test the trained model
intent_classifier = IntentClassifier("C:/Users/umesh.malviya1/Downloads/intent_classifier")

# Test the model with a sample conversation
conversation = "Hi,i want to change my delivery address.?"
predicted_intent = intent_classifier.predict_intent(conversation)
print("Predicted Intent:", predicted_intent)
