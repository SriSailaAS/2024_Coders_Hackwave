from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=6)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load('checkpoint.pth', map_location=torch.device('cpu'))
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
model.load_state_dict(checkpoint['model_state_dict'])
print("analyzing")

inp="saila is a dumb"

input_ids = tokenizer.encode(inp, add_special_tokens=True, return_tensors="pt")
with torch.no_grad():
    model.eval()
    input_ids = input_ids.to(device) 
    outputs = model(input_ids)
logits = outputs.logits
threshold = 0.5 
predictions = (torch.sigmoid(logits) > threshold).cpu().numpy()

predicted_labels = [label for label, prediction in zip(labels, predictions[0]) if prediction]
output_string = f"Input Text: '{inp}'\nPredicted Labels: {', '.join(predicted_labels)}"
print(output_string)