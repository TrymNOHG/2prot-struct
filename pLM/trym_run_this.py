
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AutoTokenizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class ProteinLM(nn.Module):
    def __init__(self, model_name="Rostlab/prot_bert"):
        super(ProteinLM, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = output.last_hidden_state
        return embeddings

def t_sne(embeddings):
    X = embeddings.mean(dim=1).cpu().detach().numpy()
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=3000).fit_transform(X)
    return tsne





data = pd.read_csv("data.csv")
sequences = data["input"].apply(lambda seq: " ".join(seq)).tolist()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProteinLM().to(device)

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert")
encoding = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt", max_length=33)

input_ids, attention_mask = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

model.eval()
with torch.no_grad():
    embeddings = model(input_ids=input_ids, attention_mask=attention_mask)

tsne = t_sne(embeddings)

plt.scatter(tsne[:, 0], tsne[:, 1], alpha=0.9)
plt.title("Protein Embeddings using t-SNE")
plt.xlabel("")
plt.ylabel("")
plt.show()

data = pd.read_csv("data.csv")
data["input"] = embeddings.detach().cpu().numpy().tolist()
data.to_csv("embedded_data.csv", index=False)
