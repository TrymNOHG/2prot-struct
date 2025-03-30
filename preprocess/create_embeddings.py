import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


class ProteinLM(nn.Module):
    def __init__(self, model_name="Rostlab/prot_bert"):
        super(ProteinLM, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = output.last_hidden_state
        return embeddings


def embed_dataset(dataset_seqs, dataset_structs, shift_left=1, shift_right=-1, output_file="embeddings.csv", batch_size=1000):
    embedding_columns = [f"c{i+1}" for i in range(1024)]
    with open(output_file, "w") as f:
        f.write(",".join(embedding_columns + ["amino_acid", "secondary_structure"]) + "\n")

    for i in range(0, len(dataset_seqs), batch_size):
        batch_seqs = dataset_seqs[i:i+batch_size]
        batch_structs = dataset_structs[i:i+batch_size]
        all_embeddings = []

        for seq, struct in tqdm(zip(batch_seqs, batch_structs), total=len(batch_seqs)):
            with torch.no_grad():
                ids = tokenizer.batch_encode_plus([seq], add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")
                embedding = model(input_ids=ids['input_ids'].to(device), attention_mask=ids['attention_mask'].to(device))
                embedding_np = embedding[0].detach().cpu().numpy()[shift_left:shift_right]

                for embed_vec, aa, sec_struct in zip(embedding_np, seq.split(), struct):
                    all_embeddings.append(list(embed_vec) + [aa, sec_struct])

        df_embeddings = pd.DataFrame(all_embeddings, columns=embedding_columns + ["amino_acid", "secondary_structure"])
        df_embeddings.to_csv(output_file, mode="a", index=False, header=False)

def embed_sequence(dataset_seqs, dataset_structs, shift_left=1, shift_right=-1, batch_size=1000):
    embedding_columns = [f"c{i+1}" for i in range(1024)]

    for i in range(0, len(dataset_seqs), batch_size):
        batch_seqs = dataset_seqs[i:i+batch_size]
        batch_structs = dataset_structs[i:i+batch_size]
        all_embeddings = []

        for seq, struct in tqdm(zip(batch_seqs, batch_structs), total=len(batch_seqs)):
            with torch.no_grad():
                ids = tokenizer.batch_encode_plus([seq], add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")
                embedding = model(input_ids=ids['input_ids'].to(device), attention_mask=ids['attention_mask'].to(device))
                embedding_np = embedding[0].detach().cpu().numpy()[shift_left:shift_right]

                for embed_vec, aa, sec_struct in zip(embedding_np, seq.split(), struct):
                    all_embeddings.append(list(embed_vec) + [aa, sec_struct])

    df_embeddings = pd.DataFrame(all_embeddings, columns=embedding_columns + ["aa", "dssp8"])
    return df_embeddings


tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProteinLM().to(device)

df = pd.read_csv("data.csv", dtype={"input": str, "dssp8": str})

df["input"] = df["input"].str.upper()
df["dssp8"] = df["dssp8"].str.upper()

df = df[df["input"].str.len() <= 200]
print("\nFiltered DataFrame:")
print(df.head())

seq_list = [' '.join(list(seq)) for seq in df["input"]]
struct_list = df["dssp8"].tolist()

embed_dataset(seq_list, struct_list, output_file="embeddings.csv")

df_embeddings = pd.read_csv("embeddings.csv")
print("\nGenerated Embeddings DataFrame:")
print(df_embeddings.head())