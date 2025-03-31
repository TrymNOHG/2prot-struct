import torch
import numpy as np
import torch.nn.functional as F
import pickle
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ml_infer')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml_infer", "models")))
from preprocess.create_embeddings import embed_sequence
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "./ml_infer/")))
# from ml_infer.models.stochastic_model import StochasticModel
# from ml_infer.models.naive_bayes_model import NaiveBayesModel
# from ml_infer import models
# from ml_infer.models import *



def infer(model_name, secondary_struct, sequences):
    print(sys.path)
    with open(f'../ml_infer/pickled_models/{model_name}', 'rb') as f:
        if '.pt' in model_name:
            model = torch.jit.load(f=f, map_location=torch.device('cpu'))
        else:
            model = pickle.load(f)
    model.eval()

    sequences = sequences.split("\n")
    sequences = [' '.join(list(seq)) for seq in sequences]
    embedding_df = embed_sequence(sequences)

    pca_model = pickle.load(open("../preprocess/dimred_data/95_pca.pkl", 'rb'))
    vals = pca_model.transform(embedding_df)

    tensors = [torch.tensor([x], dtype=torch.float32) for x in vals]

    predictions = []
    with torch.no_grad():
        for tensor in tensors:
            pred = model.forward(tensor)[0]
            predictions.append(list(F.softmax(pred, dim=-1)))
    pred_data = []
    max_vals = []
    output_data = []
    for pred in predictions:
        output_index = np.argmax(pred)
        output_data.append(secondary_struct[output_index])
        max_vals.append(float(pred[output_index]))
        pred_data.append({secondary_struct[i]: float(pred[i]) for i in range(len(pred))})
    output_data = "".join(output_data)
    return pred_data, output_data, max_vals
