from django.http import HttpResponse, JsonResponse
from django.shortcuts import render 
from .seq_form import SequenceForm
import re
import pickle
from requests import ConnectionError
from .custom_exceptions import InvalidSequenceException
import torch

def sequence(request):
    secondary_struc = list("HECTGSPIB")

    if request.method == "POST":
        form = SequenceForm(request.POST)
        if form.is_valid():
            sequence = form.cleaned_data['sequence']
            model_name = form.cleaned_data['model_choice']
        else:
            raise InvalidSequenceException("Something went wrong")
        illegal_sequence = re.search("[^a-zA-Z]", sequence) 
        if illegal_sequence or len(sequence) == 0:
            raise InvalidSequenceException("Invalid Sequence") # Add better error handling
        else:
            # Give message the inference may take a little while
            # Re-direct to results page:
            print(f'../ml_infer/pickled_models/{model_name}')
            with open(f'../ml_infer/pickled_models/{model_name}', 'rb') as f:
                model = torch.load(f=f, weights_only=False, map_location=torch.device('cpu'))
            # Create embeddings
            t = torch.Tensor(sequence)
            prediction = model.forward(t)
            # prediction = [0.1, 0.2, 0.1, 0.4, 0.05, 0.05, 0.05, 0.05, 0.05]
            max_val = max(prediction) 
            pred_data = {secondary_struc[i]: prediction[i] for i in range(len(prediction))}

    else:
        form = SequenceForm()
        pred_data = {secondary_struc[i]: 0 for i in range(len(secondary_struc))}
        max_val = 1
    context = {} 
    context['form'] = form 
    context['hist_data'] = pred_data
    context['max_val'] = max_val
    return render(request, "prot_struct/seq.html", context) 


def handle_invalid_sequence(request, exception=None):
    return render(request, 'errors/403_custom.html', status=403)

