from django.http import HttpResponse
from django.shortcuts import render 
from .seq_form import SequenceForm
import re
import pickle
from requests import ConnectionError
from .custom_exceptions import InvalidSequenceException


def sequence(request):
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
            model = pickle.load(f"../ml_infer/pickled_models/{model_name}.pkl")
            prediction = model.predict(sequence)
            # Use the softmax predictions to show the distribution.
            pass
    else:
        context = {} 
        form = SequenceForm()
        context['form'] = form 
        return render(request, "prot_struct/seq.html", context) 
    

def handle_invalid_sequence(request, exception=None):
    return render(request, 'errors/403_custom.html', status=403)

