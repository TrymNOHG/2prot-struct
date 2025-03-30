from django.http import HttpResponse, JsonResponse
from django.shortcuts import render 
from .seq_form import SequenceForm
import re
from requests import ConnectionError
from .custom_exceptions import InvalidSequenceException
from .utils import infer
def sequence(request):
    secondary_struct = list("HECTGSPIB")

    if request.method == "POST":
        form = SequenceForm(request.POST)
        if form.is_valid():
            sequences = form.cleaned_data['sequence']
            model_name = form.cleaned_data['model_choice']
        else:
            raise InvalidSequenceException("Something went wrong")
        illegal_sequence = re.search("[^a-zA-Z]", sequences) 
        if illegal_sequence or len(sequences) == 0:
            raise InvalidSequenceException("Invalid Sequence") # Add better error handling
        else:
            # Give message the inference may take a little while
            pred_data, output_data, max_vals = infer(model_name, secondary_struct, sequences)
            

    else:
        form = SequenceForm()
        pred_data = [{secondary_struct[i]: 0 for i in range(len(secondary_struct))}]
        max_vals = [1]
        sequences = ''
        output_data = ''
    context = {} 
    context['form'] = form 
    context['hist_data'] = pred_data
    context['max_vals'] = max_vals
    context['output_data'] = output_data
    context['sequence'] = sequences
    return render(request, "prot_struct/seq.html", context) 


def handle_invalid_sequence(request, exception=None):
    return render(request, 'errors/403_custom.html', status=403)

