from django.http import HttpResponse
from django.shortcuts import render 
from .seq_form import SequenceForm


def sequence(request):
    context = {} 
    form = SequenceForm()
    # form = GeeksForm(request.POST or None) 
    context['form'] = form 
    return render(request, "prot_struct/seq.html", context) 
