from django import forms
import os

class SequenceForm(forms.Form):
    sequence = forms.CharField(widget=forms.Textarea)
    model_choices = os.listdir("../ml_infer/pickled_models/")
    CHOICES = tuple([(f"Option {i}", model_name.split('.')[0]) for i, model_name in enumerate(model_choices)])
    model_choice = forms.ChoiceField(label="Which model do you want to test?", choices=CHOICES)
    