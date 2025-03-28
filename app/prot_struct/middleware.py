# myapp/middleware.py
from django.utils.deprecation import MiddlewareMixin
from django.shortcuts import render
from .custom_exceptions import InvalidSequenceException

class GlobalExceptionMiddleware(MiddlewareMixin):
    def process_exception(self, request, exception):
        if isinstance(exception, InvalidSequenceException):
            return render(request, 'prot_struct/invalid_seq.html', status=404)
        else:
            return None
