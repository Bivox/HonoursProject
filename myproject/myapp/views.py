from django.shortcuts import render
from django.urls.resolvers import ResolverMatch

# Create your views here.
def hello(request):
   return render(request, "hello.html", {'name':'Beril'})

def add(request):

   val1 = int(request.POST['num1'])
   val2 = int(request.POST['num2'])
   res = val1 + val2

   return render(request, 'result.html', {'result':res})

