from django.shortcuts import render

from myapp.models import Destination

# Create your views here.
def index(request):

   dest1 = Destination()
   dest1.price = 400

   dest2 = Destination()
   dest2.price = 620

   dests = [dest1, dest2]
   return render(request, "index.html", {'dests': dests})