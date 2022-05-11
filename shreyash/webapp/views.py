from django.shortcuts import render,redirect

from django.views.generic import CreateView
from .models import User,Management,Ho,Corporate,Branch
from .forms import ManagementSignUpFrom ,HoSignUpFrom,BranchSignUpFrom,CorporateSignUpFrom
from django.contrib.auth import authenticate, login

# Create your views here.
def register(request):
    return render(request, '..webapp/register.html')

def login(request):
    form = LoginForm(request.POST or None)
    msg = None
    if request.method == 'POST':
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None and user.is_admin:
                login(request, user)
                return redirect('adminpage')
            elif user is not None and user.is_customer:
                login(request, user)
                return redirect('customer')
            elif user is not None and user.is_employee:
                login(request, user)
                return redirect('employee')
            else:
                msg = 'invalid credentials'
        else:
            msg = 'error validating form'
    return render(request, 'login.html', {'form': form, 'msg': msg})

class Management_registration(CreateView):
    model = User
    form_class = ManagementSignUpFrom
    template_name = 'Management_registration.html'

    def get_context_data(self, **kwargs):
        kwargs['user_type'] = 'Management'
        return super().get_context_data(**kwargs)

    def form_valid(self, form):
        user = form.save()
        login(self.request, user)
        return redirect('main')

class Branch_registration(CreateView):
    model = User
    form_class = BranchSignUpFrom
    template_name = 'Branch_registration.html'

    def get_context_data(self, **kwargs):
        kwargs['user_type'] = 'Branch'
        return super().get_context_data(**kwargs)

    def form_valid(self, form):
        user = form.save()
        login(self.request, user)
        return redirect('main')

class Ho_registration(CreateView):
    model = User
    form_class = HoSignUpFrom
    template_name = 'Ho_registration.html'

    def get_context_data(self, **kwargs):
        kwargs['user_type'] = 'Ho'
        return super().get_context_data(**kwargs)

    def form_valid(self, form):
        user = form.save()
        login(self.request, user)
        return redirect('main')

class Corporate_registration(CreateView):
    model = User
    form_class = CorporateSignUpFrom
    template_name = 'Corporate_registration.html'

    def get_context_data(self, **kwargs):
        kwargs['user_type'] = 'Corporate'
        return super().get_context_data(**kwargs)

    def form_valid(self, form):
        user = form.save()
        login(self.request, user)
        return redirect('main')

