from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.db import transaction
from .models import User,Management,Ho,Corporate,Branch


class ManagementSignUpFrom(UserCreationForm):
    first_name = forms.CharField(required=True)
    last_name = forms.CharField(required=True)
    email = forms.CharField(required=True)

    class Meta(UserCreationForm.Meta):
        model = User

    @transaction.atomic
    def data_save(self):
        user = super().save(commit=False)
        user.first_name = self.cleaned_data.get('first_name')
        user.last_name = self.cleaned_data.get('last_name')
        user.email = self.cleaned_data.get('email')
        user.is_management = True
        user.save()
        management = Management.objects.create(user=user)
        management.save()
        return user

Branch_name =[('PUNE', 'PUNE'),
              ('MUMBAI', 'MUMBAI'),
              ('DELHI','DELHI'),
              ]

class BranchSignUpFrom(UserCreationForm):
    first_name = forms.CharField(required=True)
    last_name = forms.CharField(required=True)
    email = forms.CharField(required=True)
    branch = forms.CharField(widget=forms.Select(choices=Branch_name))

    class Meta(UserCreationForm.Meta):
        model = User

    @transaction.atomic
    def data_save(self):
        user = super().save(commit=False)
        user.first_name = self.cleaned_data.get('first_name')
        user.last_name = self.cleaned_data.get('last_name')
        user.email = self.cleaned_data.get('email')
        user.is_branch = True
        user.save()
        branch = Branch.objects.create(user=user)
        branch.branch = self.cleaned_data.get('branch')
        branch.designation = self.cleaned_data.get('designation')
        branch.save()
        return user


class HoSignUpFrom(UserCreationForm):
    first_name = forms.CharField(required=True)
    last_name = forms.CharField(required=True)
    email = forms.CharField(required=True)
    branch = forms.CharField(widget=forms.Select(choices=Branch_name))
    class Meta(UserCreationForm.Meta):
        model = User

    @transaction.atomic
    def data_save(self):
        user = super().save(commit=False)
        user.first_name = self.cleaned_data.get('first_name')
        user.last_name = self.cleaned_data.get('last_name')
        user.email = self.cleaned_data.get('email')
        user.is_Ho = True
        user.save()
        ho = Ho.objects.create(user=user)
        ho.branch = self.cleaned_data.get('branch')
        ho.designation = self.cleaned_data.get('designation')
        ho.save()
        return user


class CorporateSignUpFrom(UserCreationForm):
    first_name = forms.CharField(required=True)
    last_name = forms.CharField(required=True)
    email = forms.CharField(required=True)

    class Meta(UserCreationForm.Meta):
        model = User

    @transaction.atomic
    def data_save(self):
        user = super().save(commit=False)
        user.first_name = self.cleaned_data.get('first_name')
        user.last_name = self.cleaned_data.get('last_name')
        user.email = self.cleaned_data.get('email')
        user.is_corporate = True
        user.save()
        corporate = Corporate.objects.create(user=user)
        corporate.save()
        return user

class LoginForm(forms.Form):
    username = forms.CharField(
        widget= forms.TextInput(
            attrs={
                "class": "form-control"
            }
        )
    )
    password = forms.CharField(
        widget=forms.PasswordInput(
            attrs={
                "class": "form-control"
            }
        )
    )
