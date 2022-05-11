from django.db import models
from django.contrib.auth.models import AbstractUser

# Create your models here.
class User(AbstractUser):
    is_management = models.BooleanField(default=False)
    is_branch = models.BooleanField(default=False)
    is_Ho = models.BooleanField(default=False)
    is_corporate = models.BooleanField(default=False)

class Management(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE,primary_key=True)


    def __str__(self):
        return self.user.username


class Branch(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True)
    branch = models.CharField(max_length=200)
    designation = models.CharField(max_length=200)

    def __str__(self):
        return self.user.username


class Ho(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True)
    branch = models.CharField(max_length=200)
    designation = models.CharField(max_length=200)

    def __str__(self):
        return self.user.username


class Corporate(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True)

    def __str__(self):
        return self.user.username