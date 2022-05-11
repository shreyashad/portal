from django.contrib import admin
from webapp.models import User,Management,Ho,Branch,Corporate
# Register your models here.

admin.site.register(User)
admin.site.register(Management)
admin.site.register(Ho)
admin.site.register(Branch)
admin.site.register(Corporate)