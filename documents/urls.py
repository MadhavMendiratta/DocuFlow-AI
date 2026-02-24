from django.urls import path
from django.shortcuts import redirect
from . import views

app_name = 'documents'

def redirect_to_login(request):
    """Redirect unauthenticated users to admin login"""
    if request.user.is_authenticated:
        return views.dashboard(request)
    else:
        return redirect('admin:login')

urlpatterns = [
    # Web Interface URLs
    path('', redirect_to_login, name='home'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('documents/', views.document_list, name='document_list'),
    path('documents/<uuid:document_id>/', views.document_detail, name='document_detail'),
    path('upload/', views.upload_document, name='upload_document'),
]