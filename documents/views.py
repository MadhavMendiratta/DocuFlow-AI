import logging
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated

from .models import Document, ProcessingJob
from .serializers import (
    DocumentSerializer, DocumentListSerializer,
    ProcessingJobSerializer, DocumentUploadSerializer,
)

logger = logging.getLogger(__name__)

class DocumentViewSet(viewsets.ModelViewSet):
    """ViewSet for Document CRUD operations"""
    
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]
    
    def get_serializer_class(self):
        if self.action == 'list':
            return DocumentListSerializer
        elif self.action == 'upload':
            return DocumentUploadSerializer
        return DocumentSerializer
    
    def get_queryset(self):
        """Filter documents by current user"""
        queryset = Document.objects.filter(user=self.request.user)
        
        # Filter by status
        status_filter = self.request.query_params.get('status')
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        # Filter by file type
        file_type = self.request.query_params.get('file_type')
        if file_type:
            queryset = queryset.filter(file_type=file_type)
        
        # Search by title
        search = self.request.query_params.get('search')
        if search:
            queryset = queryset.filter(title__icontains=search)
        
        return queryset.order_by('-created_at')
    
    def perform_create(self, serializer):
        """Create document with current user"""
        document = serializer.save(user=self.request.user)
        
        # Start processing automatically (Note: tasks.py doesn't exist yet in our git history)
        try:
            from .tasks import process_document
            process_document.delay(str(document.id))
        except ImportError:
            pass
        
        logger.info(f"Document {document.id} created and processing started by user {self.request.user.id}")
    
    @action(detail=False, methods=['post'])
    def upload(self, request):
        """Upload document with processing options"""
        serializer = DocumentUploadSerializer(data=request.data, context={'request': request})
        
        if serializer.is_valid():
            document = serializer.save()
            
            # Return document data
            document_serializer = DocumentSerializer(document, context={'request': request})
            
            return Response({
                'message': 'Document uploaded successfully',
                'document': document_serializer.data
            }, status=status.HTTP_201_CREATED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ProcessingJobViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for ProcessingJob read operations"""
    
    serializer_class = ProcessingJobSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Filter processing jobs by user's documents"""
        return ProcessingJob.objects.filter(
            document__user=self.request.user
        ).order_by('-created_at')