import logging
from datetime import timedelta
from django.utils import timezone
from django.db.models import Sum
from django.core.paginator import Paginator

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated

from .models import Document, ProcessingJob, DocumentAnalysis, ProcessingLog
from .serializers import (
    DocumentSerializer, DocumentListSerializer,
    ProcessingJobSerializer, DocumentUploadSerializer,
    DocumentAnalysisSerializer, ProcessingLogSerializer,
    DocumentStatsSerializer
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
        
        # Start processing automatically
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
    
    @action(detail=True, methods=['post'])
    def reprocess(self, request, pk=None):
        """Restart processing for a document"""
        document = self.get_object()
        
        if document.status == 'processing':
            return Response({
                'error': 'Document is already being processed'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Reset document status
        document.status = 'uploaded'
        document.processing_error = None
        document.extracted_text = None
        document.save()
        
        # Clear existing analysis results
        try:
            document.analysis.delete()
        except DocumentAnalysis.DoesNotExist:
            pass
        
        # Start processing
        try:
            from .tasks import process_document
            process_document.delay(str(document.id))
        except ImportError:
            pass
        
        logger.info(f"Document {document.id} reprocessing started by user {request.user.id}")
        
        return Response({
            'message': 'Document reprocessing started'
        }, status=status.HTTP_200_OK)
    
    @action(detail=True, methods=['post'])
    def cancel_processing(self, request, pk=None):
        """Cancel document processing"""
        document = self.get_object()
        
        try:
            processing_job = document.processing_job
            
            if processing_job.status not in ['pending', 'started', 'progress']:
                return Response({
                    'error': 'Cannot cancel processing - job is not active'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Cancel Celery task
            try:
                from celery import current_app
                current_app.control.revoke(processing_job.celery_task_id, terminate=True)
            except ImportError:
                pass
            
            # Update status
            processing_job.status = 'revoked'
            processing_job.save()
            
            document.status = 'failed'
            document.processing_error = 'Processing cancelled by user'
            document.save()
            
            logger.info(f"Document {document.id} processing cancelled by user {request.user.id}")
            
            return Response({
                'message': 'Processing cancelled successfully'
            }, status=status.HTTP_200_OK)
            
        except ProcessingJob.DoesNotExist:
            return Response({
                'error': 'No active processing job found'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Error cancelling processing for document {document.id}: {e}")
            return Response({
                'error': 'Failed to cancel processing'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=True, methods=['get'])
    def analysis(self, request, pk=None):
        """Get detailed analysis results for a document"""
        document = self.get_object()
        
        try:
            analysis = document.analysis
            serializer = DocumentAnalysisSerializer(analysis)
            
            return Response({
                'document_id': str(document.id),
                'document_title': document.title,
                'analysis': serializer.data
            })
        except DocumentAnalysis.DoesNotExist:
            return Response({
                'document_id': str(document.id),
                'document_title': document.title,
                'analysis': None,
                'message': 'No analysis available for this document'
            })
    
    @action(detail=True, methods=['get'])
    def logs(self, request, pk=None):
        """Get processing logs for a document"""
        document = self.get_object()
        
        logs = document.processing_logs.order_by('-created_at')
        
        # Pagination
        page_size = int(request.query_params.get('page_size', 20))
        paginator = Paginator(logs, page_size)
        page_number = request.query_params.get('page', 1)
        page_obj = paginator.get_page(page_number)
        
        serializer = ProcessingLogSerializer(page_obj, many=True)
        
        return Response({
            'document_id': str(document.id),
            'logs': serializer.data,
            'pagination': {
                'page': page_obj.number,
                'pages': paginator.num_pages,
                'has_next': page_obj.has_next(),
                'has_previous': page_obj.has_previous(),
                'total': paginator.count
            }
        })
    
    @action(detail=False, methods=['get'])
    def stats(self, request):
        """Get document statistics for current user"""
        user_documents = Document.objects.filter(user=request.user)
        
        # Basic counts
        total_documents = user_documents.count()
        processing_documents = user_documents.filter(status='processing').count()
        completed_documents = user_documents.filter(status='completed').count()
        failed_documents = user_documents.filter(status='failed').count()
        
        # File size statistics
        total_file_size = user_documents.aggregate(
            total_size=Sum('file_size')
        )['total_size'] or 0
        
        # Processing time statistics
        completed_jobs = ProcessingJob.objects.filter(
            document__user=request.user,
            status='success',
            completed_at__isnull=False,
            started_at__isnull=False
        )
        
        avg_processing_time = 0
        if completed_jobs.exists():
            # Calculate average processing time in seconds
            processing_times = []
            for job in completed_jobs:
                if job.completed_at and job.started_at:
                    delta = job.completed_at - job.started_at
                    processing_times.append(delta.total_seconds())
            
            if processing_times:
                avg_processing_time = sum(processing_times) / len(processing_times)
        
        # Recent uploads (last 7 days)
        week_ago = timezone.now() - timedelta(days=7)
        recent_uploads = user_documents.filter(created_at__gte=week_ago).count()
        
        stats_data = {
            'total_documents': total_documents,
            'processing_documents': processing_documents,
            'completed_documents': completed_documents,
            'failed_documents': failed_documents,
            'total_file_size': total_file_size,
            'avg_processing_time': avg_processing_time,
            'recent_uploads': recent_uploads,
        }
        
        serializer = DocumentStatsSerializer(stats_data)
        return Response(serializer.data)


class ProcessingJobViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for ProcessingJob read operations"""
    
    serializer_class = ProcessingJobSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Filter processing jobs by user's documents"""
        return ProcessingJob.objects.filter(
            document__user=self.request.user
        ).order_by('-created_at')
    

