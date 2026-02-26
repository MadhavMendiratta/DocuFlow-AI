import logging
from datetime import timedelta
from django.utils import timezone
from django.db.models import Sum
from django.core.paginator import Paginator

from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated

from .models import (
    Document, DocumentBatch, BatchAnalysis, ProcessingJob, 
    DocumentAnalysis, ProcessingLog
)
from .serializers import (
    DocumentSerializer, DocumentListSerializer,
    ProcessingJobSerializer, DocumentUploadSerializer,
    DocumentAnalysisSerializer, ProcessingLogSerializer,
    DocumentStatsSerializer, BatchUploadSerializer, 
    BatchStatusSerializer, BatchResultSerializer
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
    
# --- API Helper Views ---

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def document_status(request, document_id):
    """Get current document processing status"""
    try:
        document = Document.objects.get(id=document_id, user=request.user)
        
        status_data = {
            'document_id': str(document.id),
            'status': document.status,
            'title': document.title,
            'created_at': document.created_at,
        }
        
        # Add processing job info if exists
        try:
            job = document.processing_job
            status_data.update({
                'job_status': job.status,
                'progress': job.progress,
                'started_at': job.started_at,
                'completed_at': job.completed_at,
                'error_message': job.error_message,
            })
        except ProcessingJob.DoesNotExist:
            pass
        
        return Response(status_data)
        
    except Document.DoesNotExist:
        return Response({
            'error': 'Document not found'
        }, status=status.HTTP_404_NOT_FOUND)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def health_check(request):
    """Health check endpoint"""
    return Response({
        'status': 'healthy',
        'timestamp': timezone.now(),
        'user': request.user.username,
    })

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def batch_upload(request):
    """Upload multiple files in a single request and create a DocumentBatch."""
    from django.http import QueryDict
    
    data = QueryDict(mutable=True)
    for key in request.data:
        if key != 'files':
            data[key] = request.data[key]
            
    files = request.FILES.getlist('files')
    if not files:
        return Response(
            {'files': ['No files were uploaded.']},
            status=status.HTTP_400_BAD_REQUEST,
        )
    data.setlist('files', files)

    serializer = BatchUploadSerializer(data=data, context={'request': request})
    if serializer.is_valid():
        batch, documents = serializer.save()
        return Response(
            {
                'message': 'Batch uploaded successfully.',
                'batch_id': str(batch.id),
                'batch_title': batch.title,
                'files_uploaded': len(documents),
                'documents': [
                    {
                        'id': str(doc.id),
                        'title': doc.title,
                        'file_type': doc.file_type,
                        'file_size': doc.file_size,
                    }
                    for doc in documents
                ],
            },
            status=status.HTTP_201_CREATED,
        )

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def batch_retry(request, batch_id):
    """Retry a failed batch."""
    try:
        batch = DocumentBatch.objects.get(id=batch_id, user=request.user)
    except DocumentBatch.DoesNotExist:
        return Response(
            {'error': 'Batch not found.'},
            status=status.HTTP_404_NOT_FOUND,
        )

    if batch.status not in ('failed', 'completed'):
        return Response(
            {'error': f'Batch is currently "{batch.status}" and cannot be retried.'},
            status=status.HTTP_409_CONFLICT,
        )

    if batch.retry_count >= batch.max_retries:
        return Response(
            {
                'error': 'Maximum retry limit reached.',
                'retry_count': batch.retry_count,
                'max_retries': batch.max_retries,
            },
            status=status.HTTP_429_TOO_MANY_REQUESTS,
        )

    # Reset batch
    batch.status = 'pending'
    batch.retry_count += 1
    batch.save(update_fields=['status', 'retry_count', 'updated_at'])

    # Reset only failed documents back to uploaded so they are reprocessed
    failed_docs = batch.documents.filter(status='failed')
    reset_count = failed_docs.update(status='uploaded', processing_error='')

    # Re-trigger Celery task
    task_id = None
    try:
        from .tasks import process_batch
        task = process_batch.delay(str(batch.id))
        task_id = task.id
        logger.info(
            f"Batch {batch_id} retry #{batch.retry_count} triggered — "
            f"task {task.id}, {reset_count} doc(s) reset."
        )
    except ImportError:
        pass

    return Response({
        'batch_id': str(batch.id),
        'status': batch.status,
        'retry_count': batch.retry_count,
        'max_retries': batch.max_retries,
        'documents_reset': reset_count,
        'celery_task_id': task_id,
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def batch_cancel(request, batch_id):
    """Cancel a batch that is pending or processing."""
    try:
        batch = DocumentBatch.objects.get(id=batch_id, user=request.user)
    except DocumentBatch.DoesNotExist:
        return Response(
            {'error': 'Batch not found.'},
            status=status.HTTP_404_NOT_FOUND,
        )

    if batch.status not in ('pending', 'processing'):
        return Response(
            {'error': f'Batch is "{batch.status}" and cannot be cancelled.'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Revoke individual document Celery tasks
    cancelled_docs = 0
    for doc in batch.documents.exclude(status='completed'):
        try:
            job = doc.processing_job
            if job.celery_task_id:
                try:
                    from celery import current_app
                    current_app.control.revoke(job.celery_task_id, terminate=True)
                except ImportError:
                    pass
            job.status = 'revoked'
            job.save(update_fields=['status'])
        except Exception:
            pass  # No processing job yet – that's fine
            
        doc.status = 'failed'
        doc.processing_error = 'Cancelled by user'
        doc.save(update_fields=['status', 'processing_error'])
        cancelled_docs += 1

    batch.status = 'failed'
    batch.save(update_fields=['status', 'updated_at'])

    logger.info(f"Batch {batch_id} cancelled by user {request.user.id}")

    return Response({
        'message': 'Batch cancelled successfully.',
        'batch_id': str(batch.id),
        'documents_cancelled': cancelled_docs,
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def batch_status(request, batch_id):
    """Return current processing status for a batch."""
    try:
        batch = DocumentBatch.objects.prefetch_related('documents').get(
            id=batch_id, user=request.user,
        )
    except DocumentBatch.DoesNotExist:
        return Response(
            {'error': 'Batch not found.'},
            status=status.HTTP_404_NOT_FOUND,
        )

    serializer = BatchStatusSerializer(batch)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def batch_result(request, batch_id):
    """Return the cross-document analysis results for a batch."""
    try:
        batch = DocumentBatch.objects.get(id=batch_id, user=request.user)
    except DocumentBatch.DoesNotExist:
        return Response(
            {'error': 'Batch not found.'},
            status=status.HTTP_404_NOT_FOUND,
        )

    try:
        analysis = batch.analysis  # BatchAnalysis (OneToOne)
    except BatchAnalysis.DoesNotExist:
        return Response({
            'batch_id': str(batch.id),
            'batch_status': batch.status,
            'combined_summary': None,
            'key_insights': None,
            'contradictions': None,
            'failure_message': (
                'Analysis has not been generated yet.'
                if batch.status != 'failed'
                else 'Batch processing failed before analysis could run.'
            ),
        })

    serializer = BatchResultSerializer(analysis)
    return Response(serializer.data)