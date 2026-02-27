import logging
from datetime import timedelta
from django.utils import timezone
from django.db.models import Count, Sum, Avg, Q
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, HttpResponse
from django.urls import reverse
from django.views.decorators.http import require_http_methods
from django.core.paginator import Paginator
from django.contrib import messages

from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated

from .models import Document, DocumentBatch, BatchAnalysis, ProcessingJob, DocumentAnalysis, ProcessingLog, APILog
from .serializers import (
    DocumentSerializer, DocumentListSerializer, DocumentAnalysisSerializer,
    ProcessingJobSerializer, DocumentUploadSerializer,
    DocumentStatsSerializer, ProcessingLogSerializer,
    BatchUploadSerializer, BatchStatusSerializer, BatchResultSerializer,
)
from .tasks import process_document, process_batch

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
        process_document.delay(str(document.id))
        
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
        process_document.delay(str(document.id))
        
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
            from celery import current_app
            current_app.control.revoke(processing_job.celery_task_id, terminate=True)
            
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


# Web Interface Views (HTML Templates)

@login_required
def dashboard(request):
    """Main dashboard view"""
    user_documents = Document.objects.filter(user=request.user)
    user_batches = DocumentBatch.objects.filter(user=request.user).prefetch_related('documents')

    # Get recent documents
    recent_documents = user_documents.select_related('batch').order_by('-created_at')[:5]

    # Get statistics
    stats = {
        'total_documents': user_documents.count(),
        'processing_documents': user_documents.filter(status='processing').count(),
        'completed_documents': user_documents.filter(status='completed').count(),
        'failed_documents': user_documents.filter(status='failed').count(),
    }

    # Recent batches with progress computation
    recent_batches_qs = user_batches.order_by('-created_at')[:5]
    recent_batches = []
    for batch in recent_batches_qs:
        total = batch.documents.count()
        completed = batch.documents.filter(status='completed').count()
        progress = round((completed / total) * 100) if total > 0 else 0
        recent_batches.append({
            'batch': batch,
            'total_docs': total,
            'completed_docs': completed,
            'progress': progress,
            'cost': float(batch.total_api_cost),
        })

    # Cost tracking aggregates
    cost_data = APILog.objects.filter(
        Q(document__user=request.user) | Q(batch__user=request.user)
    ).aggregate(
        total_tokens=Sum('total_tokens'),
        total_cost=Sum('cost_estimated'),
    )

    context = {
        'recent_documents': recent_documents,
        'recent_batches': recent_batches,
        'stats': stats,
        'total_tokens': cost_data['total_tokens'] or 0,
        'total_cost': cost_data['total_cost'] or 0,
    }

    return render(request, 'documents/dashboard.html', context)


@login_required
def document_list(request):
    """Document list view with filtering and pagination"""
    documents = Document.objects.filter(user=request.user).select_related('batch')
    
    # Apply filters
    status_filter = request.GET.get('status')
    if status_filter:
        documents = documents.filter(status=status_filter)
    
    file_type_filter = request.GET.get('file_type')
    if file_type_filter:
        documents = documents.filter(file_type=file_type_filter)
    
    search_query = request.GET.get('search')
    if search_query:
        documents = documents.filter(title__icontains=search_query)
    
    documents = documents.order_by('-created_at')
    
    # Pagination
    paginator = Paginator(documents, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'status_filter': status_filter,
        'file_type_filter': file_type_filter,
        'search_query': search_query,
    }
    
    return render(request, 'documents/document_list.html', context)


@login_required
def document_detail(request, document_id):
    """Document detail view"""
    document = get_object_or_404(Document, id=document_id, user=request.user)
    
    # Get analysis results
    try:
        analysis = document.analysis
    except DocumentAnalysis.DoesNotExist:
        analysis = None
    
    # Get processing logs
    processing_logs = document.processing_logs.order_by('-created_at')[:20]
    
    context = {
        'document': document,
        'analysis': analysis,
        'processing_logs': processing_logs,
    }
    
    return render(request, 'documents/document_detail.html', context)


@login_required
def batch_list(request):
    """Paginated list of all batches for the current user."""
    batches_qs = (
        DocumentBatch.objects
        .filter(user=request.user)
        .prefetch_related('documents')
        .order_by('-created_at')
    )

    # Build enriched list with progress / cost
    enriched = []
    for batch in batches_qs:
        total = batch.documents.count()
        completed = batch.documents.filter(status='completed').count()
        progress = round((completed / total) * 100) if total > 0 else 0
        enriched.append({
            'batch': batch,
            'total_docs': total,
            'completed_docs': completed,
            'progress': progress,
            'cost': float(batch.total_api_cost),
            'tokens': batch.total_tokens_used,
        })

    paginator = Paginator(enriched, 10)
    page_obj = paginator.get_page(request.GET.get('page'))

    return render(request, 'documents/batch_list.html', {'page_obj': page_obj})


@login_required
def batch_detail_view(request, batch_id):
    """Batch detail web view with analysis results + cost tracking."""
    batch = get_object_or_404(DocumentBatch, id=batch_id, user=request.user)
    documents = batch.documents.order_by('-created_at')

    try:
        analysis = batch.analysis
    except BatchAnalysis.DoesNotExist:
        analysis = None

    api_logs = batch.api_logs.order_by('-created_at')[:20]

    total_docs = documents.count()
    completed_docs = documents.filter(status='completed').count()
    progress = round((completed_docs / total_docs) * 100) if total_docs > 0 else 0

    context = {
        'batch': batch,
        'documents': documents,
        'analysis': analysis,
        'api_logs': api_logs,
        'progress': progress,
        'total_docs': total_docs,
        'completed_docs': completed_docs,
    }
    return render(request, 'documents/batch_detail.html', context)


@login_required
def upload_document(request):
    """Document upload view — supports both standard form POST and AJAX."""
    if request.method == 'POST':
        uploaded_file = request.FILES.get('file')
        title = request.POST.get('title', '')
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

        if not uploaded_file:
            if is_ajax:
                return JsonResponse(
                    {'status': 'error', 'message': 'Please select a file to upload.'},
                    status=400,
                )
            messages.error(request, 'Please select a file to upload.')
            return render(request, 'documents/upload.html')

        if not title:
            title = uploaded_file.name.rsplit('.', 1)[0]

        document = Document.objects.create(
            user=request.user,
            title=title,
            file=uploaded_file,
        )

        # Trigger Celery processing
        process_document.delay(str(document.id))

        if is_ajax:
            return JsonResponse({
                'status': 'success',
                'document_id': str(document.id),
                'redirect_url': reverse(
                    'documents:document_detail', args=[document.id],
                ),
                'message': f'Document "{title}" uploaded successfully.',
            })

        messages.success(
            request,
            f'Document "{title}" uploaded successfully and processing started.',
        )
        return redirect('documents:document_detail', document_id=document.id)

    return render(request, 'documents/upload.html')


# API Helper Views

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


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def batch_upload(request):
    """Upload multiple files in a single request and create a DocumentBatch.

    POST /api/batch/upload/
    Content-Type: multipart/form-data

    Fields:
        files   – one or more files (PDF, DOCX, TXT)   [required]
        title   – optional batch title                  [optional]

    Returns 201 with batch_id and file count.
    """
    # DRF's ListField expects the key "files", but multipart uploads send
    # each file under the same key.  We normalise here so the serializer
    # receives a proper list.
    # NOTE: We avoid request.data.copy() because deep-copying file handles
    # (BufferedRandom) raises TypeError on Python 3.14+.
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
    """Retry a failed batch.

    POST /api/batch/<uuid>/retry/

    Resets batch and its failed documents back to pending, increments
    retry_count, and re-triggers the Celery process_batch task.
    """
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
    task = process_batch.delay(str(batch.id))

    logger.info(
        f"Batch {batch_id} retry #{batch.retry_count} triggered — "
        f"task {task.id}, {reset_count} doc(s) reset."
    )

    return Response({
        'batch_id': str(batch.id),
        'status': batch.status,
        'retry_count': batch.retry_count,
        'max_retries': batch.max_retries,
        'documents_reset': reset_count,
        'celery_task_id': task.id,
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def batch_cancel(request, batch_id):
    """Cancel a batch that is pending or processing.

    POST /api/batch/<uuid>/cancel/

    Revokes the Celery task (if any), marks the batch as failed, and
    marks all non-completed documents as failed with a cancellation note.
    """
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
    from celery import current_app
    cancelled_docs = 0
    for doc in batch.documents.exclude(status='completed'):
        try:
            job = doc.processing_job
            if job.celery_task_id:
                current_app.control.revoke(job.celery_task_id, terminate=True)
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
    """Return current processing status for a batch.

    GET /api/batch/<uuid>/status/

    Response includes:
        - status (pending / processing / completed / failed)
        - progress  (0–100 %)
        - documents_processed / total_documents
        - per-document breakdown
        - failure_message (if applicable)
        - total_api_cost / total_tokens_used
    """
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
    """Return the cross-document analysis results for a batch.

    GET /api/batch/<uuid>/result/

    Response includes:
        - combined_summary
        - key_insights
        - contradictions
        - failure_message (if the batch failed)
    """
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
        # Batch exists but cross-document analysis hasn’t run yet
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


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def health_check(request):
    """Health check endpoint"""
    return Response({
        'status': 'healthy',
        'timestamp': timezone.now(),
        'user': request.user.username,
    })
