import os
import uuid
from django.db import models
from django.contrib.auth.models import User
from django.core.validators import FileExtensionValidator
from django.urls import reverse

def document_upload_path(instance, filename):
    """Generate upload path for documents"""
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join('documents', str(instance.user.id), filename)

class DocumentBatch(models.Model):
    """Model for grouping multiple documents into a single analysis session"""

    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='batches')
    title = models.CharField(max_length=255, blank=True, default='')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')

    retry_count = models.PositiveIntegerField(default=0)
    max_retries = models.PositiveIntegerField(default=3)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'Document batches'
        indexes = [
            models.Index(fields=['user', '-created_at']),
            models.Index(fields=['status']),
        ]

    def __str__(self):
        return f"Batch {self.id} ({self.get_status_display()})"

    @property
    def document_count(self):
        return self.documents.count()

    @property
    def all_documents_completed(self):
        """Check whether every document in the batch has finished processing."""
        return (
            self.documents.exists()
            and not self.documents.exclude(status='completed').exists()
        )

    @property
    def total_api_cost(self):
        """Sum of estimated costs across every API call linked to this batch."""
        from django.db.models import Sum
        return self.api_logs.aggregate(total=Sum('cost_estimated'))['total'] or 0

    @property
    def total_tokens_used(self):
        """Sum of tokens used across every API call linked to this batch."""
        from django.db.models import Sum
        return self.api_logs.aggregate(total=Sum('total_tokens'))['total'] or 0


class Document(models.Model):
    """Model for uploaded documents"""
    
    STATUS_CHOICES = [
        ('uploaded', 'Uploaded'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    FILE_TYPE_CHOICES = [
        ('pdf', 'PDF'),
        ('txt', 'Text'),
        ('docx', 'Word Document'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='documents')
    batch = models.ForeignKey(
        DocumentBatch,
        on_delete=models.CASCADE,
        related_name='documents',
        blank=True,
        null=True,
        help_text='The batch / analysis session this document belongs to',
    )
    title = models.CharField(max_length=255)
    file = models.FileField(
        upload_to=document_upload_path,
        validators=[FileExtensionValidator(allowed_extensions=['pdf', 'txt', 'docx'])]
    )
    file_type = models.CharField(max_length=10, choices=FILE_TYPE_CHOICES, blank=True, default='')
    file_size = models.PositiveIntegerField(help_text="File size in bytes", default=0)
    content_hash = models.CharField(
        max_length=64, blank=True, default='', db_index=True,
        help_text='SHA-256 hash of file content for deduplication',
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='uploaded')
    
    # Processing metadata
    extracted_text = models.TextField(blank=True, null=True)
    processing_started_at = models.DateTimeField(blank=True, null=True)
    processing_completed_at = models.DateTimeField(blank=True, null=True)
    processing_error = models.TextField(blank=True, null=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', '-created_at']),
            models.Index(fields=['status']),
        ]
    
    def __str__(self):
        return f"{self.title} ({self.user.username})"

    def get_absolute_url(self):
        return reverse('documents:document_detail', args=[self.id])

    @property
    def file_size_mb(self):
        """Return file size in MB"""
        return round(self.file_size / (1024 * 1024), 2)
    
    def save(self, *args, **kwargs):
        if self.file:
            # Set file type based on extension
            ext = self.file.name.split('.')[-1].lower()
            self.file_type = ext
            
            # Set file size
            if hasattr(self.file, 'size'):
                self.file_size = self.file.size
        
        super().save(*args, **kwargs)


class ProcessingJob(models.Model):
    """Model to track document processing jobs"""
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('started', 'Started'),
        ('progress', 'In Progress'),
        ('success', 'Success'),
        ('failure', 'Failure'),
        ('retry', 'Retry'),
        ('revoked', 'Revoked'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.OneToOneField(Document, on_delete=models.CASCADE, related_name='processing_job')
    celery_task_id = models.CharField(max_length=255, unique=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    progress = models.PositiveIntegerField(default=0, help_text="Progress percentage (0-100)")
    
    # Task metadata
    started_at = models.DateTimeField(blank=True, null=True)
    completed_at = models.DateTimeField(blank=True, null=True)
    error_message = models.TextField(blank=True, null=True)
    retry_count = models.PositiveIntegerField(default=0)
    
    # Processing steps tracking
    text_extraction_completed = models.BooleanField(default=False)
    ai_analysis_completed = models.BooleanField(default=False)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['celery_task_id']),
            models.Index(fields=['status']),
        ]
    
    def __str__(self):
        return f"Processing Job for {self.document.title}"


class DocumentAnalysis(models.Model):
    """Single analysis record per document with all analysis types"""
    
    SENTIMENT_CHOICES = [
        ('positive', 'Positive'),
        ('negative', 'Negative'),
        ('neutral', 'Neutral'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.OneToOneField(Document, on_delete=models.CASCADE, related_name='analysis')
    
    # Analysis Results - Each as separate fields
    summary = models.TextField(blank=True, null=True, help_text="Document summary")
    key_points = models.JSONField(blank=True, null=True, help_text="List of key points")
    sentiment = models.CharField(max_length=20, choices=SENTIMENT_CHOICES, blank=True, null=True)
    sentiment_score = models.FloatField(blank=True, null=True, help_text="Sentiment confidence score (0-1)")
    topics = models.JSONField(blank=True, null=True, help_text="List of identified topics")
    
    # AI model metadata (consolidated)
    model_used = models.CharField(max_length=100, default='gemini-pro')
    total_tokens_used = models.PositiveIntegerField(blank=True, null=True)
    total_processing_time = models.FloatField(blank=True, null=True, help_text="Total processing time in seconds")
    
    # Processing status for each analysis type
    summary_completed = models.BooleanField(default=False)
    key_points_completed = models.BooleanField(default=False)
    sentiment_completed = models.BooleanField(default=False)
    topics_completed = models.BooleanField(default=False)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['document']),
            models.Index(fields=['sentiment']),
        ]
    
    def __str__(self):
        return f"Analysis for {self.document.title}"
    
    @property
    def is_complete(self):
        """Check if all analysis types are completed"""
        return all([
            self.summary_completed,
            self.key_points_completed,
            self.sentiment_completed,
            self.topics_completed,
        ])
    
    @property
    def completion_percentage(self):
        """Get completion percentage of analysis"""
        completed = sum([
            self.summary_completed,
            self.key_points_completed,
            self.sentiment_completed,
            self.topics_completed,
        ])
        return (completed / 4) * 100


class BatchAnalysis(models.Model):
    """Cross-document analysis results for an entire batch"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    batch = models.OneToOneField(
        DocumentBatch,
        on_delete=models.CASCADE,
        related_name='analysis',
    )
    combined_summary = models.TextField(
        blank=True, null=True,
        help_text='AI-generated summary combining all documents in the batch',
    )
    key_insights = models.JSONField(
        blank=True, null=True,
        help_text='Key insights extracted across all documents',
    )
    contradictions = models.JSONField(
        blank=True, null=True,
        help_text='Contradictions or inconsistencies found across documents',
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'Batch analyses'

    def __str__(self):
        return f"Batch Analysis for {self.batch}"


class DocumentTag(models.Model):
    """Model for document tags/categories"""
    
    name = models.CharField(max_length=50, unique=True)
    color = models.CharField(max_length=7, default='#007bff', help_text="Hex color code")
    description = models.TextField(blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['name']
    
    def __str__(self):
        return self.name


class DocumentTagging(models.Model):
    """Many-to-many relationship between documents and tags"""
    
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    tag = models.ForeignKey(DocumentTag, on_delete=models.CASCADE)
    confidence = models.FloatField(default=1.0, help_text="Tag confidence (0-1)")
    auto_generated = models.BooleanField(default=False, help_text="Whether tag was auto-generated by AI")
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['document', 'tag']
        indexes = [
            models.Index(fields=['document']),
            models.Index(fields=['tag']),
        ]
    
    def __str__(self):
        return f"{self.document.title} - {self.tag.name}"


class ProcessingLog(models.Model):
    """Model to log processing steps and events"""
    
    LOG_LEVEL_CHOICES = [
        ('debug', 'Debug'),
        ('info', 'Info'),
        ('warning', 'Warning'),
        ('error', 'Error'),
        ('critical', 'Critical'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='processing_logs')
    processing_job = models.ForeignKey(
        ProcessingJob, 
        on_delete=models.CASCADE, 
        related_name='logs',
        blank=True, 
        null=True
    )
    
    level = models.CharField(max_length=10, choices=LOG_LEVEL_CHOICES, default='info')
    message = models.TextField()
    step = models.CharField(max_length=100, blank=True, help_text="Processing step name")
    metadata = models.JSONField(blank=True, null=True, help_text="Additional log metadata")
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['document', '-created_at']),
            models.Index(fields=['processing_job', '-created_at']),
            models.Index(fields=['level']),
        ]
    
    def __str__(self):
        return f"{self.level.upper()}: {self.message[:50]}..."


class APILog(models.Model):
    """Per-call log of Gemini API usage with token counts and estimated cost."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Link to either a single document, a batch, or both
    document = models.ForeignKey(
        Document, on_delete=models.CASCADE,
        related_name='api_logs', blank=True, null=True,
    )
    batch = models.ForeignKey(
        DocumentBatch, on_delete=models.CASCADE,
        related_name='api_logs', blank=True, null=True,
    )

    # What type of analysis prompt was this?
    analysis_type = models.CharField(
        max_length=50,
        help_text='e.g. summary, key_points, sentiment, topics, '
                  'combined_summary, common_themes, contradictions',
    )
    model_used = models.CharField(max_length=100, default='gemini-1.5-flash')

    # Token counters
    input_tokens = models.PositiveIntegerField(default=0)
    output_tokens = models.PositiveIntegerField(default=0)
    total_tokens = models.PositiveIntegerField(default=0)

    # Cost
    cost_estimated = models.DecimalField(
        max_digits=10, decimal_places=6, default=0,
        help_text='Estimated cost in USD for this single API call',
    )

    # Timing
    response_time = models.FloatField(
        blank=True, null=True,
        help_text='Wall-clock time for the API call in seconds',
    )
    success = models.BooleanField(default=True)
    error_message = models.TextField(blank=True, default='')

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'API Log'
        verbose_name_plural = 'API Logs'
        indexes = [
            models.Index(fields=['document', '-created_at']),
            models.Index(fields=['batch', '-created_at']),
            models.Index(fields=['analysis_type']),
        ]

    def __str__(self):
        target = self.document or self.batch or 'unknown'
        return (
            f"{self.analysis_type} | {self.total_tokens} tokens | "
            f"${self.cost_estimated} | {target}"
        )