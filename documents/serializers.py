from rest_framework import serializers
from django.contrib.auth.models import User
from .models import (
    Document, DocumentBatch, BatchAnalysis, ProcessingJob,
    DocumentAnalysis, DocumentTag, ProcessingLog, APILog,
)


class DocumentSerializer(serializers.ModelSerializer):
    """Serializer for Document model"""
    
    file_size_mb = serializers.ReadOnlyField()
    processing_job = serializers.SerializerMethodField()
    analysis_results = serializers.SerializerMethodField()
    
    class Meta:
        model = Document
        fields = [
            'id', 'title', 'file', 'file_type', 'file_size', 'file_size_mb',
            'status', 'extracted_text', 'processing_started_at', 
            'processing_completed_at', 'processing_error', 'created_at', 
            'updated_at', 'processing_job', 'analysis_results'
        ]
        read_only_fields = [
            'id', 'file_type', 'file_size', 'file_size_mb', 'status',
            'extracted_text', 'processing_started_at', 'processing_completed_at',
            'processing_error', 'created_at', 'updated_at'
        ]
    
    def get_processing_job(self, obj):
        """Get processing job information"""
        try:
            job = obj.processing_job
            return {
                'id': str(job.id),
                'status': job.status,
                'progress': job.progress,
                'started_at': job.started_at,
                'completed_at': job.completed_at,
                'error_message': job.error_message,
                'text_extraction_completed': job.text_extraction_completed,
                'ai_analysis_completed': job.ai_analysis_completed,
            }
        except ProcessingJob.DoesNotExist:
            return None
    
    def get_analysis_results(self, obj):
        """Get analysis results summary"""
        try:
            analysis = obj.analysis
            return {
                'id': str(analysis.id),
                'completion_percentage': analysis.completion_percentage,
                'is_complete': analysis.is_complete,
                'summary_completed': analysis.summary_completed,
                'key_points_completed': analysis.key_points_completed,
                'sentiment_completed': analysis.sentiment_completed,
                'topics_completed': analysis.topics_completed,
                'created_at': analysis.created_at,
                'updated_at': analysis.updated_at,
            }
        except DocumentAnalysis.DoesNotExist:
            return None
    
    def validate_file(self, value):
        """Validate uploaded file"""
        if not value:
            raise serializers.ValidationError("No file provided")
        
        # Check file size
        if value.size > 10 * 1024 * 1024:  # 10MB
            raise serializers.ValidationError("File size cannot exceed 10MB")
        
        # Check file extension
        allowed_extensions = ['pdf', 'txt', 'docx']
        file_extension = value.name.split('.')[-1].lower()
        
        if file_extension not in allowed_extensions:
            raise serializers.ValidationError(
                f"File type '{file_extension}' not supported. "
                f"Allowed types: {', '.join(allowed_extensions)}"
            )
        
        return value
    
    def create(self, validated_data):
        """Create document with current user"""
        validated_data['user'] = self.context['request'].user
        
        # Set title from filename if not provided
        if not validated_data.get('title'):
            filename = validated_data['file'].name
            validated_data['title'] = filename.rsplit('.', 1)[0]
        
        return super().create(validated_data)


class DocumentListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for document lists"""
    
    file_size_mb = serializers.ReadOnlyField()
    processing_status = serializers.SerializerMethodField()
    
    class Meta:
        model = Document
        fields = [
            'id', 'title', 'file_type', 'file_size', 'file_size_mb',
            'status', 'created_at', 'updated_at', 'processing_status'
        ]
    
    def get_processing_status(self, obj):
        """Get basic processing status"""
        try:
            job = obj.processing_job
            return {
                'status': job.status,
                'progress': job.progress,
            }
        except ProcessingJob.DoesNotExist:
            return None


class DocumentAnalysisSerializer(serializers.ModelSerializer):
    """Serializer for DocumentAnalysis model"""
    
    completion_percentage = serializers.ReadOnlyField()
    is_complete = serializers.ReadOnlyField()
    
    class Meta:
        model = DocumentAnalysis
        fields = [
            'id', 'summary', 'key_points', 'sentiment', 'sentiment_score', 
            'topics', 'model_used', 'total_tokens_used', 'total_processing_time',
            'summary_completed', 'key_points_completed', 'sentiment_completed', 
            'topics_completed', 'completion_percentage', 'is_complete',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


class ProcessingJobSerializer(serializers.ModelSerializer):
    """Serializer for ProcessingJob model"""
    
    document_title = serializers.CharField(source='document.title', read_only=True)
    logs = serializers.SerializerMethodField()
    
    class Meta:
        model = ProcessingJob
        fields = [
            'id', 'document_title', 'status', 'progress', 'started_at',
            'completed_at', 'error_message', 'retry_count',
            'text_extraction_completed', 'ai_analysis_completed',
            'created_at', 'logs'
        ]
        read_only_fields = ['id', 'created_at']
    
    def get_logs(self, obj):
        """Get recent processing logs"""
        logs = obj.logs.order_by('-created_at')[:10]
        return [
            {
                'id': str(log.id),
                'level': log.level,
                'message': log.message,
                'step': log.step,
                'created_at': log.created_at,
            }
            for log in logs
        ]


class ProcessingLogSerializer(serializers.ModelSerializer):
    """Serializer for ProcessingLog model"""
    
    document_title = serializers.CharField(source='document.title', read_only=True)
    
    class Meta:
        model = ProcessingLog
        fields = [
            'id', 'document_title', 'level', 'message', 'step',
            'metadata', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class DocumentTagSerializer(serializers.ModelSerializer):
    """Serializer for DocumentTag model"""
    
    document_count = serializers.SerializerMethodField()
    
    class Meta:
        model = DocumentTag
        fields = ['id', 'name', 'color', 'description', 'created_at', 'document_count']
        read_only_fields = ['id', 'created_at']
    
    def get_document_count(self, obj):
        """Get count of documents with this tag"""
        return obj.documenttagging_set.count()


class DocumentUploadSerializer(serializers.Serializer):
    """Serializer for document upload with processing options"""
    
    file = serializers.FileField()
    title = serializers.CharField(max_length=255, required=False)
    start_processing = serializers.BooleanField(default=True)
    analysis_types = serializers.ListField(
        child=serializers.ChoiceField(choices=[
            'summary', 'key_points', 'sentiment', 'topics', 'entities'
        ]),
        required=False,
        default=['summary', 'key_points', 'sentiment', 'topics']
    )
    
    def validate_file(self, value):
        """Validate uploaded file"""
        if not value:
            raise serializers.ValidationError("No file provided")
        
        # Check file size
        if value.size > 10 * 1024 * 1024:  # 10MB
            raise serializers.ValidationError("File size cannot exceed 10MB")
        
        # Check file extension
        allowed_extensions = ['pdf', 'txt', 'docx']
        file_extension = value.name.split('.')[-1].lower()
        
        if file_extension not in allowed_extensions:
            raise serializers.ValidationError(
                f"File type '{file_extension}' not supported. "
                f"Allowed types: {', '.join(allowed_extensions)}"
            )
        
        return value
    
    def create(self, validated_data):
        """Create document and optionally start processing"""
        from .tasks import process_document
        
        # Extract processing options
        start_processing = validated_data.pop('start_processing', True)
        analysis_types = validated_data.pop('analysis_types', [])
        
        # Set title from filename if not provided
        if not validated_data.get('title'):
            filename = validated_data['file'].name
            validated_data['title'] = filename.rsplit('.', 1)[0]
        
        # Create document
        validated_data['user'] = self.context['request'].user
        document = Document.objects.create(**validated_data)
        
        # Start processing if requested
        if start_processing:
            process_document.delay(str(document.id))
        
        return document


class BatchUploadSerializer(serializers.Serializer):
    """Serializer for uploading multiple files as a single batch."""

    ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx'}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB per file

    files = serializers.ListField(
        child=serializers.FileField(),
        allow_empty=False,
        help_text='One or more files (PDF, DOCX, TXT) to include in the batch.',
    )
    title = serializers.CharField(
        max_length=255,
        required=False,
        default='',
        help_text='Optional human-readable title for the batch.',
    )

    # ── Validation ───────────────────────────────────────────────────────

    def validate_files(self, files):
        """Validate every file in the upload list."""
        if not files:
            raise serializers.ValidationError('At least one file is required.')

        errors = []
        for f in files:
            ext = f.name.rsplit('.', 1)[-1].lower() if '.' in f.name else ''
            if ext not in self.ALLOWED_EXTENSIONS:
                errors.append(
                    f"'{f.name}': unsupported type '{ext}'. "
                    f"Allowed: {', '.join(sorted(self.ALLOWED_EXTENSIONS))}"
                )
            if f.size > self.MAX_FILE_SIZE:
                errors.append(
                    f"'{f.name}': {f.size / (1024*1024):.1f} MB exceeds "
                    f"the {self.MAX_FILE_SIZE / (1024*1024):.0f} MB limit."
                )

        if errors:
            raise serializers.ValidationError(errors)

        return files

    # ── Creation ─────────────────────────────────────────────────────────

    def create(self, validated_data):
        """Create a DocumentBatch with one Document per uploaded file."""
        from .tasks import process_batch

        user = self.context['request'].user
        files = validated_data['files']
        batch_title = validated_data.get('title', '') or f'Batch — {len(files)} file(s)'

        batch = DocumentBatch.objects.create(user=user, title=batch_title)

        documents = []
        for f in files:
            title = f.name.rsplit('.', 1)[0]
            doc = Document.objects.create(
                user=user,
                batch=batch,
                title=title,
                file=f,
            )
            documents.append(doc)

        # Mark the batch as processing and fire the Celery task
        batch.status = 'processing'
        batch.save(update_fields=['status'])
        process_batch.delay(str(batch.id))

        return batch, documents


class DocumentStatsSerializer(serializers.Serializer):
    """Serializer for document statistics"""
    
    total_documents = serializers.IntegerField()
    processing_documents = serializers.IntegerField()
    completed_documents = serializers.IntegerField()
    failed_documents = serializers.IntegerField()
    total_file_size = serializers.IntegerField()
    avg_processing_time = serializers.FloatField()
    recent_uploads = serializers.IntegerField()


class BatchDocumentStatusSerializer(serializers.ModelSerializer):
    """Lightweight per-document status used inside batch status responses."""

    class Meta:
        model = Document
        fields = [
            'id', 'title', 'file_type', 'status',
            'processing_error', 'created_at',
        ]
        read_only_fields = fields


class BatchStatusSerializer(serializers.ModelSerializer):
    """Full status view of a batch — progress, counts, per-doc breakdown."""

    documents = BatchDocumentStatusSerializer(many=True, read_only=True)
    total_documents = serializers.SerializerMethodField()
    documents_processed = serializers.SerializerMethodField()
    progress = serializers.SerializerMethodField()
    failure_message = serializers.SerializerMethodField()
    total_api_cost = serializers.SerializerMethodField()
    total_tokens_used = serializers.SerializerMethodField()

    class Meta:
        model = DocumentBatch
        fields = [
            'id', 'title', 'status', 'progress',
            'retry_count', 'max_retries',
            'total_documents', 'documents_processed',
            'failure_message',
            'total_api_cost', 'total_tokens_used',
            'created_at', 'updated_at',
            'documents',
        ]
        read_only_fields = fields

    # ── computed fields ──────────────────────────────────────────────────

    def get_total_documents(self, obj):
        return obj.documents.count()

    def get_documents_processed(self, obj):
        return obj.documents.filter(status='completed').count()

    def get_progress(self, obj):
        """Return an integer 0–100 representing batch-wide progress."""
        total = obj.documents.count()
        if total == 0:
            return 0
        completed = obj.documents.filter(status='completed').count()
        return round((completed / total) * 100)

    def get_failure_message(self, obj):
        """Aggregate error messages from any failed documents."""
        if obj.status != 'failed':
            return None
        errors = (
            obj.documents
            .filter(status='failed', processing_error__isnull=False)
            .exclude(processing_error='')
            .values_list('title', 'processing_error')
        )
        if not errors:
            return 'Batch failed with no specific error messages.'
        return [
            {'document': title, 'error': err}
            for title, err in errors
        ]

    def get_total_api_cost(self, obj):
        return float(obj.total_api_cost)

    def get_total_tokens_used(self, obj):
        return obj.total_tokens_used


class BatchResultSerializer(serializers.ModelSerializer):
    """Serializer for cross-document analysis results (BatchAnalysis)."""

    batch_id = serializers.UUIDField(source='batch.id', read_only=True)
    batch_status = serializers.CharField(source='batch.status', read_only=True)
    failure_message = serializers.SerializerMethodField()

    class Meta:
        model = BatchAnalysis
        fields = [
            'id', 'batch_id', 'batch_status',
            'combined_summary', 'key_insights', 'contradictions',
            'failure_message',
            'created_at', 'updated_at',
        ]
        read_only_fields = fields

    def get_failure_message(self, obj):
        batch = obj.batch
        if batch.status != 'failed':
            return None
        errors = (
            batch.documents
            .filter(status='failed', processing_error__isnull=False)
            .exclude(processing_error='')
            .values_list('title', 'processing_error')
        )
        if not errors:
            return 'Batch failed with no specific error messages.'
        return [
            {'document': title, 'error': err}
            for title, err in errors
        ]