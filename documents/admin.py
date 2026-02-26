"""
Django admin configuration for the documents app.

Includes admin actions for reprocessing, cancellation, and status management.
"""

from django.contrib import admin
from django.utils.html import format_html
from django.utils.safestring import mark_safe
from django.urls import reverse
from django.contrib import messages

from .models import (
    Document, ProcessingJob, DocumentAnalysis,
    DocumentTag, DocumentTagging, ProcessingLog,
)
from .tasks import process_document


# ── Admin Actions ────────────────────────────────────────────────────────────


def reprocess_documents(modeladmin, request, queryset):
    """Re-trigger processing for selected documents."""
    count = 0
    for doc in queryset.exclude(status='processing'):
        doc.status = 'uploaded'
        doc.processing_error = None
        doc.extracted_text = None
        doc.save(update_fields=['status', 'processing_error', 'extracted_text'])

        DocumentAnalysis.objects.filter(document=doc).delete()
        process_document.delay(str(doc.id))
        count += 1

    messages.success(request, f"Reprocessing started for {count} document(s).")


reprocess_documents.short_description = "Reprocess selected documents"


def mark_as_failed(modeladmin, request, queryset):
    """Mark selected documents as failed."""
    updated = queryset.exclude(status='failed').update(
        status='failed',
        processing_error='Manually marked as failed by admin',
    )
    messages.warning(request, f"Marked {updated} document(s) as failed.")


mark_as_failed.short_description = "Mark selected as failed"


def cancel_processing_jobs(modeladmin, request, queryset):
    """Cancel selected processing jobs."""
    from celery import current_app

    count = 0
    for job in queryset.filter(status__in=['pending', 'started', 'progress']):
        try:
            current_app.control.revoke(job.celery_task_id, terminate=True)
            job.status = 'revoked'
            job.save(update_fields=['status'])

            job.document.status = 'failed'
            job.document.processing_error = 'Cancelled by admin'
            job.document.save(update_fields=['status', 'processing_error'])
            count += 1
        except Exception:
            pass

    messages.success(request, f"Cancelled {count} processing job(s).")


cancel_processing_jobs.short_description = "Cancel selected processing jobs"


# ── Model Admins ─────────────────────────────────────────────────────────────


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = [
        'title', 'user', 'file_type', 'status_badge',
        'file_size_display', 'created_at', 'processing_actions',
    ]
    list_filter = ['status', 'file_type', 'created_at', 'user']
    search_fields = ['title', 'user__username', 'user__email']
    readonly_fields = [
        'id', 'file_type', 'file_size', 'extracted_text',
        'processing_started_at', 'processing_completed_at',
        'created_at', 'updated_at',
    ]
    actions = [reprocess_documents, mark_as_failed]

    fieldsets = (
        ('Basic Information', {
            'fields': ('id', 'user', 'title', 'file', 'file_type', 'file_size'),
        }),
        ('Processing Status', {
            'fields': (
                'status', 'processing_started_at',
                'processing_completed_at', 'processing_error',
            ),
        }),
        ('Extracted Content', {
            'fields': ('extracted_text',),
            'classes': ('collapse',),
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',),
        }),
    )

    def file_size_display(self, obj):
        if obj.file_size:
            return f"{obj.file_size_mb} MB"
        return "N/A"

    file_size_display.short_description = "File Size"

    def status_badge(self, obj):
        colours = {
            'uploaded': '#3b82f6',
            'processing': '#f59e0b',
            'completed': '#10b981',
            'failed': '#ef4444',
        }
        colour = colours.get(obj.status, '#6b7280')
        return format_html(
            '<span style="background:{}; color:#fff; padding:3px 10px; '
            'border-radius:12px; font-size:11px;">{}</span>',
            colour,
            obj.get_status_display(),
        )

    status_badge.short_description = "Status"

    def processing_actions(self, obj):
        if obj.status == 'processing':
            return mark_safe(
                '<span style="color:#f59e0b;">&#9203; Processing…</span>'
            )
        if obj.status == 'completed':
            try:
                analysis = obj.analysis
                url = reverse(
                    'admin:documents_documentanalysis_change', args=[analysis.id],
                )
                return format_html('<a href="{}">View Analysis</a>', url)
            except DocumentAnalysis.DoesNotExist:
                return mark_safe('<span style="color:#9ca3af;">No analysis</span>')
        if obj.status == 'failed':
            return mark_safe(
                '<span style="color:#ef4444;">&#10007; Failed</span>'
            )
        return mark_safe('<span style="color:#9ca3af;">&mdash;</span>')

    processing_actions.short_description = "Actions"

    def save_model(self, request, obj, form, change):
        """Auto-trigger processing when a document is created via admin."""
        super().save_model(request, obj, form, change)
        if not change:  # New document
            process_document.delay(str(obj.id))
            messages.info(request, f'Processing started for "{obj.title}".')

    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user')


@admin.register(ProcessingJob)
class ProcessingJobAdmin(admin.ModelAdmin):
    list_display = [
        'document_title', 'status', 'progress',
        'started_at', 'completed_at', 'retry_count',
    ]
    list_filter = ['status', 'created_at']
    search_fields = ['document__title', 'document__user__username']
    readonly_fields = ['id', 'celery_task_id', 'created_at', 'updated_at']
    actions = [cancel_processing_jobs]

    fieldsets = (
        ('Job', {
            'fields': ('id', 'document', 'celery_task_id', 'status', 'progress'),
        }),
        ('Timing', {
            'fields': ('started_at', 'completed_at', 'retry_count'),
        }),
        ('Steps', {
            'fields': ('text_extraction_completed', 'ai_analysis_completed'),
        }),
        ('Errors', {
            'fields': ('error_message',),
            'classes': ('collapse',),
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',),
        }),
    )

    def document_title(self, obj):
        url = reverse('admin:documents_document_change', args=[obj.document.id])
        return format_html('<a href="{}">{}</a>', url, obj.document.title)

    document_title.short_description = "Document"

    def get_queryset(self, request):
        return super().get_queryset(request).select_related(
            'document', 'document__user',
        )


@admin.register(DocumentAnalysis)
class DocumentAnalysisAdmin(admin.ModelAdmin):
    list_display = [
        'document_title', 'completion_display', 'sentiment',
        'model_used', 'total_processing_time', 'created_at',
    ]
    list_filter = ['model_used', 'sentiment', 'created_at']
    search_fields = ['document__title']
    readonly_fields = ['id', 'created_at', 'updated_at']

    fieldsets = (
        ('Info', {
            'fields': ('id', 'document', 'model_used'),
        }),
        ('Results', {
            'fields': (
                'summary', 'key_points', 'sentiment',
                'sentiment_score', 'topics',
            ),
        }),
        ('Completion', {
            'fields': (
                'summary_completed', 'key_points_completed',
                'sentiment_completed', 'topics_completed',
            ),
        }),
        ('Metrics', {
            'fields': ('total_tokens_used', 'total_processing_time'),
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',),
        }),
    )

    def document_title(self, obj):
        url = reverse('admin:documents_document_change', args=[obj.document.id])
        return format_html('<a href="{}">{}</a>', url, obj.document.title)

    document_title.short_description = "Document"

    def completion_display(self, obj):
        pct = obj.completion_percentage
        if pct == 100:
            return format_html(
                '<span style="color:#10b981;">&#10003; {}%</span>', int(pct),
            )
        if pct > 0:
            return format_html(
                '<span style="color:#f59e0b;">&#9203; {}%</span>', int(pct),
            )
        return mark_safe('<span style="color:#9ca3af;">0%</span>')

    completion_display.short_description = "Completion"

    def get_queryset(self, request):
        return super().get_queryset(request).select_related(
            'document', 'document__user',
        )


@admin.register(DocumentTag)
class DocumentTagAdmin(admin.ModelAdmin):
    list_display = ['name', 'color_display', 'document_count', 'created_at']
    search_fields = ['name', 'description']
    readonly_fields = ['created_at']

    def color_display(self, obj):
        return format_html(
            '<span style="display:inline-block;width:16px;height:16px;'
            'background:{};border-radius:3px;border:1px solid #d1d5db;"></span>',
            obj.color,
        )

    color_display.short_description = "Color"

    def document_count(self, obj):
        return obj.documenttagging_set.count()

    document_count.short_description = "Documents"


@admin.register(DocumentTagging)
class DocumentTaggingAdmin(admin.ModelAdmin):
    list_display = [
        'document_title', 'tag', 'confidence', 'auto_generated', 'created_at',
    ]
    list_filter = ['auto_generated', 'tag']
    search_fields = ['document__title', 'tag__name']

    def document_title(self, obj):
        url = reverse('admin:documents_document_change', args=[obj.document.id])
        return format_html('<a href="{}">{}</a>', url, obj.document.title)

    document_title.short_description = "Document"


@admin.register(ProcessingLog)
class ProcessingLogAdmin(admin.ModelAdmin):
    list_display = [
        'document_title', 'level_badge', 'step', 'message_preview', 'created_at',
    ]
    list_filter = ['level', 'step', 'created_at']
    search_fields = ['document__title', 'message']
    readonly_fields = ['id', 'created_at']

    fieldsets = (
        ('Log', {
            'fields': ('id', 'document', 'processing_job', 'level', 'step'),
        }),
        ('Content', {
            'fields': ('message',),
        }),
        ('Metadata', {
            'fields': ('metadata',),
            'classes': ('collapse',),
        }),
        ('Timestamp', {
            'fields': ('created_at',),
        }),
    )

    def document_title(self, obj):
        url = reverse('admin:documents_document_change', args=[obj.document.id])
        return format_html('<a href="{}">{}</a>', url, obj.document.title)

    document_title.short_description = "Document"

    def level_badge(self, obj):
        colours = {
            'debug': '#9ca3af',
            'info': '#3b82f6',
            'warning': '#f59e0b',
            'error': '#ef4444',
            'critical': '#dc2626',
        }
        colour = colours.get(obj.level, '#6b7280')
        return format_html(
            '<span style="color:{}; font-weight:600;">{}</span>',
            colour,
            obj.level.upper(),
        )

    level_badge.short_description = "Level"

    def message_preview(self, obj):
        if len(obj.message) > 80:
            return obj.message[:80] + '…'
        return obj.message

    message_preview.short_description = "Message"

    def get_queryset(self, request):
        return super().get_queryset(request).select_related(
            'document', 'processing_job',
        )


# ── Site branding ────────────────────────────────────────────────────────────

admin.site.site_header = "AI Document Processor"
admin.site.site_title = "AI Document Processor Admin"
admin.site.index_title = "Administration"