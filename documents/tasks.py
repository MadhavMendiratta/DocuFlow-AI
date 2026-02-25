import json
import time
import uuid
import logging
from typing import Dict, Any

from celery import shared_task
from django.conf import settings
from django.utils import timezone
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

from .models import Document, ProcessingJob, ProcessingLog, DocumentAnalysis

logger = logging.getLogger(__name__)


# ── WebSocket Helpers ────────────────────────────────────────────────────────

def send_websocket_update(document_id: str, event_type: str, data: Dict[str, Any]):
    """Send real-time update to clients watching this document."""
    try:
        channel_layer = get_channel_layer()
        if channel_layer:
            async_to_sync(channel_layer.group_send)(
                f'document_{document_id}',
                {
                    'type': event_type,
                    'document_id': str(document_id),
                    'timestamp': timezone.now().isoformat(),
                    **data,
                },
            )
    except Exception as e:
        logger.error(f"WebSocket send failed: {e}")


def log_processing_step(
    document_id: str,
    processing_job_id: str,
    level: str,
    message: str,
    step: str = '',
    progress: int = 0,
    metadata: Dict = None,
):
    """Write a processing log row AND push a WebSocket notification."""
    try:
        ProcessingLog.objects.create(
            document_id=document_id,
            processing_job_id=processing_job_id,
            level=level,
            message=message,
            step=step,
            metadata=metadata or {},
        )
        send_websocket_update(str(document_id), 'processing_update', {
            'status': 'processing',
            'progress': progress,
            'message': message,
            'step': step,
        })
    except Exception as e:
        logger.error(f"Error logging processing step: {e}")


# ── Main Processing Task ────────────────────────────────────────────────────

@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def process_document(self, document_id: str) -> Dict[str, Any]:
    """
    Main Celery task — processes a document through the full pipeline:
    1. Extract text from file
    2. Analyse with Gemini AI
    3. Store results
    4. Notify via WebSocket
    """
    try:
        document = Document.objects.get(id=document_id)
    except Document.DoesNotExist:
        logger.error(f"Document {document_id} not found — aborting task")
        return {'status': 'error', 'message': 'Document not found'}

    # Create / update processing job
    task_id = self.request.id or str(uuid.uuid4())
    processing_job, created = ProcessingJob.objects.get_or_create(
        document=document,
        defaults={
            'celery_task_id': task_id,
            'status': 'started',
            'started_at': timezone.now(),
        },
    )
    if not created:
        processing_job.celery_task_id = task_id
        processing_job.status = 'started'
        processing_job.started_at = timezone.now()
        processing_job.error_message = None
        processing_job.save(update_fields=[
            'celery_task_id', 'status', 'started_at', 'error_message',
        ])

    # Mark document as processing
    document.status = 'processing'
    document.processing_started_at = timezone.now()
    document.processing_error = None
    document.save(update_fields=['status', 'processing_started_at', 'processing_error'])

    log_processing_step(
        str(document_id), str(processing_job.id), 'info',
        f'Starting processing for "{document.title}"',
        'initialization', progress=10,
    )

    try:
        # ── Step 1: Text Extraction ──────────────────────────────────────
        if self.request.id:
            self.update_state(state='PROGRESS', meta={'progress': 20, 'step': 'text_extraction'})
        processing_job.progress = 20
        processing_job.save(update_fields=['progress'])

        extracted_text = extract_text_from_document(document)
        if not extracted_text or not extracted_text.strip():
            raise ValueError("No text could be extracted from the document")

        document.extracted_text = extracted_text
        document.save(update_fields=['extracted_text'])

        processing_job.text_extraction_completed = True
        processing_job.progress = 50
        processing_job.save(update_fields=['text_extraction_completed', 'progress'])

        log_processing_step(
            str(document_id), str(processing_job.id), 'info',
            f'Text extracted — {len(extracted_text):,} characters.',
            'text_extraction', progress=50,
        )

        # ── Step 2: AI Analysis ──────────────────────────────────────────
        if self.request.id:
            self.update_state(state='PROGRESS', meta={'progress': 60, 'step': 'ai_analysis'})

        analysis_results = analyze_document_with_gemini(document, extracted_text)

        processing_job.ai_analysis_completed = True
        processing_job.progress = 90
        processing_job.save(update_fields=['ai_analysis_completed', 'progress'])

        log_processing_step(
            str(document_id), str(processing_job.id), 'info',
            'AI analysis completed.',
            'ai_analysis', progress=90,
        )

        # ── Step 3: Finalise ─────────────────────────────────────────────
        if self.request.id:
            self.update_state(state='PROGRESS', meta={'progress': 100, 'step': 'finalization'})

        document.status = 'completed'
        document.processing_completed_at = timezone.now()
        document.save(update_fields=['status', 'processing_completed_at'])

        processing_job.status = 'success'
        processing_job.progress = 100
        processing_job.completed_at = timezone.now()
        processing_job.save(update_fields=['status', 'progress', 'completed_at'])

        log_processing_step(
            str(document_id), str(processing_job.id), 'info',
            'Processing completed successfully.',
            'finalization', progress=100,
        )

        send_websocket_update(str(document_id), 'processing_complete', {
            'status': 'completed',
            'results': {
                'text_length': len(extracted_text),
                'analysis_complete': analysis_results.get('completion_percentage', 0) == 100,
            },
        })

        return {
            'status': 'success',
            'document_id': str(document_id),
            'text_length': len(extracted_text),
        }

    except Exception as exc:
        logger.error(f"Error processing document {document_id}: {exc}", exc_info=True)

        will_retry = self.request.retries < self.max_retries

        try:
            if will_retry:
                processing_job.status = 'retry'
                processing_job.retry_count = (processing_job.retry_count or 0) + 1
                processing_job.save(update_fields=['status', 'retry_count'])
            else:
                document.status = 'failed'
                document.processing_error = str(exc)
                document.save(update_fields=['status', 'processing_error'])

                processing_job.status = 'failure'
                processing_job.error_message = str(exc)
                processing_job.save(update_fields=['status', 'error_message'])

                send_websocket_update(str(document_id), 'processing_error', {
                    'status': 'failed',
                    'error': str(exc),
                })

            log_processing_step(
                str(document_id), str(processing_job.id), 'error',
                f'Processing failed: {exc}',
                'error',
            )
        except Exception as inner:
            logger.error(f"Failed to update status for document {document_id}: {inner}")

        if will_retry:
            raise self.retry(countdown=60 * (2 ** self.request.retries), exc=exc)

        raise


# ── Text Extraction ──────────────────────────────────────────────────────────

def extract_text_from_document(document: Document) -> str:
    """Route to the correct extractor based on file type."""
    file_path = document.file.path
    file_type = document.file_type.lower()

    extractors = {
        'pdf': extract_text_from_pdf,
        'txt': extract_text_from_txt,
        'docx': extract_text_from_docx,
    }

    extractor = extractors.get(file_type)
    if not extractor:
        raise ValueError(f"Unsupported file type: {file_type}")

    return extractor(file_path)

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF using pdfplumber with PyPDF2 fallback."""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"--- Page {i} ---\n{page_text}")
                except Exception as e:
                    logger.warning(f"pdfplumber: page {i} error: {e}")
        if text_parts:
            return '\n\n'.join(text_parts)
    except Exception as e:
        logger.warning(f"pdfplumber failed, falling back to PyPDF2: {e}")

    import PyPDF2
    text_parts = []
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages, 1):
            try:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"--- Page {i} ---\n{page_text}")
            except Exception as e:
                logger.warning(f"PyPDF2: page {i} error: {e}")

    return '\n\n'.join(text_parts)

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from a plain-text file (tries multiple encodings)."""
    for encoding in ('utf-8', 'latin-1', 'cp1252'):
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode text file with any known encoding")

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a .docx file."""
    from docx import Document as DocxDocument
    doc = DocxDocument(file_path)
    return '\n'.join(p.text for p in doc.paragraphs if p.text.strip())


# ── AI Analysis ──────────────────────────────────────────────────────────────

MAX_PROMPT_CHARS = 4000

def _configure_genai():
    """Lazily configure and return the google.generativeai module."""
    import google.generativeai as genai
    if settings.GEMINI_API_KEY:
        genai.configure(api_key=settings.GEMINI_API_KEY)
    return genai

def _extract_token_counts(response) -> Dict[str, int]:
    """Pull input/output/total token counts from a Gemini response."""
    counts = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
    try:
        meta = getattr(response, 'usage_metadata', None)
        if meta:
            counts['input_tokens'] = getattr(meta, 'prompt_token_count', 0) or 0
            counts['output_tokens'] = getattr(meta, 'candidates_token_count', 0) or 0
            counts['total_tokens'] = getattr(meta, 'total_token_count', 0) or 0
            if not counts['total_tokens']:
                counts['total_tokens'] = counts['input_tokens'] + counts['output_tokens']
    except Exception:
        pass
    return counts

def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost based on per-token pricing from settings."""
    cost_in = input_tokens * getattr(settings, 'GEMINI_COST_PER_INPUT_TOKEN', 0.000000075)
    cost_out = output_tokens * getattr(settings, 'GEMINI_COST_PER_OUTPUT_TOKEN', 0.0000003)
    return round(cost_in + cost_out, 6)

GEMINI_CALL_DELAY = getattr(settings, 'GEMINI_CALL_DELAY', 13)

def _gemini_generate(model, prompt, genai, *, max_retries: int = 3):
    """Call model.generate_content with automatic retry on 429 rate-limit."""
    import re as _re
    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=settings.GEMINI_MAX_TOKENS,
                    temperature=settings.GEMINI_TEMPERATURE,
                ),
            )
            return response
        except Exception as exc:
            err_str = str(exc)
            if '429' in err_str and attempt < max_retries:
                match = _re.search(r'retry in ([\d.]+)s', err_str, _re.IGNORECASE)
                wait = float(match.group(1)) + 2 if match else 15 * (2 ** (attempt - 1))
                logger.warning(
                    f"Gemini 429 rate-limit hit (attempt {attempt}/{max_retries}). "
                    f"Waiting {wait:.1f}s before retry…"
                )
                time.sleep(wait)
            else:
                raise

def analyze_document_with_gemini(document: Document, text_content: str) -> dict:
    """Run all analysis types via Gemini AI and persist to DocumentAnalysis."""
    genai = _configure_genai()

    if not settings.GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not configured — skipping AI analysis")
        analysis, _ = DocumentAnalysis.objects.get_or_create(
            document=document, defaults={'model_used': 'none'},
        )
        return {'analysis_id': str(analysis.id), 'completion_percentage': 0}

    model = genai.GenerativeModel(settings.GEMINI_MODEL)
    truncated = text_content[:MAX_PROMPT_CHARS]

    analysis, _ = DocumentAnalysis.objects.get_or_create(
        document=document, defaults={'model_used': settings.GEMINI_MODEL},
    )

    prompts = {
        'summary': (
            "Please provide a concise summary of the following document "
            "in 2-3 sentences:\n\n" + truncated
        ),
        'key_points': (
            "Extract the key points from the following document. "
            "Respond with a JSON array of strings:\n\n" + truncated
        ),
        'sentiment': (
            "Analyze the sentiment of the following document. "
            "Respond with exactly one word: positive, negative, or neutral.\n\n"
            + truncated
        ),
        'topics': (
            "Identify the main topics discussed in the following document. "
            "Respond with a JSON array of short topic strings:\n\n" + truncated
        ),
    }

    total_time = 0.0
    total_tokens = 0

    for analysis_type, prompt in prompts.items():
        try:
            start = time.time()
            response = _gemini_generate(model, prompt, genai)
            elapsed = time.time() - start
            total_time += elapsed

            if not response.text:
                logger.warning(f"{analysis_type}: empty Gemini response")
                continue

            result = response.text.strip()
            counts = _extract_token_counts(response)
            total_tokens += counts['total_tokens']

            if analysis_type == 'summary':
                analysis.summary = result
                analysis.summary_completed = True
            elif analysis_type == 'key_points':
                analysis.key_points = _parse_json_list(result)
                analysis.key_points_completed = True
            elif analysis_type == 'sentiment':
                sentiment = result.lower().strip().rstrip('.')
                if sentiment in ('positive', 'negative', 'neutral'):
                    analysis.sentiment = sentiment
                    analysis.sentiment_score = 0.85
                elif 'positive' in sentiment:
                    analysis.sentiment = 'positive'
                    analysis.sentiment_score = 0.6
                elif 'negative' in sentiment:
                    analysis.sentiment = 'negative'
                    analysis.sentiment_score = 0.6
                else:
                    analysis.sentiment = 'neutral'
                    analysis.sentiment_score = 0.5
                analysis.sentiment_completed = True
            elif analysis_type == 'topics':
                analysis.topics = _parse_json_list(result)
                analysis.topics_completed = True

            logger.info(f"{analysis_type} completed in {elapsed:.2f}s")
            time.sleep(GEMINI_CALL_DELAY)

        except Exception as e:
            logger.error(f"Error in {analysis_type} analysis: {e}", exc_info=True)
            continue

    analysis.total_processing_time = total_time
    analysis.total_tokens_used = total_tokens or None
    analysis.save()

    return {
        'analysis_id': str(analysis.id),
        'completion_percentage': analysis.completion_percentage,
        'total_processing_time': total_time,
    }

def _parse_json_list(text: str) -> list:
    """Parse text as a JSON list with robust fallbacks."""
    cleaned = text.strip()

    if cleaned.startswith('```'):
        lines = cleaned.split('\n')
        lines = [ln for ln in lines if not ln.strip().startswith('```')]
        cleaned = '\n'.join(lines).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (json.JSONDecodeError, ValueError):
        pass

    items = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        for prefix in ('-', '•', '*', '–'):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                break
        if line and line[0].isdigit():
            parts = line.split('.', 1)
            if len(parts) == 2 and parts[0].strip().isdigit():
                line = parts[1].strip()
        if line:
            items.append(line)
    return items