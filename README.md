# AI Document Processor

A Django-based document processing pipeline that extracts text from uploaded files (PDF, DOCX, TXT), runs multi-faceted AI analysis via Google Gemini, and delivers real-time progress updates over WebSockets. Supports both single-document and multi-document batch workflows with cross-document analysis, content-hash-based deduplication caching, and per-call API cost tracking.

## Features

- **Multi-format text extraction** -- PDF (pdfplumber with PyPDF2 fallback), DOCX (python-docx), and plain text with automatic encoding detection.
- **AI-powered document analysis** -- summary generation, key-point extraction, sentiment analysis, and topic identification using Google Gemini (`gemini-1.5-flash` by default).
- **Batch processing** -- upload multiple documents in a single request; the system processes each individually, then runs cross-document analysis producing a combined summary, common themes, key insights, and contradiction detection.
- **Real-time progress** -- WebSocket consumers push processing status, progress percentage, and step-level updates to connected clients.
- **Content-hash deduplication** -- SHA-256 hashes of file content are cached in Redis. Duplicate uploads skip both text extraction and AI analysis entirely.
- **API cost tracking** -- every Gemini call is logged with input/output token counts and estimated USD cost, aggregated per document and per batch.
- **Celery task pipeline** -- asynchronous processing with automatic retries (exponential backoff), rate limiting (respects Gemini free-tier quotas), task cancellation, and periodic cleanup jobs.
- **REST API + Web UI** -- full CRUD API via Django REST Framework alongside server-rendered HTML templates (dashboard, document list, batch list, upload form, detail pages).
- **Admin interface** -- custom admin actions for bulk reprocessing, status management, and inline viewing of analysis results and processing logs.

## Tech Stack

| Layer | Technology |
|---|---|
| Backend framework | Django 4.2+, Django REST Framework |
| Async task queue | Celery with Redis broker |
| WebSockets | Django Channels, channels-redis |
| AI / LLM | Google Gemini API (`google-generativeai`) |
| Document parsing | pdfplumber, PyPDF2, python-docx |
| Database | PostgreSQL (SQLite fallback for development) |
| Cache | Redis (Django cache framework) |
| ASGI server | Daphne |
| Static files | WhiteNoise |
| Configuration | python-decouple, dj-database-url |

## Project Structure

```
ai-document-processor/
├── core/                       # Django project configuration
│   ├── settings.py             # DB, Redis, Celery, Gemini, CORS, logging
│   ├── urls.py                 # Root URL conf (admin, auth, documents app)
│   ├── celery.py               # Celery app, beat schedule, rate-limit annotations
│   ├── asgi.py                 # ASGI entrypoint — HTTP + WebSocket routing
│   └── wsgi.py                 # WSGI entrypoint
├── documents/                  # Main application
│   ├── models.py               # Document, DocumentBatch, BatchAnalysis,
│   │                           # DocumentAnalysis, ProcessingJob, ProcessingLog,
│   │                           # APILog, DocumentTag, DocumentTagging
│   ├── views.py                # DRF ViewSets, batch API endpoints, template views
│   ├── serializers.py          # Request/response serializers incl. batch upload
│   ├── tasks.py                # Celery tasks: process_document, process_batch,
│   │                           # text extractors, Gemini analysis, cleanup jobs
│   ├── consumers.py            # WebSocket consumers (document + list updates)
│   ├── routing.py              # WebSocket URL patterns
│   ├── cache.py                # Redis caching — content hash, text, analysis
│   ├── admin.py                # Admin config with custom actions
│   ├── urls.py                 # API router + web interface URL patterns
│   └── migrations/             # Database migrations
├── templates/                  # Server-rendered HTML templates
│   ├── base.html
│   └── documents/              # dashboard, document_list, document_detail,
│                               # batch_list, batch_detail, upload
├── media/                      # User-uploaded files (per-user subdirectories)
├── static/                     # Static assets
├── logs/                       # Application log files
├── manage.py
└── requirements.txt
```

## Installation and Setup

### Prerequisites

- Python 3.10+
- PostgreSQL 14+ (optional -- SQLite works for development)
- Redis 6+
- A Google Gemini API key ([get one here](https://aistudio.google.com/app/apikey))

### Clone the Repository

```bash
git clone <repository-url>
cd ai-document-processor
```

### Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file in the project root:

```dotenv
# Required
GEMINI_API_KEY=your_gemini_api_key

# Database (defaults to sqlite:///db.sqlite3 if omitted)
DATABASE_URL=postgresql://user:password@localhost:5432/ai_doc_processor

# Redis (defaults to redis://localhost:6379/0)
REDIS_URL=redis://localhost:6379/0

# Optional overrides
SECRET_KEY=your-secret-key
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
MAX_UPLOAD_SIZE=10485760
ALLOWED_FILE_TYPES=pdf,txt,docx
GEMINI_MODEL=gemini-1.5-flash
GEMINI_MAX_TOKENS=1000
GEMINI_TEMPERATURE=0.7
GEMINI_COST_PER_INPUT_TOKEN=0.000000075
GEMINI_COST_PER_OUTPUT_TOKEN=0.0000003
DOCUMENT_CACHE_TTL=604800
```

### Initialize the Database

```bash
python manage.py migrate
python manage.py createsuperuser
python manage.py collectstatic --noinput
```

## Running the Project

Three processes are required: Redis, Celery worker, and the Django server.

**Terminal 1 -- Redis:**

```bash
redis-server
```

**Terminal 2 -- Celery worker:**

```bash
celery -A core worker --loglevel=info
```

**Terminal 3 -- Django development server:**

```bash
python manage.py runserver
```

Open `http://127.0.0.1:8000` in a browser. The admin panel is at `http://127.0.0.1:8000/admin`.

For production, use Daphne (ASGI) instead of `runserver` to enable WebSocket support:

```bash
daphne -b 0.0.0.0 -p 8000 core.asgi:application
```

To run periodic tasks (failed-job cleanup, old-log pruning), start a Celery beat process alongside the worker:

```bash
celery -A core beat --loglevel=info
```

## API Endpoints

All API endpoints require session authentication. Prefix: `/api/`.

### Documents

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/documents/` | List documents (filterable by `status`, `file_type`, `search`) |
| POST | `/api/documents/` | Create and auto-process a document |
| GET | `/api/documents/{id}/` | Retrieve document details |
| PUT | `/api/documents/{id}/` | Update a document |
| DELETE | `/api/documents/{id}/` | Delete a document |
| POST | `/api/documents/upload/` | Upload with processing options (`file`, `title`, `start_processing`, `analysis_types`) |
| POST | `/api/documents/{id}/reprocess/` | Re-trigger the processing pipeline |
| POST | `/api/documents/{id}/cancel_processing/` | Cancel an active processing job |
| GET | `/api/documents/{id}/analysis/` | Retrieve AI analysis results |
| GET | `/api/documents/{id}/logs/` | Paginated processing logs |
| GET | `/api/documents/stats/` | Aggregate statistics for the current user |
| GET | `/api/documents/{id}/status/` | Lightweight status polling endpoint |

### Batches

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/batch/upload/` | Upload multiple files as a batch (`files`, optional `title`) |
| GET | `/api/batch/{id}/status/` | Batch progress, per-document breakdown, cost/token totals |
| GET | `/api/batch/{id}/result/` | Cross-document analysis: combined summary, insights, contradictions |
| POST | `/api/batch/{id}/retry/` | Retry a failed batch (resets failed docs, respects max retries) |
| POST | `/api/batch/{id}/cancel/` | Cancel a pending/processing batch |

### Processing Jobs

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/processing-jobs/` | List processing jobs for the current user |
| GET | `/api/processing-jobs/{id}/` | Retrieve a specific processing job |

### Other

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health/` | Health check |

### WebSocket

| Endpoint | Description |
|---|---|
| `ws/documents/{document_id}/` | Real-time processing updates for a specific document |
| `ws/documents/` | Real-time updates for the document list |

Supported inbound messages: `get_status`, `cancel_processing`.

```javascript
const socket = new WebSocket('ws://localhost:8000/ws/documents/<document-uuid>/');

socket.onmessage = (e) => {
    const data = JSON.parse(e.data);
    // data.type: 'processing_update' | 'processing_complete' | 'processing_error'
    // data.progress: 0-100
    // data.message: human-readable step description
};
```

## Usage

1. Log in via the admin panel (`/admin`) or session authentication.
2. Navigate to the upload page or POST a file to `/api/documents/upload/`.
3. The system automatically extracts text and runs AI analysis asynchronously.
4. Monitor progress in real time via the web dashboard or a WebSocket connection.
5. View results on the document detail page or via `/api/documents/{id}/analysis/`.
6. For multi-document analysis, use `/api/batch/upload/` with multiple files, then poll `/api/batch/{id}/status/` or fetch results from `/api/batch/{id}/result/`.

## Future Improvements

- Token-based authentication (JWT) for stateless API access.
- Support for additional file formats (XLSX, PPTX, HTML, images via OCR).
- Configurable analysis prompts and custom analysis types per document.
- Streaming Gemini responses for incremental result delivery.
- Batch-level WebSocket consumer for real-time batch progress.
- Role-based access control and team/organization document sharing.
- Export analysis results as PDF or structured reports.
- Comprehensive test suite covering task pipeline, API, and WebSocket flows.
- Containerized deployment with Docker Compose (Django, Celery, Redis, PostgreSQL).

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Commit changes with clear messages.
4. Push to your fork and open a pull request.
5. Ensure any new functionality includes tests and does not break existing behaviour.


