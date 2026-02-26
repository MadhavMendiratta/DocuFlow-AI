
import django.db.models.deletion
import uuid
from django.db import migrations, models

class Migration(migrations.Migration):

    dependencies = [
        ('documents', '0002_alter_document_file_size_alter_document_file_type'),
    ]

    operations = [
        migrations.CreateModel(
            name='APILog',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('analysis_type', models.CharField(help_text='e.g. summary, key_points, sentiment, topics', max_length=50)),
                ('model_used', models.CharField(default='gemini-1.5-flash', max_length=100)),
                ('input_tokens', models.PositiveIntegerField(default=0)),
                ('output_tokens', models.PositiveIntegerField(default=0)),
                ('total_tokens', models.PositiveIntegerField(default=0)),
                ('cost_estimated', models.DecimalField(decimal_places=6, default=0, help_text='Estimated cost in USD for this single API call', max_digits=10)),
                ('response_time', models.FloatField(blank=True, help_text='Wall-clock time for the API call in seconds', null=True)),
                ('success', models.BooleanField(default=True)),
                ('error_message', models.TextField(blank=True, default='')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('document', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='api_logs', to='documents.document')),
            ],
            options={
                'verbose_name': 'API Log',
                'verbose_name_plural': 'API Logs',
                'ordering': ['-created_at'],
                'indexes': [
                    models.Index(fields=['document', '-created_at'], name='documents_a_documen_b42c4e_idx'), 
                    models.Index(fields=['analysis_type'], name='documents_a_analysi_a61562_idx')
                ],
            },
        ),
    ]