
from django.db import migrations, models

class Migration(migrations.Migration):

    dependencies = [
        # Updated to point to the Commit 22 migration where DocumentBatch was created
        ('documents', '0006_documentbatch_batchanalysis_document_batch'),
    ]

    operations = [
        migrations.AddField(
            model_name='documentbatch',
            name='max_retries',
            field=models.PositiveIntegerField(default=3),
        ),
        migrations.AddField(
            model_name='documentbatch',
            name='retry_count',
            field=models.PositiveIntegerField(default=0),
        ),
    ]