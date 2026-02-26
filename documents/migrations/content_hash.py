from django.db import migrations, models

class Migration(migrations.Migration):

    dependencies = [
        # Updated to point to our previous Commit 16 migration
        ('documents', '0005_apilog'), 
    ]

    operations = [
        migrations.AddField(
            model_name='document',
            name='content_hash',
            field=models.CharField(blank=True, db_index=True, default='', help_text='SHA-256 hash of file content for deduplication', max_length=64),
        ),
    ]
    