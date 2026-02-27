
from django.db import migrations, models

class Migration(migrations.Migration):

    dependencies = [
        ('documents', '0005_apilog'),
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