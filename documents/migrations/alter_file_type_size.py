
from django.db import migrations, models

class Migration(migrations.Migration):

    dependencies = [
        ('documents', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='document',
            name='file_size',
            field=models.PositiveIntegerField(default=0, help_text='File size in bytes'),
        ),
        migrations.AlterField(
            model_name='document',
            name='file_type',
            field=models.CharField(blank=True, choices=[('pdf', 'PDF'), ('txt', 'Text'), ('docx', 'Word Document')], default='', max_length=10),
        ),
    ]