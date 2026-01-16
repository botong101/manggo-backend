

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('mangosense', '0015_mangoimage_model_filename'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='userconfirmation',
            options={'ordering': ['-id']},
        ),
        migrations.RemoveField(
            model_name='userconfirmation',
            name='client_ip',
        ),
        migrations.RemoveField(
            model_name='userconfirmation',
            name='confirmed_at',
        ),
    ]
