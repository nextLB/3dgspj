from django.core.management.base import BaseCommand
from django.conf import settings
from django.contrib.auth.models import User
from projects.models import DatasetDirectory
from accounts.models import Profile
import os


class Command(BaseCommand):
    help = 'Seed initial data for VastGaussian system'

    def handle(self, *args, **options):
        # Create admin user if not exists
        if not User.objects.filter(username='admin').exists():
            admin = User.objects.create_superuser('admin', 'admin@vastgs.local', 'admin123')
            Profile.objects.update_or_create(user=admin, defaults={'role': 'admin'})
            self.stdout.write(self.style.SUCCESS('Created admin user: admin / admin123'))
        else:
            self.stdout.write('Admin user already exists.')

        # Create demo user
        if not User.objects.filter(username='demo').exists():
            user = User.objects.create_user('demo', 'demo@vastgs.local', 'demo123')
            Profile.objects.update_or_create(user=user, defaults={'role': 'user'})
            self.stdout.write(self.style.SUCCESS('Created demo user: demo / demo123'))

        # Seed dataset directories
        datasets_base = settings.DATASETS_BASE
        presets = [
            {'name': 'Mill-19 Rubble', 'path': os.path.join(datasets_base, 'Mill19/rubble'),
             'description': 'Mill-19 数据集 - rubble 场景'},
            {'name': 'Mill-19 Building', 'path': os.path.join(datasets_base, 'Mill19/building'),
             'description': 'Mill-19 数据集 - building 场景'},
            {'name': 'T&T Train', 'path': os.path.join(datasets_base, 'tandt/train'),
             'description': 'Tanks and Temples - train 场景'},
            {'name': 'T&T Truck', 'path': os.path.join(datasets_base, 'tandt/truck'),
             'description': 'Tanks and Temples - truck 场景'},
        ]

        for preset in presets:
            DatasetDirectory.objects.update_or_create(
                name=preset['name'],
                defaults={
                    'path': preset['path'],
                    'description': preset['description'],
                    'is_active': True,
                }
            )
            self.stdout.write(self.style.SUCCESS(f'Seeded preset dataset: {preset["name"]}'))

        self.stdout.write(self.style.SUCCESS('\nSeed data completed!'))
        self.stdout.write(f'\nVastGaussian base: {settings.VASTGAUSSIAN_BASE}')
        self.stdout.write(f'Datasets base: {datasets_base}')
