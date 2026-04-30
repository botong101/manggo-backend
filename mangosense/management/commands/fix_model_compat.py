"""
fix_model_compat.py
-------------------
One-time fix for .keras model files saved with a newer Keras version
that serialises quantization_config=None inside Dense layer configs.

Older Keras raises:
    "Unrecognized keyword arguments passed to Dense: {'quantization_config': None}"

This command rewrites the config.json inside every .keras (ZIP) file in the
models/ directory, recursively stripping any 'quantization_config' key whose
value is None. Weights are untouched.
"""
import json
import os
import shutil
import tempfile
import zipfile

from django.conf import settings
from django.core.management.base import BaseCommand


def _strip_quantization_config(obj):
    """Recursively remove quantization_config: None from a JSON structure."""
    if isinstance(obj, dict):
        return {
            k: _strip_quantization_config(v)
            for k, v in obj.items()
            if not (k == 'quantization_config' and v is None)
        }
    if isinstance(obj, list):
        return [_strip_quantization_config(item) for item in obj]
    return obj


class Command(BaseCommand):
    help = 'Strip quantization_config from .keras model files for Keras version compatibility'

    def add_arguments(self, parser):
        parser.add_argument(
            '--models-dir',
            default=os.path.join(settings.BASE_DIR, 'models'),
            help='Directory containing .keras model files (default: BASE_DIR/models/)',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be changed without modifying any files',
        )

    def handle(self, *args, **options):
        models_dir = options['models_dir']
        dry_run = options['dry_run']

        if not os.path.isdir(models_dir):
            self.stderr.write(f'Models directory not found: {models_dir}')
            return

        keras_files = [
            os.path.join(models_dir, f)
            for f in os.listdir(models_dir)
            if f.endswith('.keras')
        ]

        if not keras_files:
            self.stdout.write('No .keras files found.')
            return

        for model_path in sorted(keras_files):
            self._patch_file(model_path, dry_run)

        if dry_run:
            self.stdout.write(self.style.WARNING('Dry run — no files were modified.'))
        else:
            self.stdout.write(self.style.SUCCESS('All model files patched.'))

    def _patch_file(self, model_path, dry_run):
        filename = os.path.basename(model_path)

        try:
            with zipfile.ZipFile(model_path, 'r') as zf:
                raw_config = zf.read('config.json')
        except Exception as exc:
            self.stderr.write(f'  {filename}: could not read — {exc}')
            return

        original = json.loads(raw_config)
        patched = _strip_quantization_config(original)

        if original == patched:
            self.stdout.write(f'  {filename}: no quantization_config found, skipping')
            return

        if dry_run:
            self.stdout.write(self.style.WARNING(f'  {filename}: would strip quantization_config'))
            return

        # Rewrite: copy all ZIP entries, replacing config.json
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.keras', dir=os.path.dirname(model_path))
        os.close(tmp_fd)
        try:
            with zipfile.ZipFile(model_path, 'r') as src_zip, \
                 zipfile.ZipFile(tmp_path, 'w', compression=zipfile.ZIP_DEFLATED) as dst_zip:
                for entry in src_zip.infolist():
                    if entry.filename == 'config.json':
                        dst_zip.writestr(entry, json.dumps(patched))
                    else:
                        dst_zip.writestr(entry, src_zip.read(entry.filename))

            shutil.move(tmp_path, model_path)
            self.stdout.write(self.style.SUCCESS(f'  {filename}: patched'))
        except Exception as exc:
            os.unlink(tmp_path)
            self.stderr.write(f'  {filename}: failed to patch — {exc}')
