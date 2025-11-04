# bulk_exiff

Utility for fixing EXIF timestamps in bulk. The provided script updates the
``DateTime``-related EXIF fields of every ``.jpg``/``.jpeg`` file inside a
directory so that they match the file's modification time.

## Requirements

* Python 3.10+
* [`piexif`](https://pypi.org/project/piexif/)

Install the dependency with::

    pip install piexif

## Usage

Run the script by passing the directory that contains your images or run it
without arguments and choose the folder from the interactive prompt/dialog:

```bash
python bulk_exif_fix.py /path/to/images
# or
python bulk_exif_fix.py
```

Optional arguments:

* ``--no-recursive`` – process only the top-level directory.
* ``--dry-run`` – show which files would be updated without modifying them.
