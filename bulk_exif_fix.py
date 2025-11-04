"""Utility for synchronizing EXIF timestamps for JPEG images.

The script walks through a chosen directory (including sub-directories) and
updates the EXIF ``DateTime``, ``DateTimeOriginal`` and ``DateTimeDigitized``
fields based on the file modification time.  The timestamps are written using
the local timezone and formatted according to the EXIF specification.

Example usage::

    python bulk_exif_fix.py /path/to/folder

Use ``--no-recursive`` to restrict the search to the top-level directory and
``--dry-run`` to inspect the changes without modifying the files.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
from pathlib import Path
from typing import Iterable, Iterator

try:
    import piexif
except ImportError as exc:  # pragma: no cover - dependency availability
    raise SystemExit(
        "The 'piexif' package is required to run this script. Install it with "
        "'pip install piexif'."
    ) from exc


JPEG_EXTENSIONS = {".jpg", ".jpeg"}


def request_directory_from_user() -> Path:
    """Prompt the user to choose a directory interactively.

    Tkinter is used when available to show a native directory selection dialog.
    If Tkinter is not installed or a display is not available, the function
    falls back to asking for the directory path via ``input``.
    """

    # Try Tkinter first for a nicer UX.
    try:  # pragma: no cover - depends on Tk availability
        import tkinter
        from tkinter import filedialog
    except Exception:  # noqa: BLE001 - broad to catch missing display modules
        tkinter = None  # type: ignore[assignment]
    else:
        root = tkinter.Tk()
        root.withdraw()
        selected = filedialog.askdirectory(title="Select directory with JPEG images")
        root.destroy()
        if selected:
            return Path(selected)

    # Fallback to terminal prompt.
    while True:
        response = input("Enter the path to the directory with JPEG images: ").strip()
        if not response:
            continue
        directory = Path(response).expanduser()
        if directory.exists() and directory.is_dir():
            return directory
        print(f"'{directory}' is not a valid directory. Please try again.")


def iter_jpeg_files(root: Path, recursive: bool = True) -> Iterator[Path]:
    """Yield JPEG files from ``root``.

    Parameters
    ----------
    root:
        Directory that contains the images.
    recursive:
        If ``True`` (the default), walk through sub-directories as well.
    """

    if recursive:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in JPEG_EXTENSIONS:
                yield path
    else:
        for path in root.iterdir():
            if path.is_file() and path.suffix.lower() in JPEG_EXTENSIONS:
                yield path


def format_timestamp(timestamp: float) -> bytes:
    """Convert a POSIX timestamp into an EXIF compatible byte string."""

    dt = _dt.datetime.fromtimestamp(timestamp).astimezone()
    return dt.strftime("%Y:%m:%d %H:%M:%S").encode("ascii")


def synchronize_exif_datetime(path: Path, dry_run: bool = False) -> bool:
    """Update EXIF timestamps for ``path``.

    Returns ``True`` when an update was performed (or would be performed in
    dry-run mode) and ``False`` otherwise.
    """

    timestamp = format_timestamp(path.stat().st_mtime)

    try:
        exif_dict = piexif.load(str(path))
    except piexif.InvalidImageDataError:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

    zeroth = exif_dict.setdefault("0th", {})
    exif_ifd = exif_dict.setdefault("Exif", {})

    updated = (
        zeroth.get(piexif.ImageIFD.DateTime) != timestamp
        or exif_ifd.get(piexif.ExifIFD.DateTimeOriginal) != timestamp
        or exif_ifd.get(piexif.ExifIFD.DateTimeDigitized) != timestamp
    )

    if not updated:
        return False

    zeroth[piexif.ImageIFD.DateTime] = timestamp
    exif_ifd[piexif.ExifIFD.DateTimeOriginal] = timestamp
    exif_ifd[piexif.ExifIFD.DateTimeDigitized] = timestamp

    if dry_run:
        return True

    piexif.insert(piexif.dump(exif_dict), str(path))
    return True


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        help="Directory that contains JPEG images",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Do not walk directories recursively",
    )
    parser.set_defaults(recursive=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would be updated without modifying them",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    directory: Path | None = args.directory

    if directory is None:
        directory = request_directory_from_user()

    if not directory.exists():
        raise SystemExit(f"Directory '{directory}' does not exist.")
    if not directory.is_dir():
        raise SystemExit(f"'{directory}' is not a directory.")

    updated_files = []

    for file_path in iter_jpeg_files(directory, recursive=args.recursive):
        if synchronize_exif_datetime(file_path, dry_run=args.dry_run):
            updated_files.append(file_path)

    if not updated_files:
        print("No files required updates.")
    else:
        action = "Would update" if args.dry_run else "Updated"
        print(f"{action} {len(updated_files)} file(s):")
        for file_path in updated_files:
            print(f" - {file_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
