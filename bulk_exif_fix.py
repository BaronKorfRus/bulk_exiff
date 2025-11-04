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
import json
import os
import re
import subprocess
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator, NamedTuple
import shutil

EXIFTOOL_PATH = shutil.which("exiftool")
if EXIFTOOL_PATH is None:
    raise SystemExit(
        "The 'exiftool' executable is required to run this script. "
        "Install it with 'brew install exiftool' or from "
        "https://exiftool.org/."
    )

try:
    from geotext import GeoText
except ImportError as exc:
    raise SystemExit(
        "The 'GeoText' package is required to parse city names. Install it with "
        "'pip install geotext'."
    ) from exc

try:
    import geonamescache
except ImportError as exc:
    raise SystemExit(
        "The 'geonamescache' package is required to resolve country names. "
        "Install it with 'pip install geonamescache'."
    ) from exc

JPEG_EXTENSIONS = {".jpg", ".jpeg"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}

DATE_CANDIDATE_RE = re.compile(
    r"(?P<date>"
    r"(?:19|20)\d{2}[-_.](?:0[1-9]|1[0-2])[-_.](?:0[1-9]|[12]\d|3[01])"
    r"|(?:0[1-9]|[12]\d|3[01])[-_.](?:0[1-9]|1[0-2])[-_.](?:19|20)\d{2}"
    r"|(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])"
    r")"
)
DATE_PATTERNS = (
    "%Y-%m-%d",
    "%Y.%m.%d",
    "%Y_%m_%d",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%Y%m%d",
    "%d%m%Y",
)

CYRILLIC_LATIN_MAP = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "kh",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "shch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
    "й": "y",
    "ё": "yo",
}

GEOTEXT_FALLBACK_RU = {
    "moscow": "Москва",
    "saintpetersburg": "Санкт-Петербург",
    "stpetersburg": "Санкт-Петербург",
    "stpeterburg": "Санкт-Петербург",
}

GEOTEXT_TOKEN_OVERRIDES = {
    "moskva": "Moscow",
    "moskou": "Moscow",
    "moscu": "Moscow",
    "moscva": "Moscow",
    "sanktpeterburg": "Saint Petersburg",
    "sanktpetersburg": "Saint Petersburg",
    "sanktpeterburg": "Saint Petersburg",
}

GEONAMES_CACHE = geonamescache.GeonamesCache()
COUNTRY_NAMES = {
    code: data["name"] for code, data in GEONAMES_CACHE.get_countries().items()
}


class _CityCacheEntry(NamedTuple):
    population: int
    country_code: str
    latitude: float
    longitude: float


CITY_COUNTRY_INDEX: dict[str, _CityCacheEntry] | None = None


def _normalize_ascii(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z]", "", ascii_only.lower())


def _transliterate_ru_to_ascii(text: str) -> str:
    """Transliterate Cyrillic to a GeoText-friendly ASCII representation."""

    result: list[str] = []
    prev_cyrillic: str | None = None
    for char in text:
        lower = char.lower()
        translit: str

        if lower == "е":
            boundary = prev_cyrillic is None or prev_cyrillic in {"ъ", "ь"}
            translit = "ye" if boundary else "e"
            translit = translit.capitalize() if char.isupper() else translit
        elif lower in {"ё", "ю", "я"}:
            base = {"ё": "yo", "ю": "yu", "я": "ya"}[lower]
            translit = base.capitalize() if char.isupper() else base
        elif lower == "й":
            translit = "Y" if char.isupper() else "y"
        else:
            base = CYRILLIC_LATIN_MAP.get(lower)
            if base is None:
                translit = char
            else:
                translit = base.capitalize() if char.isupper() else base

        result.append(translit)

        if lower in CYRILLIC_LATIN_MAP or lower in {"е", "ё", "ю", "я", "й"}:
            if char in {" ", "-", "_"}:
                prev_cyrillic = None
            elif lower in {"ъ", "ь"}:
                prev_cyrillic = lower
            else:
                prev_cyrillic = lower
        elif char in {" ", "-", "_"}:
            prev_cyrillic = None
        else:
            prev_cyrillic = None

    return "".join(result)

def _city_name_keys(name: str) -> set[str]:
    variants = {name}
    transliterated = _transliterate_ru_to_ascii(name)
    if transliterated != name:
        variants.add(transliterated)

    normalized = unicodedata.normalize("NFKD", name)
    ascii_variant = normalized.encode("ascii", "ignore").decode("ascii")
    if ascii_variant:
        variants.add(ascii_variant)

    keys = {_normalize_ascii(variant) for variant in variants}
    return {key for key in keys if key}


def _build_city_country_index() -> dict[str, _CityCacheEntry]:
    global CITY_COUNTRY_INDEX
    if CITY_COUNTRY_INDEX is not None:
        return CITY_COUNTRY_INDEX

    index: dict[str, _CityCacheEntry] = {}
    for payload in GEONAMES_CACHE.get_cities().values():
        population = int(payload.get("population") or 0)
        country_code = payload.get("countrycode") or ""
        try:
            latitude = float(payload.get("latitude"))
            longitude = float(payload.get("longitude"))
        except (TypeError, ValueError):
            continue
        names = {payload.get("name", "")}
        ascii_name = payload.get("ascii")
        if ascii_name:
            names.add(ascii_name)

        alternate = payload.get("alternatenames")
        if alternate:
            if isinstance(alternate, str):
                candidates = (token.strip() for token in alternate.split(","))
            else:
                candidates = (str(token).strip() for token in alternate)
            names.update(token for token in candidates if token)

        for name in names:
            for key in _city_name_keys(name):
                existing = index.get(key)
                if existing is None or population > existing.population:
                    index[key] = _CityCacheEntry(population, country_code, latitude, longitude)

    CITY_COUNTRY_INDEX = index
    return index


def _lookup_city_location(
    city: str | None,
) -> tuple[str | None, float | None, float | None]:
    if not city:
        return None, None, None

    index = _build_city_country_index()
    keys: list[str] = []

    base = _normalize_ascii(city)
    if base:
        keys.append(base)

    transliterated = _normalize_ascii(_transliterate_ru_to_ascii(city))
    if transliterated and transliterated not in keys:
        keys.append(transliterated)

    for key in keys:
        entry = index.get(key)
        if entry:
            country_code = entry.country_code
            country = COUNTRY_NAMES.get(country_code, country_code) if country_code else None
            return country, entry.latitude, entry.longitude

    return None, None, None


def _lookup_country_name(city: str | None) -> str | None:
    country, _, _ = _lookup_city_location(city)
    return country


def _extract_description_fragment(fragment: str, city: str | None) -> str | None:
    """Derive description text by removing the detected city from the fragment."""

    if not fragment:
        return None

    result = fragment
    if city:
        pattern_word = re.compile(rf"(?i)\b{re.escape(city)}\b")
        updated = pattern_word.sub(" ", result, count=1)
        if updated == result:
            pattern_partial = re.compile(re.escape(city), re.IGNORECASE)
            updated = pattern_partial.sub(" ", result, count=1)
        result = updated

    result = re.sub(r"\s+", " ", result).strip(" -_,")
    return result or None


def _compose_caption(
    country: str | None, city: str | None, description: str | None
) -> str | None:
    parts: list[str] = []
    if country:
        parts.append(country)
    if city:
        parts.append(city)
    if description:
        parts.append(description)
    if not parts:
        return None
    return ", ".join(parts)


def _normalize_string(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        value = value[0]
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8").strip()
        except UnicodeDecodeError:
            return value.decode("latin-1", errors="ignore").strip()
    return str(value).strip()


def _parse_float_value(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            return float(candidate.split()[0])
        except ValueError:
            return None
    return None


def _compute_datetimes(path: Path, shoot_date: _dt.date | None) -> tuple[str, str]:
    local_tz = _dt.datetime.now().astimezone().tzinfo
    if shoot_date:
        dt_local = _dt.datetime.combine(shoot_date, _dt.time(), tzinfo=local_tz)
    else:
        dt_local = _dt.datetime.fromtimestamp(path.stat().st_mtime, tz=_dt.timezone.utc).astimezone(local_tz)
    dt_local = dt_local.replace(microsecond=0)
    exif_str = dt_local.strftime("%Y:%m:%d %H:%M:%S")
    dt_utc = dt_local.astimezone(_dt.timezone.utc)
    qt_str = dt_utc.strftime("%Y:%m:%d %H:%M:%S%z")
    if len(qt_str) > 5:
        qt_str = qt_str[:-2] + ":" + qt_str[-2:]
    return exif_str, qt_str


def _read_metadata_with_exiftool(path: Path, is_video: bool) -> dict[str, object]:
    tags = [
        "-ImageDescription",
        "-XPTitle",
        "-XPComment",
        "-IPTC:Caption-Abstract",
        "-XMP:Description",
        "-DateTimeOriginal",
        "-CreateDate",
        "-ModifyDate",
        "-XMP:DateTimeOriginal",
        "-XMP:CreateDate",
        "-XMP:ModifyDate",
        "-GPSLatitude",
        "-GPSLongitude",
        "-GPSLatitudeRef",
        "-GPSLongitudeRef",
        "-GPSMapDatum",
    ]
    if is_video:
        tags.extend(
            [
                "-QuickTime:CreateDate",
                "-QuickTime:ContentCreateDate",
                "-QuickTime:CreationDate",
                "-QuickTime:ModifyDate",
                "-QuickTime:GPSLatitude",
                "-QuickTime:GPSLongitude",
                "-QuickTime:GPSCoordinates",
                "-Keys:GPSCoordinates",
            ]
        )

    cmd = [EXIFTOOL_PATH, "-j", "-n"] + tags + [str(path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return {}
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}
    if not data:
        return {}
    entry = data[0]
    if not isinstance(entry, dict):
        return {}
    return entry


def _build_expected_metadata(
    path: Path,
    caption: str | None,
    gps_coords: tuple[float, float] | None,
    shoot_date: _dt.date | None,
    is_video: bool,
) -> tuple[dict[str, object], list[str]]:
    expected: dict[str, object] = {}
    args: list[str] = []

    def assign_string(tag: str, value: str | None) -> None:
        value = value or ""
        args.append(f"-{tag}=")
        expected[tag] = value
        if value:
            args.append(f"-{tag}={value}")

    def assign_float(tag: str, value: float | None, precision: int = 6) -> None:
        args.append(f"-{tag}=")
        if value is None:
            expected[tag] = None
        else:
            formatted = f"{value:.{precision}f}"
            args.append(f"-{tag}={formatted}")
            expected[tag] = round(value, precision)

    assign_string("ImageDescription", "")
    assign_string("XPTitle", caption)
    assign_string("XPComment", caption)
    assign_string("IPTC:Caption-Abstract", caption)
    assign_string("XMP:Description", caption)

    if gps_coords:
        latitude, longitude = gps_coords
        assign_float("GPSLatitude", abs(latitude))
        assign_float("GPSLongitude", abs(longitude))
        assign_string("GPSLatitudeRef", "N" if latitude >= 0 else "S")
        assign_string("GPSLongitudeRef", "E" if longitude >= 0 else "W")
        assign_string("GPSMapDatum", "WGS-84")
    else:
        assign_float("GPSLatitude", None)
        assign_float("GPSLongitude", None)
        assign_string("GPSLatitudeRef", "")
        assign_string("GPSLongitudeRef", "")
        assign_string("GPSMapDatum", "")

    exif_dt, qt_dt = _compute_datetimes(path, shoot_date)
    assign_string("DateTimeOriginal", exif_dt)
    assign_string("CreateDate", exif_dt)
    assign_string("ModifyDate", exif_dt)
    assign_string("XMP:DateTimeOriginal", exif_dt)
    assign_string("XMP:CreateDate", exif_dt)
    assign_string("XMP:ModifyDate", exif_dt)

    if is_video:
        assign_string("QuickTime:Comment", caption)
        if gps_coords:
            latitude, longitude = gps_coords
            assign_float("QuickTime:GPSLatitude", latitude)
            assign_float("QuickTime:GPSLongitude", longitude)
            coord_pair = f"{latitude:.6f} {longitude:.6f}"
            assign_string("QuickTime:GPSCoordinates", coord_pair)
            assign_string("Keys:GPSCoordinates", coord_pair)
        else:
            assign_float("QuickTime:GPSLatitude", None)
            assign_float("QuickTime:GPSLongitude", None)
            assign_string("QuickTime:GPSCoordinates", "")
            assign_string("Keys:GPSCoordinates", "")

        assign_string("QuickTime:CreateDate", qt_dt)
        assign_string("QuickTime:ContentCreateDate", qt_dt)
        assign_string("QuickTime:CreationDate", qt_dt)
        assign_string("QuickTime:ModifyDate", qt_dt)

    return expected, args


def _metadata_matches(existing: dict[str, object], expected: dict[str, object]) -> bool:
    for tag, desired in expected.items():
        current = existing.get(tag)
        if isinstance(desired, float):
            current_value = _parse_float_value(current)
            if current_value is None or abs(current_value - desired) > 1e-6:
                return False
        elif desired == "":
            if _normalize_string(current) != "":
                return False
        elif desired is None:
            if _normalize_string(current) != "":
                return False
        else:
            if _normalize_string(current) != desired:
                return False
    return True


def _resolve_city_name(
    geotext_candidates: list[str], original_fragment: str
) -> str | None:
    """Match GeoText candidates back to the original fragment."""

    tokens = re.findall(r"[^\s]+", original_fragment, flags=re.UNICODE)
    transliterated_tokens = [_transliterate_ru_to_ascii(token) for token in tokens]
    for candidate in geotext_candidates:
        candidate_norm = _normalize_ascii(candidate)
        # Try contiguous token spans.
        for start in range(len(tokens)):
            accum: list[str] = []
            for end in range(start, len(tokens)):
                accum.append(transliterated_tokens[end])
                span_translit = " ".join(accum)
                if _normalize_ascii(span_translit) == candidate_norm:
                    return " ".join(tokens[start : end + 1])

        fallback = GEOTEXT_FALLBACK_RU.get(candidate_norm)
        if fallback:
            if fallback in original_fragment:
                return fallback
            return fallback
        if candidate_norm == "rostov" and "Ростов-на-Дону" in original_fragment:
            return "Ростов-на-Дону"

    return None


def _build_geotext_input(fragment: str) -> str:
    """Prepare GeoText input string with transliteration and overrides."""

    tokens = re.findall(r"[^\s]+", fragment, flags=re.UNICODE)
    prepared: list[str] = []
    for token in tokens:
        translit = _transliterate_ru_to_ascii(token)
        override = GEOTEXT_TOKEN_OVERRIDES.get(_normalize_ascii(translit))
        candidate = override if override else translit
        if "-" in candidate and not GeoText(candidate).cities:
            candidate_with_spaces = candidate.replace("-", " ")
            if GeoText(candidate_with_spaces).cities:
                candidate = candidate_with_spaces
        prepared.append(candidate)

    city_indexes: set[int] = set()
    max_span = min(3, len(prepared))
    for span in range(max_span, 0, -1):
        for start in range(0, len(prepared) - span + 1):
            phrase = " ".join(prepared[start : start + span])
            if GeoText(phrase).cities:
                city_indexes.update(range(start, start + span))

    final_tokens = [
        token if idx in city_indexes else token.lower()
        for idx, token in enumerate(prepared)
    ]
    return " ".join(final_tokens)

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
    """Yield JPEG files from ``root``."""

    if recursive:
        for dirpath, _, filenames in os.walk(root, followlinks=True):
            base = Path(dirpath)
            for filename in filenames:
                suffix = Path(filename).suffix.lower()
                if suffix in JPEG_EXTENSIONS:
                    yield base / filename
    else:
        for path in root.iterdir():
            if path.is_file() and path.suffix.lower() in JPEG_EXTENSIONS:
                yield path


def iter_video_files(root: Path, recursive: bool = True) -> Iterator[Path]:
    """Yield supported video files from ``root``."""

    if recursive:
        for dirpath, _, filenames in os.walk(root, followlinks=True):
            base = Path(dirpath)
            for filename in filenames:
                suffix = Path(filename).suffix.lower()
                if suffix in VIDEO_EXTENSIONS:
                    yield base / filename
    else:
        for path in root.iterdir():
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                yield path


def synchronize_exif_datetime(
    path: Path,
    caption: str | None = None,
    gps_coords: tuple[float, float] | None = None,
    shoot_date: _dt.date | None = None,
    dry_run: bool = False,
) -> bool:
    """Update metadata for ``path`` using exiftool."""

    suffix = path.suffix.lower()
    if suffix not in JPEG_EXTENSIONS and suffix not in VIDEO_EXTENSIONS:
        return False

    is_video = suffix in VIDEO_EXTENSIONS
    existing = _read_metadata_with_exiftool(path, is_video)
    expected, args = _build_expected_metadata(path, caption, gps_coords, shoot_date, is_video)

    if _metadata_matches(existing, expected):
        return False

    if dry_run:
        return True

    cmd = [
        EXIFTOOL_PATH,
        "-overwrite_original",
        "-P",
        "-q",
        "-charset",
        "iptc=UTF8",
        "-charset",
        "xmp=UTF8",
        "-charset",
        "exif=UTF8",
    ] + args + [str(path)]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        error = result.stderr.strip() or "unknown error"
        print(f"Failed to update metadata for '{path}': {error}")
        return False

    return True




def extract_directory_metadata(
    directory: Path,
) -> tuple[
    str | None, str | None, str | None, tuple[float, float] | None, _dt.date | None
]:
    """Extract city, country, description, coordinates and date from the directory name."""

    name = directory.name
    match = DATE_CANDIDATE_RE.search(name)
    if match:
        raw_date = match.group("date")
        normalized = re.sub(r"[_.]", "-", raw_date)
        parsed_date = None
        candidates = (raw_date, normalized)
        for pattern in DATE_PATTERNS:
            for candidate in candidates:
                try:
                    parsed_date = _dt.datetime.strptime(candidate, pattern).date()
                    break
                except ValueError:
                    continue
            if parsed_date:
                break
    else:
        parsed_date = None

    if match:
        city_fragment = (name[: match.start()] + " " + name[match.end() :]).strip(" _-.")
    else:
        city_fragment = name

    cleaned_fragment = re.sub(r"_+", " ", city_fragment)
    cleaned_fragment = re.sub(r"\s+", " ", cleaned_fragment).strip()
    city: str | None = None
    country: str | None = None
    coordinates: tuple[float, float] | None = None
    if cleaned_fragment:
        geotext_input = _build_geotext_input(cleaned_fragment)
        geo = GeoText(geotext_input)
        if geo.cities:
            city = _resolve_city_name(geo.cities, cleaned_fragment)
            country, latitude, longitude = _lookup_city_location(city)
            if latitude is not None and longitude is not None:
                coordinates = (latitude, longitude)

    description = (
        _extract_description_fragment(cleaned_fragment, city) if cleaned_fragment else None
    )

    return city, country, description, coordinates, parsed_date


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

    city, country, description, coordinates, shoot_date = extract_directory_metadata(directory)
    caption = _compose_caption(country, city, description)
    if city or country or description or coordinates or shoot_date:
        print("Parsed directory metadata:")
        if city:
            print(f" - City: {city}")
        else:
            print(" - City: not recognized by GeoText")
        if country:
            print(f" - Country: {country}")
        elif city:
            print(" - Country: not resolved for the detected city")
        if description:
            print(f" - Description: {description}")
        if caption and caption != description:
            print(f" - Caption: {caption}")
        if coordinates:
            lat, lon = coordinates
            print(f" - Coordinates: {lat:.6f}, {lon:.6f}")
        elif city:
            print(" - Coordinates: not resolved for the detected city")
        if shoot_date:
            print(f" - Date: {shoot_date.isoformat()}")

    jpeg_paths = list(iter_jpeg_files(directory, recursive=args.recursive))
    video_paths = list(iter_video_files(directory, recursive=args.recursive))

    updated_images: list[Path] = []
    updated_videos: list[Path] = []
    unchanged_images: list[Path] = []
    unchanged_videos: list[Path] = []
    for file_path in jpeg_paths:
        if synchronize_exif_datetime(
            file_path,
            caption=caption,
            gps_coords=coordinates,
            shoot_date=shoot_date,
            dry_run=args.dry_run,
        ):
            updated_images.append(file_path)
        else:
            unchanged_images.append(file_path)

    for file_path in video_paths:
        updated = synchronize_exif_datetime(
            file_path,
            caption=caption,
            gps_coords=coordinates,
            shoot_date=shoot_date,
            dry_run=args.dry_run,
        )
        if updated:
            updated_videos.append(file_path)
        else:
            unchanged_videos.append(file_path)

    total_updates = len(updated_images) + len(updated_videos)
    if not total_updates:
        print("No files required updates.")
    else:
        action = "Would update" if args.dry_run else "Updated"
        print(f"{action} {total_updates} file(s):")
        if updated_images:
            print("  Images:")
            for file_path in updated_images:
                print(f"   - {file_path}")
        if updated_videos:
            print("  Videos:")
            for file_path in updated_videos:
                print(f"   - {file_path}")

    unchanged_total = len(unchanged_images) + len(unchanged_videos)
    if unchanged_total:
        print(f"{unchanged_total} file(s) already contained the requested metadata:")
        if unchanged_images:
            print("  Images:")
            for file_path in unchanged_images:
                print(f"   - {file_path}")
        if unchanged_videos:
            print("  Videos:")
            for file_path in unchanged_videos:
                print(f"   - {file_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
