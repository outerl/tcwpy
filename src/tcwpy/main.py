"""Read TransCAD fixed-width binary tables.

A table consists of a ``.bin`` data file and a matching ``.dcb`` dictionary.
The dictionary supplies each column's name, type, one-based start byte, and width, plus
the full record width. :func:`read_transcad_binary` is the public entry point.
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["TranscadBinaryError", "TranscadDictionaryError", "read_transcad_binary"]

PathLike = str | Path

# caliperR maps these to R date/time classes, but TransCAD's on-disk encoding for
# them is undocumented and has not been verified against a real file. Rejecting
# them is safer than decoding them into plausible-looking wrong dates.
_UNVERIFIED_TYPES = {"DATE", "TIME", "DATETIME"}  # TODO is this what we want to do

# The two extensions that name a table; either one identifies the pair.
_TABLE_SUFFIXES = (".bin", ".dcb")

# Keys are TransCAD types

_NUMPY_FORMATS = {
    "I": "<i4",
    "S": "<i2",
    "R": "<f8",
    "F": "<f4",
}

# The values TransCAD uses for missing values
_TRANSCAD_MISSING_VALUES: dict[str, int | float] = {
    "I": int(np.iinfo(np.int32).min) + 1,
    "S": int(np.iinfo(np.int16).min) + 1,
    "R": float(np.finfo(np.float64).min),
    "F": float(np.finfo(np.float32).min),
}

_PANDAS_INTEGER_DTYPES = {
    "I": "Int32",
    "S": "Int16",
}

# TransCAD marks a deleted FFB record by overwriting its first 16 bytes with
# this identifier (or the identifier's prefix when the record is narrower).
_DELETED_RECORD_MARKER = bytes.fromhex(
    "91 8b 4a 5c bc db 4f 14 63 23 7f 78 a6 95 0d 27"
)


class TranscadBinaryError(Exception):
    """Base error for invalid TransCAD binary tables."""


class TranscadDictionaryError(TranscadBinaryError):
    """Raised when a TransCAD ``.dcb`` dictionary is malformed."""


@dataclass(frozen=True)
class _Column:
    name: str
    transcad_type: str
    numpy_format: str
    offset: int
    width: int
    description: str
    display_name: str | None


def _table_stem(path: PathLike) -> Path:
    """Drop a ``.bin`` or ``.dcb`` extension, leaving any other dotted name intact."""
    resolved = Path(path).expanduser()
    return (
        resolved.with_suffix("")
        if resolved.suffix.lower() in _TABLE_SUFFIXES
        else resolved
    )


def _companion_path(stem: Path, suffix: str) -> Path:
    """
    TransCAD writes the files with inconsistent case, so find the version that exists.
    """
    candidates = [stem.with_name(stem.name + case) for case in (suffix.lower(), suffix.upper())]
    return next(
        (candidate for candidate in candidates if candidate.is_file()), candidates[0]
    )


def _numpy_format(
    type_code: str, width: int, dictionary_path: Path, line_number: int
) -> str:
    transcad_type = type_code.strip().upper()
    if transcad_type in _UNVERIFIED_TYPES:
        raise TranscadDictionaryError(
            f"{dictionary_path} line {line_number}: TransCAD {type_code.strip()!r} fields are not supported "
            f"because their binary encoding is unverified; export the table from TransCAD instead"
        )
    if transcad_type == "C":
        return f"S{width}"

    try:
        numpy_format = _NUMPY_FORMATS[transcad_type]
    except KeyError as exc:
        raise TranscadDictionaryError(
            f"{dictionary_path} line {line_number}: unsupported field type {type_code!r}"
        ) from exc

    expected_width = np.dtype(numpy_format).itemsize
    if width != expected_width:
        raise TranscadDictionaryError(
            f"{dictionary_path} line {line_number}: TransCAD field type {transcad_type!r} "
            f"requires width {expected_width}, got {width}"
        )
    return numpy_format


def _parse_column(line: str, line_number: int, dictionary_path: Path) -> _Column:
    try:
        parts = next(csv.reader([line], skipinitialspace=True, strict=True))
    except csv.Error as exc:
        raise TranscadDictionaryError(
            f"{dictionary_path} line {line_number}: malformed comma-separated field definition"
        ) from exc
    if len(parts) < 4:
        raise TranscadDictionaryError(
            f"{dictionary_path} line {line_number}: expected at least 4 comma-separated values"
        )

    name = parts[0].strip()
    if not name:
        raise TranscadDictionaryError(
            f"{dictionary_path} line {line_number}: field name is empty"
        )
    try:
        start_byte = int(parts[2])
        width = int(parts[3])
    except ValueError as exc:
        raise TranscadDictionaryError(
            f"{dictionary_path} line {line_number}: field start byte and width must be integers"
        ) from exc
    if start_byte <= 0 or width <= 0:
        raise TranscadDictionaryError(
            f"{dictionary_path} line {line_number}: field start byte and width must be positive"
        )

    return _Column(
        name=name,
        transcad_type=parts[1].strip().upper(),
        # Passed unchanged so that errors quote the dictionary's own spelling.
        numpy_format=_numpy_format(parts[1], width, dictionary_path, line_number),
        # DCB positions are one-based; NumPy byte offsets are zero-based.
        offset=start_byte - 1,
        width=width,
        description=parts[9] if len(parts) > 9 else "",
        display_name=parts[12] or None if len(parts) > 12 else None,
    )


def _decode_dictionary(dictionary_path: Path) -> str:
    """Decode a dictionary, falling back to TransCAD's Windows code page.

    Dictionaries written by TransCAD on Windows carry field descriptions in
    cp1252, which is not valid UTF-8. Descriptions are discarded, so undecodable
    bytes are replaced rather than raised on.
    """
    raw = dictionary_path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        logger.debug("%s is not UTF-8; decoding it as cp1252", dictionary_path)
        return raw.decode("cp1252", errors="replace")


def _read_dictionary(dictionary_path: Path) -> tuple[int, list[_Column]]:
    logger.debug(f"Reading TransCAD dictionary {dictionary_path}")
    lines = _decode_dictionary(dictionary_path).splitlines()
    if len(lines) < 3:
        raise TranscadDictionaryError(
            f"{dictionary_path} is too short to be a valid dictionary"
        )

    try:
        record_width = int(lines[1].split()[0])
    except (IndexError, ValueError) as exc:
        raise TranscadDictionaryError(
            f"{dictionary_path}: could not read record width from line 2"
        ) from exc
    if record_width <= 0:
        raise TranscadDictionaryError(
            f"{dictionary_path}: record width must be positive"
        )

    columns = [
        _parse_column(line, line_number, dictionary_path)
        for line_number, line in enumerate(lines[2:], start=3)
        if line.strip()
    ]
    if not columns:
        raise TranscadDictionaryError(f"{dictionary_path} declares no fields")
    if len({column.name for column in columns}) != len(columns):
        raise TranscadDictionaryError(
            f"{dictionary_path} contains duplicate field names"
        )

    columns.sort(key=lambda column: column.offset)
    previous_end = 0
    for column in columns:
        if column.offset < previous_end:
            raise TranscadDictionaryError(
                f"{dictionary_path}: field {column.name!r} overlaps the previous field"
            )
        if column.offset + column.width > record_width:
            raise TranscadDictionaryError(
                f"{dictionary_path}: field {column.name!r} extends beyond the {record_width}-byte record"
            )
        previous_end = column.offset + column.width

    return record_width, columns


def _record_dtype(record_width: int, columns: list[_Column]) -> np.dtype:
    return np.dtype(
        cast(
            Any,
            {
                "names": [column.name for column in columns],
                "formats": [column.numpy_format for column in columns],
                "offsets": [column.offset for column in columns],
                "itemsize": record_width,
            },
        )
    )


def _read_records(
    binary_path: Path, record_width: int, columns: list[_Column]
) -> np.ndarray:
    raw_records = np.fromfile(binary_path, dtype=np.dtype((np.void, record_width)))
    record_bytes = raw_records.view(np.uint8).reshape(-1, record_width)
    marker_width = min(record_width, len(_DELETED_RECORD_MARKER))
    marker = np.frombuffer(_DELETED_RECORD_MARKER, dtype=np.uint8, count=marker_width)
    deleted = np.all(record_bytes[:, :marker_width] == marker, axis=1)
    # Boolean indexing copies the whole file, so skip it for the usual case of a
    # table with nothing deleted.
    kept = raw_records[~deleted] if deleted.any() else raw_records
    return kept.view(_record_dtype(record_width, columns)).reshape(-1)


def _to_dataframe(data: np.ndarray, columns: list[_Column]) -> pd.DataFrame:
    frame = pd.DataFrame(data)
    for field in columns:
        values = data[field.name]
        if field.transcad_type == "C":
            # readChar() stops at the first NUL, trimws() trims these four ASCII
            # whitespace characters, and caliperR maps an empty result to NA.
            # Bytes remain bytes because the DCB does not declare an encoding.
            frame[field.name] = pd.Series(
                [
                    stripped
                    if (stripped := bytes(value).split(b"\0", 1)[0].strip(b" \t\r\n"))
                    else pd.NA
                    for value in values
                ],
                dtype=object,
            )
            continue

        missing_value = _TRANSCAD_MISSING_VALUES[field.transcad_type]
        if field.transcad_type in _PANDAS_INTEGER_DTYPES:
            nullable_dtype = _PANDAS_INTEGER_DTYPES[field.transcad_type]
            column = pd.array(values, dtype=nullable_dtype)
            column[values == missing_value] = pd.NA
            frame[field.name] = column
        else:
            column = values.copy()
            column[column == missing_value] = np.nan
            frame[field.name] = column

    # pandas has no Hmisc-style per-column labels. A name-keyed mapping in
    # DataFrame.attrs preserves the same DCB metadata without changing values.
    frame.attrs["field_descriptions"] = {
        field.name: field.description for field in columns
    }
    display_names = {
        field.name: field.display_name
        for field in columns
        if field.display_name is not None
    }
    if display_names:
        frame.attrs["display_names"] = display_names

    return frame


# @function_logging(
#     "Reading TransCAD binary table {path}", logger=logger, level=logging.DEBUG
# )
def read_transcad_binary(
    path: PathLike, dictionary_path: PathLike | None = None
) -> pd.DataFrame:
    """Read a TransCAD ``.bin``/``.dcb`` table into a DataFrame.

    ``path`` may be the binary path, dictionary path, or their shared stem. Only
    a ``.bin`` or ``.dcb`` extension is stripped, so a stem may contain dots.
    Pass ``dictionary_path`` when the dictionary does not share the binary's
    directory and stem.
    """
    logger.debug(f"Reading TransCAD binary table {path}")
    logger.warning(
        "Results for %s may be incorrect! The FFB format is reverse engineered, so always verify against "
        "TransCAD. Prefer any other source, such as a table TransCAD exported to a common format.",
        path,
    )
    stem = _table_stem(path)
    binary_path = _companion_path(stem, ".bin")
    resolved_dictionary = (
        Path(dictionary_path).expanduser()
        if dictionary_path is not None
        else _companion_path(stem, ".dcb")
    )
    if not binary_path.is_file():
        raise FileNotFoundError(f"TransCAD binary file not found: {binary_path}")
    if not resolved_dictionary.is_file():
        raise FileNotFoundError(
            f"TransCAD dictionary file not found: {resolved_dictionary}"
        )

    record_width, columns = _read_dictionary(resolved_dictionary)
    file_size = binary_path.stat().st_size
    if file_size % record_width:
        raise TranscadBinaryError(
            f"{binary_path} size ({file_size} bytes) is not a multiple of its {record_width}-byte record width"
        )

    data = _read_records(binary_path, record_width, columns)
    return _to_dataframe(data, columns)
