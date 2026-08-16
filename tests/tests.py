import hashlib
import struct
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from transcad_binary_parser.read_binary import (
    TranscadBinaryError,
    TranscadDictionaryError,
    read_transcad_binary,
)

_DELETED_RECORD_MARKER = bytes.fromhex(
    "91 8b 4a 5c bc db 4f 14 63 23 7f 78 a6 95 0d 27"
)
_FLOAT32_MISSING = float(np.finfo(np.float32).min)
_FLOAT64_MISSING = float(np.finfo(np.float64).min)


def _write_table(
    tmp_path: Path,
    name: str = "table",
    dictionary_lines: list[str] | None = None,
    records: Sequence[tuple[int, float]] = ((1, 3.14), (2, _FLOAT32_MISSING)),
) -> Path:
    if dictionary_lines is None:
        dictionary_lines = [
            '"table",\n',
            "8 2\n",
            '"ID1",I,1,4\n',
            '"Val",F,5,4\n',
        ]
    stem = tmp_path / name
    stem.with_suffix(".dcb").write_text("".join(dictionary_lines))
    with stem.with_suffix(".bin").open("wb") as binary_file:
        for identifier, value in records:
            binary_file.write(struct.pack("<if", identifier, value))
    return stem


def test_reads_fields_records_and_maps_missing_values_to_na(tmp_path: Path) -> None:
    stem = _write_table(tmp_path)

    result = read_transcad_binary(stem)

    assert result["ID1"].tolist() == [1, 2]
    assert result["Val"].iloc[0] == pytest.approx(3.14)
    assert np.isnan(result["Val"].iloc[1])


_TOY_TABLE_DICTIONARY = (
    "\n"
    "52\n"
    '"first",I,1,4,0,8,0,,"","first field",,"Sum","field_a"\n'
    '"second",C,5,16,0,16,0,,"","second field",,"Copy","field_b"\n'
    '"third",C,21,16,0,16,0,,"","third field",,"Copy","field_c"\n'
    '"fourth",R,37,8,0,10,2,,"","fourth field",,"Sum","field_d"\n'
    '"fifth",R,45,8,0,10,2,,"","fifth field",,"Sum","field_e"\n'
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _toy_table_record(first: int, second: bytes, fourth: float) -> bytes:
    """Build one 52-byte record in caliperR's toy_table layout (space padded)."""
    return (
        struct.pack("<i", first)
        + second.ljust(16)
        + b" " * 16
        + struct.pack("<dd", fourth, _FLOAT64_MISSING)
    )


def _write_toy_table(tmp_path: Path, dictionary_suffix: str = ".dcb") -> Path:
    stem = tmp_path / "toy_table"
    stem.with_suffix(dictionary_suffix).write_text(_TOY_TABLE_DICTIONARY)
    stem.with_suffix(".bin").write_bytes(
        b"".join(
            [
                _toy_table_record(1, b"a", 1.27),
                _toy_table_record(2, b"b", 2.0),
                _toy_table_record(3, b"", 7.8),
                _toy_table_record(4, b"d", 8.5),
                _toy_table_record(-2147483647, b"e", 9.0),
                _DELETED_RECORD_MARKER
                + b" " * 20
                + struct.pack("<dd", 9.0, _FLOAT64_MISSING),
                _toy_table_record(7, b"g", 3.2),
            ]
        )
    )
    return stem


def test_matches_caliperr_reference_fixture(tmp_path: Path) -> None:
    """Reproduce caliperR's toy_table.DCB/bin fixture byte for byte.

    Expectations mirror caliperR's own ``test-zzz-read-bin-no-com.R``: the second
    field's padding is trimmed, its blank third value is NA, the all-missing third
    and fifth fields keep their declared types, field descriptions are retained,
    and the deleted record is dropped, leaving six rows.
    """
    stem = _write_toy_table(tmp_path)

    # Guards that the fixture built above is still byte for byte the pair from
    # caliperR/inst/extdata/gisdk/testing/, rather than merely resembling it.
    assert (
        _sha256(stem.with_suffix(".bin"))
        == "26c551b9ac485926e21f300e0849cd3fb62d2c12a8fb02f08e1b60bcb535f31c"
    )
    assert (
        _sha256(stem.with_suffix(".dcb"))
        == "6eeb2436d44204ccaaa1fd5269077c055437fd72c41cb091a5b77857dbafd45b"
    )

    result = read_transcad_binary(stem)

    assert len(result) == 6
    assert result["first"].tolist()[:4] == [1, 2, 3, 4]
    assert pd.isna(result["first"].iloc[4])
    assert result["first"].iloc[5] == 7
    assert result["second"].tolist() == [b"a", b"b", pd.NA, b"d", b"e", b"g"]
    assert result["third"].isna().all()
    assert result["third"].dtype == np.dtype("object")
    assert result["fourth"].tolist() == pytest.approx([1.27, 2.0, 7.8, 8.5, 9.0, 3.2])
    assert result["fifth"].isna().all()
    assert result["fifth"].dtype == np.dtype("f8")
    assert result.attrs["field_descriptions"] == {
        "first": "first field",
        "second": "second field",
        "third": "third field",
        "fourth": "fourth field",
        "fifth": "fifth field",
    }
    assert result.attrs["display_names"] == {
        "first": "field_a",
        "second": "field_b",
        "third": "field_c",
        "fourth": "field_d",
        "fifth": "field_e",
    }


def test_finds_an_uppercase_dictionary_beside_a_lowercase_binary(
    tmp_path: Path,
) -> None:
    """caliperR's own fixture is toy_table.bin next to toy_table.DCB."""
    stem = _write_toy_table(tmp_path, dictionary_suffix=".DCB")

    assert len(read_transcad_binary(stem.with_suffix(".bin"))) == 6
    assert len(read_transcad_binary(stem.with_suffix(".DCB"))) == 6


@pytest.mark.parametrize(
    "transcad_type,width,struct_format,valid_value,missing_value,expected_dtype",
    [
        ("S", 2, "<h", -32768, -32767, "Int16"),
        ("I", 4, "<i", -2147483648, -2147483647, "Int32"),
    ],
)
def test_integer_missing_values_are_matched_exactly(
    tmp_path: Path,
    transcad_type: str,
    width: int,
    struct_format: str,
    valid_value: int,
    missing_value: int,
    expected_dtype: str,
) -> None:
    stem = tmp_path / f"integer_{width}"
    stem.with_suffix(".dcb").write_text(
        f'"table",\n{width} 1\n"Value",{transcad_type},1,{width}\n'
    )
    stem.with_suffix(".bin").write_bytes(
        struct.pack(struct_format, valid_value)
        + struct.pack(struct_format, missing_value)
    )

    result = read_transcad_binary(stem)

    assert str(result["Value"].dtype) == expected_dtype
    assert result["Value"].iloc[0] == valid_value
    assert pd.isna(result["Value"].iloc[1])


@pytest.mark.parametrize(
    "transcad_type,width,struct_format,numpy_format",
    [
        ("F", 4, "<f", "f4"),
        ("R", 8, "<d", "f8"),
    ],
)
def test_real_missing_values_are_matched_exactly(
    tmp_path: Path,
    transcad_type: str,
    width: int,
    struct_format: str,
    numpy_format: str,
) -> None:
    dtype = np.dtype(numpy_format)
    missing_value = np.finfo(dtype).min
    valid_minimum = np.nextafter(missing_value, np.inf, dtype=dtype)
    stem = tmp_path / f"real_{width}"
    stem.with_suffix(".dcb").write_text(
        f'"table",\n{width} 1\n"Value",{transcad_type},1,{width}\n'
    )
    stem.with_suffix(".bin").write_bytes(
        struct.pack(struct_format, valid_minimum)
        + struct.pack(struct_format, missing_value)
    )

    result = read_transcad_binary(stem)

    assert result["Value"].dtype == dtype
    assert result["Value"].iloc[0] == valid_minimum
    assert np.isnan(result["Value"].iloc[1])


@pytest.mark.parametrize(
    "transcad_type,width",
    [
        ("I", 1),
        ("I", 2),
        ("I", 8),
        ("S", 1),
        ("S", 4),
        ("R", 4),
        ("F", 8),
    ],
)
def test_rejects_widths_that_contradict_the_transcad_type(
    tmp_path: Path,
    transcad_type: str,
    width: int,
) -> None:
    stem = tmp_path / f"{transcad_type}_{width}"
    stem.with_suffix(".dcb").write_text(
        f'"table",\n{width} 1\n"Value",{transcad_type},1,{width}\n'
    )
    stem.with_suffix(".bin").write_bytes(b"\0" * width)

    with pytest.raises(
        TranscadDictionaryError, match=f"type {transcad_type!r} requires width"
    ):
        read_transcad_binary(stem)


def test_character_fields_are_nul_terminated_trimmed_bytes_with_empty_values_as_na(
    tmp_path: Path,
) -> None:
    stem = tmp_path / "characters"
    stem.with_suffix(".dcb").write_text('"table",\n8 1\n"Name",C,1,8\n')
    stem.with_suffix(".bin").write_bytes(b" Alpha  " + b" \t\r\n\0xy " + b"a\0ignore")

    result = read_transcad_binary(stem)

    assert result["Name"].tolist() == [b"Alpha", pd.NA, b"a"]


def test_uses_one_based_start_bytes_and_honours_padding(tmp_path: Path) -> None:
    stem = tmp_path / "padded"
    stem.with_suffix(".dcb").write_text(
        "".join(
            [
                '"padded",\n',
                "20 2\n",
                '"Val",F,13,4\n',
                '"ID1",I,5,4\n',
            ]
        )
    )
    with stem.with_suffix(".bin").open("wb") as binary_file:
        for identifier, value in ((1, 1.5), (2, 2.5)):
            binary_file.write(
                b"\0" * 4
                + struct.pack("<i", identifier)
                + b"\0" * 4
                + struct.pack("<f", value)
                + b"\0" * 4
            )

    result = read_transcad_binary(stem)

    assert list(result.columns) == ["ID1", "Val"]
    assert result["ID1"].tolist() == [1, 2]
    assert result["Val"].tolist() == [1.5, 2.5]


def test_csv_dictionary_parsing_preserves_commas_in_quoted_field_names(
    tmp_path: Path,
) -> None:
    stem = tmp_path / "quoted"
    stem.with_suffix(".dcb").write_text(
        '"table",\n4 1\n"Value, total",I,1,4,0,8,0,,"","a, description"\n'
    )
    stem.with_suffix(".bin").write_bytes(struct.pack("<i", 42))

    result = read_transcad_binary(stem)

    assert result.columns.tolist() == ["Value, total"]
    assert result.iloc[0, 0] == 42


def test_removes_only_records_that_start_with_the_deleted_marker(
    tmp_path: Path,
) -> None:
    stem = tmp_path / "deleted"
    stem.with_suffix(".dcb").write_text('"table",\n20 1\n"Value",C,1,20\n')
    valid_with_internal_marker = b"x" + _DELETED_RECORD_MARKER + b" " * 3
    stem.with_suffix(".bin").write_bytes(
        b"keep".ljust(20)
        + _DELETED_RECORD_MARKER
        + b"gone"
        + valid_with_internal_marker
    )

    result = read_transcad_binary(stem)

    assert len(result) == 2
    assert result["Value"].iloc[0] == b"keep"
    assert result["Value"].iloc[1].startswith(b"x" + _DELETED_RECORD_MARKER)


def test_deleted_marker_is_truncated_for_records_narrower_than_16_bytes(
    tmp_path: Path,
) -> None:
    stem = tmp_path / "narrow_deleted"
    stem.with_suffix(".dcb").write_text('"table",\n4 1\n"Value",I,1,4\n')
    stem.with_suffix(".bin").write_bytes(
        struct.pack("<i", 1) + _DELETED_RECORD_MARKER[:4] + struct.pack("<i", 2)
    )

    assert read_transcad_binary(stem)["Value"].tolist() == [1, 2]


@pytest.mark.parametrize("path_kind", ["stem", "binary", "dictionary", "string"])
def test_accepts_flexible_paths(tmp_path: Path, path_kind: str) -> None:
    stem = _write_table(tmp_path)
    paths = {
        "stem": stem,
        "binary": stem.with_suffix(".bin"),
        "dictionary": stem.with_suffix(".dcb"),
        "string": str(stem),
    }

    assert len(read_transcad_binary(paths[path_kind])) == 2


def test_only_bin_and_dcb_extensions_are_stripped_from_a_stem(tmp_path: Path) -> None:
    """A stem such as ``trips.2016`` must not be truncated to ``trips``."""
    (tmp_path / "trips.2016.dcb").write_text('"table",\n4 1\n"ID1",I,1,4\n')
    (tmp_path / "trips.2016.bin").write_bytes(struct.pack("<ii", 1, 2))
    # A decoy that a truncated stem would read instead.
    _write_table(tmp_path, name="trips", records=((9, 9.0),))

    assert read_transcad_binary(tmp_path / "trips.2016")["ID1"].tolist() == [1, 2]
    assert read_transcad_binary(tmp_path / "trips.2016.bin")["ID1"].tolist() == [1, 2]


def test_reads_a_dictionary_that_is_not_utf8(tmp_path: Path) -> None:
    """TransCAD writes descriptions on Windows, so they are usually cp1252."""
    stem = tmp_path / "cp1252"
    stem.with_suffix(".dcb").write_bytes(
        '"table",\n4 1\n"Value",I,1,4,0,8,0,,"","Temperature in °C — site"\n'.encode(
            "cp1252"
        )
    )
    stem.with_suffix(".bin").write_bytes(struct.pack("<i", 42))

    result = read_transcad_binary(stem)

    assert result.columns.tolist() == ["Value"]
    assert result.iloc[0, 0] == 42


@pytest.mark.parametrize("transcad_type", ["Date", "Time", "DateTime"])
def test_rejects_date_and_time_fields_whose_encoding_is_unverified(
    tmp_path: Path, transcad_type: str
) -> None:
    stem = tmp_path / "dated"
    stem.with_suffix(".dcb").write_text(f'"table",\n8 1\n"When",{transcad_type},1,8\n')
    stem.with_suffix(".bin").write_bytes(b"\0" * 8)

    with pytest.raises(TranscadDictionaryError, match=f"{transcad_type}.*unverified"):
        read_transcad_binary(stem)


def test_accepts_an_explicit_dictionary_path(tmp_path: Path) -> None:
    stem = _write_table(tmp_path)
    dictionary_path = tmp_path / "renamed.dcb"
    stem.with_suffix(".dcb").rename(dictionary_path)

    assert len(read_transcad_binary(stem, dictionary_path=dictionary_path)) == 2


def test_accepts_an_empty_binary_as_a_zero_row_table(tmp_path: Path) -> None:
    stem = _write_table(tmp_path)
    stem.with_suffix(".bin").write_bytes(b"")

    result = read_transcad_binary(stem)

    assert result.empty
    assert result.columns.tolist() == ["ID1", "Val"]
    assert str(result["ID1"].dtype) == "Int32"
    assert result["Val"].dtype == np.dtype("f4")


def test_rejects_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="binary file not found"):
        read_transcad_binary(tmp_path / "missing")

    (tmp_path / "table.bin").write_bytes(b"\0" * 8)
    with pytest.raises(FileNotFoundError, match="dictionary file not found"):
        read_transcad_binary(tmp_path / "table")


@pytest.mark.parametrize(
    "dictionary_lines,error",
    [
        (['"table",\n', "8 2\n"], "too short"),
        (['"table",\n', "8 2\n", "\n"], "declares no fields"),
        (['"table",\n', "not-a-width\n", '"ID1",I,1,4\n'], "record width"),
        (['"table",\n', "0 1\n", '"ID1",I,1,4\n'], "must be positive"),
        (['"table",\n', "8 1\n", '"ID1",Z,1,4\n'], "unsupported field type"),
        (['"table",\n', "8 1\n", '"ID1",I,1,8\n'], "type 'I' requires width"),
        (['"table",\n', "2 1\n", '"Value",R,1,2\n'], "type 'R' requires width"),
        (['"table",\n', "8 1\n", '"",I,1,4\n'], "field name is empty"),
        (['"table",\n', "8 1\n", '"ID1",I,zero,4\n'], "must be integers"),
        (
            ['"table",\n', "8 1\n", '"ID1",I,0,4\n'],
            "start byte and width must be positive",
        ),
        (
            ['"table",\n', "8 1\n", '"ID1",I,1,0\n'],
            "start byte and width must be positive",
        ),
        (
            ['"table",\n', "8 2\n", '"ID1",I,1,4\n', '"ID1",F,5,4\n'],
            "duplicate field names",
        ),
        (
            ['"table",\n', "8 2\n", '"ID1",I,1,4\n', '"Val",F,3,4\n'],
            "overlaps",
        ),
        (['"table",\n', "8 1\n", '"ID1",I,7,4\n'], "extends beyond"),
        (['"table",\n', "8 1\n", '"unterminated,I,1,4\n'], "malformed comma-separated"),
    ],
)
def test_rejects_invalid_dictionaries(
    tmp_path: Path, dictionary_lines: list[str], error: str
) -> None:
    stem = _write_table(tmp_path, dictionary_lines=dictionary_lines)

    with pytest.raises(TranscadDictionaryError, match=error):
        read_transcad_binary(stem)


def test_rejects_partial_binary_records(tmp_path: Path) -> None:
    stem = _write_table(tmp_path)
    stem.with_suffix(".bin").write_bytes(b"\0" * 9)

    with pytest.raises(TranscadBinaryError, match="not a multiple"):
        read_transcad_binary(stem)
