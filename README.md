# TransCAD binary table reader

`read_binary.py` reads a TransCAD fixed-format binary (`.bin`) table using its
matching data dictionary (`.dcb`) and returns a pandas `DataFrame`. It is a
Python port of Caliper Corporation's
[`caliperR::read_bin()`](https://github.com/Caliper-Corporation/caliperR/blob/master/R/convert_bin.R),
with NumPy used to decode records directly.

> [!WARNING]
> The FFB format is not publicly specified. This implementation follows
> `caliperR`, its reference fixture, and independently observed TransCAD files,
> but should still be checked against a TransCAD export before its output is
> used for production decisions.

## Usage

```python
from pathlib import Path

from tcwpy import read_transcad_binary

table = read_transcad_binary(Path("inputs/trips.bin"))
```

The first argument may be the `.bin` path, the `.dcb` path, or their shared
stem. By default, the files must share a directory and stem:

```python
table = read_transcad_binary("inputs/trips")
```

Only a `.bin` or `.dcb` extension is removed, so a stem may contain dots
(`inputs/trips.2016` reads `inputs/trips.2016.bin`). Both extensions are matched
in lower and upper case, because TransCAD writes the pair inconsistently — the
`caliperR` fixture itself is `toy_table.bin` next to `toy_table.DCB`.

Pass the dictionary explicitly when it has a different name or location:

```python
table = read_transcad_binary(
    "inputs/trips.bin",
    dictionary_path="dictionaries/trips_dictionary.dcb",
)
```

An empty `.bin` is a valid zero-row table. A missing file, malformed dictionary,
overlapping/out-of-bounds field, or partial final record raises an exception
instead of returning partial or shifted data.

## Data conversion

Each numeric DCB type has a fixed storage width, matching `caliperR`'s type
conversion and missing-value rules:

| DCB type | Width | On-disk value | pandas dtype |
| --- | ---: | --- | --- |
| `I` | 4 | little-endian signed integer | `Int32` |
| `S` | 2 | little-endian signed integer | `Int16` |
| `R` | 8 | little-endian IEEE float | `float64` |
| `F` | 4 | little-endian IEEE float | `float32` |
| `C` | any positive width | fixed-width character bytes | `object` (`bytes`/`pd.NA`) |

A numeric field whose width contradicts its type is rejected rather than
silently decoded as another TransCAD type.

`Date`, `Time`, and `DateTime` fields are rejected. `caliperR` reads them as
integers or doubles and then applies a Unix epoch origin, but TransCAD's
on-disk encoding for them is undocumented, so a wrong date is more likely than a
right one. Export such tables from TransCAD instead.

Integer columns use pandas nullable integer dtypes. TransCAD's exact missing
sentinels (`-32767` and `-2147483647`) become `pd.NA`; the adjacent representable
values are left untouched. The exact four- and eight-byte float sentinels become
`NaN`.


Character fields remain `bytes`, because the DCB does not identify a character
encoding. Reading stops at the first NUL byte, leading/trailing spaces, tabs,
carriage returns, and newlines are trimmed, and an empty result becomes
`pd.NA`. Decode a known encoding explicitly after reading, for example:

```python
table["Name"] = table["Name"].map(
    lambda value: value.decode("cp1252") if isinstance(value, bytes) else value
)
```

## Implementation notes

- DCB start-byte positions are **one-based**. They are converted to zero-based
  NumPy offsets, so leading, internal, and trailing record padding is retained.
  `caliperR` ignores the start byte and instead lays fields out end to end from
  their widths, which shifts every field after a gap.
- DCB field rows are parsed as CSV; quoted names containing commas are valid.
- The dictionary is decoded as UTF-8, then as cp1252 if that fails. TransCAD
  commonly writes descriptions in the Windows code page.
- Fields are returned in physical start-byte order, matching `caliperR`.
- Deleted records are omitted when a **record** begins with TransCAD's 16-byte
  deletion marker. For records shorter than 16 bytes, the corresponding marker
  prefix is used. `caliperR` searches the whole file for 16 consecutive bytes
  drawn from the marker's byte set, in any order, so it can delete live records.
- The first token on DCB line 2 is the record width. Additional text on that
  line (for example, `binary`) is permitted.
- Field descriptions are exposed as `table.attrs["field_descriptions"]`.
  Thirteen-column dictionaries also expose non-empty display names as
  `table.attrs["display_names"]`. These name-keyed mappings are the pandas
  analogue of `caliperR`'s `Hmisc` labels and `returnDnames` metadata.

## Licence and provenance

This work is derivative of
[`caliperR`](https://github.com/Caliper-Corporation/caliperR) and
[`tcwpy`](https://github.com/pedrocamargo/tcwpy). Both are licensed under the
Apache License 2.0, so this directory retains the same licence in
[`LICENSE`](LICENSE). Apache-2.0 is a permissive licence, not a copyleft licence.
