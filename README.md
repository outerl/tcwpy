# TransCAD binary table reader

`read_binary.py` reads a TransCAD fixed-format binary (`.bin`) table using its
matching data dictionary (`.dcb`) and returns a pandas `DataFrame`. It is a
Python port of Caliper Corporation's
[`caliperR::read_bin()`](https://github.com/Caliper-Corporation/caliperR/blob/master/R/convert_bin.R),
with NumPy used to decode records directly.

The FFB format is not publicly specified. This implementation follows `caliperR` and independently observed TransCAD files, but should still be checked against a TransCAD export before its output is used for production decisions.


## Usage

```python
from pathlib import Path

from tcwpy import read_transcad_binary

table = read_transcad_binary(Path("inputs/trips.bin"))
```

The first argument may be the `.bin` path, the `.dcb` path, or their shared
stem. By default, the files must share a directory and stem. Alternatively, the two paths may be passed separately:

```python
table = read_transcad_binary(
    "inputs/trips.bin",
    dictionary_path="dictionaries/trips_dictionary.dcb",
)
```

This reader only converts the following DCB types: `I`, `S`, `R`, `F`, and `C`. The `Date`, `Time`, and `DateTime` fields are not implemented adn will raise an error if encountered.

## Licence

This work is derivative of
[`caliperR`](https://github.com/Caliper-Corporation/caliperR) and
[`tcwpy`](https://github.com/pedrocamargo/tcwpy). Both are licensed under the
Apache License 2.0, so this directory retains the same licence in
[`LICENSE`](LICENSE). Apache-2.0 is a permissive licence, not a copyleft licence.
