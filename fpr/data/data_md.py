"""Write generated sections into docs/DATA.md.

Scripts 01-06 call `write_section` so that every count in DATA.md comes
from the data, not typed by hand. Each section sits between HTML markers
and is replaced in place when a script is re-run.
"""
import pandas as pd

import config


def versioned(name: str) -> str:
    """Section name for data that depends on config.DATA_VERSION (scripts 05-08, 10): v2 keeps the
    plain name, other versions get their own section '<name>_<version>' next to it."""
    return name if config.DATA_VERSION == "v2" else f"{name}_{config.DATA_VERSION}"


def write_section(name: str, markdown: str) -> None:
    """Replace (or append) the block named `name` in docs/DATA.md.

    The block is delimited by `<!-- begin:name -->` and `<!-- end:name -->`.
    """
    begin, end = f"<!-- begin:{name} -->", f"<!-- end:{name} -->"
    block = f"{begin}\n{markdown.strip()}\n{end}"
    text = config.DATA_MD.read_text() if config.DATA_MD.exists() else ""
    if begin in text and end in text:
        before = text.split(begin)[0]
        after = text.split(end)[1]
        text = before + block + after
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    config.DATA_MD.write_text(text)


def table_to_markdown(table: pd.DataFrame) -> str:
    """Render a small DataFrame as a GitHub markdown table (no extra deps)."""
    header = "| " + " | ".join(str(c) for c in table.columns) + " |"
    rule = "|" + "---|" * len(table.columns)
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join([header, rule] + rows)
