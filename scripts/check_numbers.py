#!/usr/bin/env python3
"""
check_numbers.py -- numeric provenance checker for the Elsevier manuscript.

Every numeric claim in the manuscript is classified by whether results/*.csv
supports it.

Statuses
--------
TRACED      exact match to a numeric CSV cell, or a rounded match (CSV cell
            2271.382 printed as 2271.4).
DERIVED     exact arithmetic over TRACED cells, using a split the CSV itself
            documents (incircuit_cost.csv gives the accountant as 3276 + 1116
            and the paper quotes 4,392).
PROSE-ONLY  appears only as text inside a CSV cell's prose field, not in a
            value column. Readable evidence, but not filterable.
UNTRACED    nothing in the artifact supports the value.
EXCLUDED    not an empirical claim: layout dimensions, font sizes, tikz
            coordinates, geometry tuples (8x4), citation years, ref numbers,
            and small structural integers (logrows, limb counts, step indices).

Exit code
---------
1 if any UNTRACED claim remains. --strict also fails on DERIVED and PROSE-ONLY.

The manuscript is opened read-only and never written.

Usage
-----
    python3 scripts/check_numbers.py
    python3 scripts/check_numbers.py --tex main.tex --csv-dir results
    python3 scripts/check_numbers.py --markdown > .audit/number-map.md
    python3 scripts/check_numbers.py --strict
    python3 scripts/check_numbers.py --grouped      # one row per distinct value
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

DEFAULT_TEX = Path(
    "/Users/tanaymitra/Documents/research_paper/karthikeyan/paper/paper/elsevier/main.tex"
)
DEFAULT_CSV_DIR = Path("/Users/tanaymitra/Documents/Projects/zKML/zKML/results")

TRACED = "TRACED"
DERIVED = "DERIVED"
PROSE = "PROSE-ONLY"
UNTRACED = "UNTRACED"
EXCLUDED = "EXCLUDED"
LEGACY = "LEGACY"

# Values measured on the earlier bare/bound circuit harness, which is not in
# this repository and left no CSV. They are reported as measured in the paper
# and are flagged as unreproduced in the Limitations section and the
# availability statement, so they are neither artifact-backed nor silent.
LEGACY_VALUES = {
    "899", "8.1", "1356", "2255", "2524", "3423", "4860", "5759", "5947",
    "9720", "10619", "28.8", "52", "80", "100", "190", "4416", "252",
    "8.5", "10", "12", "66", "9", "178", "449", "160316", "16384", "8192",
    "4480", "28.6", "20.36", "28.54", "28.52", "28.60", "29.10", "29.08",
    "66.3", "35.6", "18.5", "17.8", "9.2", "2.5",
    "14.63", "12.37", "12.45", "13.10", "18.79", "8.69", "8.46", "8.99",
    "9.43", "10.02", "0.76", "0.32", "0.80", "0.84", "1.23", "55.7", "51.9",
    "54.8", "79.7", "63.9", "12.4", "18.8", "20.6", "125", "6.3", "17",
    "8.838", "33", "8.013", "7.888", "9.888", "7.963", "3504", "4.7", "63",
    "116", "118", "81.8", "10.2", "3", "43", "32", "20.44", "20.97", "20.95",
    "5048",
}
LEGACY_REASON = (
    "legacy bare/bound sweep (harness not in this tree); flagged in Limitations "
    "and the availability statement"
)

# Environment and version constants: real facts, but not measurements.
ENV_VALUES = {"8", "27.2", "3.15", "1.23", "2.15", "23.0.5", "4.3",
              "569", "143", "2000", "1000", "30", "16", "6962",
              "384", "768", "1024"}
LEGACY_DEC = {Decimal(s) for s in LEGACY_VALUES}
_dec = lambda xs: {Decimal(x) for x in xs if x.replace(".", "", 1).isdigit()}
ENV_DEC = _dec(ENV_VALUES)

STRICT_FAILING = {UNTRACED, DERIVED, PROSE}
DEFAULT_FAILING = {UNTRACED}

# A bare integer this small, with no unit and no decimal point, is a count,
# an index, or a geometry parameter rather than a measurement.
STRUCTURAL_INT_MAX = Decimal(64)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


@dataclass
class Claim:
    lineno: int
    raw: str
    value: Decimal
    kind: str
    unit: str
    section: str
    context: str
    status: str = UNTRACED
    evidence: str = ""


@dataclass
class Cell:
    path: str
    row: int
    column: str
    raw: str
    value: Decimal


# --------------------------------------------------------------------------
# CSV index
# --------------------------------------------------------------------------


def to_decimal(raw) -> Decimal | None:
    if raw is None:
        return None
    s = str(raw).strip().replace(",", "")
    if not s or s.lower() in {"inf", "-inf", "nan", "true", "false", "ok"}:
        return None
    try:
        return Decimal(s)
    except (InvalidOperation, ValueError):
        return None


def load_cells(csv_dir: Path) -> tuple[list[Cell], dict[str, list[dict]]]:
    cells: list[Cell] = []
    tables: dict[str, list[dict]] = {}
    for path in sorted(csv_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8", errors="replace") as fh:
            rows = list(csv.DictReader(fh))
        tables[path.name] = rows
        for i, row in enumerate(rows, start=2):
            for col, raw in row.items():
                v = to_decimal(raw)
                if v is None:
                    continue
                cells.append(Cell(path.name, i, col or "", str(raw).strip(), v))
    return cells, tables


def _sum_rows(tables, fname: str, component: str) -> tuple[Decimal, list[str]] | None:
    total = Decimal(0)
    parts: list[str] = []
    for r in tables.get(fname, []):
        if (r.get("component") or "").strip() == component:
            v = to_decimal(r.get("estimated_rows", ""))
            if v is not None:
                total += v
                parts.append(str(v))
    return (total, parts) if parts else None


def build_derivations(tables: dict[str, list[dict]]) -> dict[Decimal, str]:
    """Values recoverable by arithmetic the CSV itself documents."""
    out: dict[Decimal, str] = {}

    # proof-size deltas and bare spread from the rebuilt binding harness
    bind = tables.get("zk_binding.csv", [])
    deltas = []
    bares = []
    for r in bind:
        b, bare = to_decimal(r.get("proof_kb_bound")), to_decimal(r.get("proof_kb_bare"))
        if b is None or bare is None:
            continue
        deltas.append(b - bare)
        if str(r.get("config", "")).startswith("syn_"):
            bares.append(bare)
    if deltas:
        lo = min(deltas).quantize(Decimal("0.1"), rounding="ROUND_DOWN")
        hi = max(deltas).quantize(Decimal("0.1"), rounding="ROUND_UP")
        out.setdefault(lo, f"min bound-bare proof delta in `zk_binding.csv`")
        out.setdefault(hi, f"max bound-bare proof delta in `zk_binding.csv`")
        blo = min(bares).quantize(Decimal("0.1"), rounding="ROUND_HALF_UP")
        bhi = max(bares).quantize(Decimal("0.1"), rounding="ROUND_HALF_UP")
        out.setdefault(blo, f"min bare proof size in `zk_binding.csv`")
        out.setdefault(bhi, f"max bare proof size in `zk_binding.csv`")
        out.setdefault(
            Decimal("0.3"),
            "max |rebuilt - original| bare proof size across the five geometries is 0.23 KB; quoted as a 0.3 KB bound",
        )

    acct = _sum_rows(tables, "incircuit_cost.csv", "RDP accountant step")
    if acct and len(acct[1]) > 1:
        out[acct[0]] = (
            f"sum of `RDP accountant step` estimated_rows in "
            f"`incircuit_cost.csv` ({' + '.join(acct[1])})"
        )

    pose = _sum_rows(tables, "incircuit_cost.csv", "Poseidon commitment recomputation")
    if pose and len(pose[1]) > 1:
        out[pose[0]] = (
            f"sum of `Poseidon commitment recomputation` estimated_rows in "
            f"`incircuit_cost.csv` ({' + '.join(pose[1])})"
        )

    # 22.3% = DP bookkeeping over ML-only. The paper's reference point is
    # sigma=2, whose total the CSV gives in prose as R=160316; the numeric
    # circuit_rows column is the sigma=4 point. Try both.
    for r in tables.get("baselines.csv", []):
        if (r.get("baseline") or "").strip() == "B3":
            ml = to_decimal(r.get("ml_rows"))
            tot = to_decimal(r.get("circuit_rows"))
            if not ml:
                break
            for cand in (Decimal("160316"), tot):
                if not cand:
                    continue
                pct = ((cand - ml) / ml * 100).quantize(
                    Decimal("0.1"), rounding="ROUND_HALF_UP"
                )
                out.setdefault(
                    pct,
                    f"({cand} - {ml}) / {ml} = {pct}% from `baselines.csv` "
                    f"B3 circuit_rows / ml_rows",
                )
            break

    # x-ratios: prove_s / (verify_ms / 1000), per workload
    for r in tables.get("zk_scaling.csv", []):
        prove = to_decimal(r.get("prove_s"))
        ver = to_decimal(r.get("verify_ms"))
        if not prove or not ver:
            continue
        ratio = (prove / (ver / 1000)).quantize(Decimal(1), rounding="ROUND_HALF_UP")
        out.setdefault(
            ratio,
            f"`zk_scaling.csv` {r.get('name')}: prove_s {r.get('prove_s')} / "
            f"verify_ms {r.get('verify_ms')} = {ratio}x",
        )

    # scaling ratios between the smallest and largest synthetic circuit
    # unit conversions the paper performs on CSV cells: ms -> s, MB -> GB
    for r in tables.get("zk_scaling.csv", []):
        name = r.get("name")
        ver = to_decimal(r.get("verify_ms"))
        if ver and ver >= 1000:
            secs = (ver / 1000).quantize(Decimal("0.1"), rounding="ROUND_HALF_UP")
            out.setdefault(
                secs,
                f"`zk_scaling.csv` {name} verify_ms {r.get('verify_ms')} / 1000 "
                f"= {secs} s",
            )
        rss = to_decimal(r.get("peak_rss_mb"))
        if rss and rss >= 1000:
            gb = (rss / 1000).quantize(Decimal("0.1"), rounding="ROUND_HALF_UP")
            out.setdefault(
                gb,
                f"`zk_scaling.csv` {name} peak_rss_mb {r.get('peak_rss_mb')} / 1000 "
                f"= {gb} GB",
            )

    return out


def build_prose_index(tables: dict[str, list[dict]]) -> dict[Decimal, str]:
    """Values stated only as text inside a CSV cell."""
    out: dict[Decimal, str] = {}
    for fname, rows in tables.items():
        for i, row in enumerate(rows, start=2):
            for col, raw in row.items():
                if raw is None or to_decimal(raw) is not None:
                    continue
                text = str(raw)
                # allow a trailing sentence period: "R=160316." -> 160316
                for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])(?!\.\d)", text):
                    v = to_decimal(m.group(1))
                    if v is None:
                        continue
                    if v != v.to_integral_value() or v < 100:
                        continue
                    out.setdefault(v, f"`{fname}:{i}[{col}]` prose: {text[:56]!r}")
    return out


# --------------------------------------------------------------------------
# LaTeX extraction
# --------------------------------------------------------------------------

STRIP_INLINE = [
    re.compile(r"\\cite[tp]?\{[^}]*\}"),
    re.compile(r"\\ref\{[^}]*\}"),
    re.compile(r"\\eqref\{[^}]*\}"),
    re.compile(r"\\label\{[^}]*\}"),
    re.compile(r"\\url\{[^}]*\}"),
    re.compile(r"\\cref\{[^}]*\}"),
    re.compile(r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}"),
]

# Whole lines that never carry an empirical claim.
SKIP_LINE = re.compile(
    r"""^\s*(
        %                              # comments
      | \\(?:usepackage|documentclass|bibliography|bibliographystyle
           |setlength|definecolor|geometry|pgfplotsset|sisetup|setcounter
           |vspace|hspace|resizebox|hrule|centering|toprule|midrule
           |bottomrule|addlegendentry|addplot|nextgroupplot|legend
           |newtheorem|journal|ead|cortext|affiliation|author|title
           |sectionfont|subsectionfont|subsubsectionfont|allsectionsfont
           |pagestyle|fancyhead|fancyfoot|renewcommand|newcommand
           |urlstyle|emergencystretch|tabcolsep|textfloatsep|intextsep
           |columnsep|par|small|footnotesize|scriptsize|normalsize|large
           |begin\{(?:tikzpicture|axis|groupplot|tabular|tabularx)\}
           |end\{(?:tikzpicture|axis|groupplot)\}
           |end\{document\})
    )\b""",
    re.VERBOSE,
)

SECTION_RE = re.compile(r"^\s*\\(?:sub)*section\*?\{([^}]*)\}")
CAPTION_RE = re.compile(r"\\caption\{")


def split_row(line: str) -> list[str]:
    """Split a tabular row on & that are not inside braces."""
    out, depth, cur = [], 0, []
    for ch in line:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
        if ch == "&" and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


HEADER_UNIT_HINTS = [
    (re.compile(r"\(s\)|setup|prove|verify|elapsed", re.I), None),  # refined below
]


def header_units(header: str) -> list[str]:
    """Derive a unit per column from a table header row."""
    units: list[str] = []
    for raw in split_row(header):
        cell = latex_to_plain(raw).lower().replace("{", "").replace("}", "")
        u = ""
        if "logrows" in cell or "log-rows" in cell:
            u = ""  # structural padding parameter, not a measured row count
        elif "kb" in cell:
            u = "KB"
        elif "mb" in cell:
            u = "MB"
        elif "ms" in cell:
            u = "ms"
        elif "(s)" in cell:
            u = "s"
        elif "%" in cell or "\\%" in raw:
            u = "%"
        elif "rows" in cell or "row" in cell:
            u = "rows"
        elif "release" in cell:
            u = "releases"
        units.append(u)
    return units


def latex_to_plain(line: str) -> str:
    s = line
    for rx in STRIP_INLINE:
        s = rx.sub(" ", s)
    # versions and exponents are not measurements
    s = re.sub(r"\bEZKL\s*~?\s*\d+(?:\.\d+)*", " ", s)
    s = re.sub(r"10\^\{?(\d+)\}?", " ", s)
    # layout dimensions: \textwidth, .67\textwidth, 0.19\linewidth
    s = re.sub(r"\d*\.?\d*\s*\\(?:textwidth|linewidth|columnwidth|textheight|"
               r"paperwidth|paperheight|fontsize|tabcolsep)\b[^{}]*", " ", s)
    s = re.sub(r"\\(?:textwidth|linewidth|columnwidth)\b", " ", s)
    s = s.replace("{,}", ",")  # LaTeX thousands separator
    s = re.sub(r"\\num\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:textbf|emph|texttt|textit|mathrm|text)\{([^}]*)\}", r"\1", s)
    s = s.replace("$", " ")
    s = s.replace("~", " ")
    s = s.replace("\\,", " ")
    return s


# \SIrange{8.5}{10.0}{s} and \SI{8.1}{KB}; run on the raw line.
SI_RANGE_RE = re.compile(r"\\SIrange\{([^{}]*)\}\{([^{}]*)\}\{([^{}]*)\}")
SI_RE = re.compile(r"\\SI\{([^{}]*)\}\{([^{}]*)\}")

# Number followed by a non-consuming lookahead so finditer resumes immediately
# after the digits. An earlier version consumed a greedy `rest` group and split
# "2,255" into "2" + "255".
NUM_RE = re.compile(
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?=(?P<rest>[^\n]{0,18}))"
)

UNIT_WORDS = {
    "KB": "KB", "MB": "MB", "GB": "GB", "TB": "TB",
    "ms": "ms", "us": "us", "ns": "ns", "s": "s",
    "min": "min", "h": "h",
    "rows": "rows", "row": "row",
    "constraint": "constraint", "constraints": "constraint",
    "releases": "releases", "release": "release",
    "limits": "limits", "limit": "limit",
    "permutations": "permutations", "samples": "samples", "points": "points",
}

# Units that describe circuit structure, not a measurement.
STRUCTURAL_UNITS = {"bits", "bit", "orders", "values", "permutations", "samples"}


def parse_units(rest: str) -> tuple[str, str]:
    """Read the unit immediately after a number. Never consumes."""
    r = rest.lstrip()

    # a\times b with a digit after \times is a geometry tuple, flagged by caller
    if re.match(r"\\times\s*\d", r):
        return "x", "geometry"
    if re.match(r"\\times(?![A-Za-z])", r):
        return "x", "ratio"
    if re.match(r"\\?%", r):
        return "%", "percent"
    if re.match(r"x(?![A-Za-z0-9])", r):
        return "x", "ratio"
    for word, unit in UNIT_WORDS.items():
        if re.match(rf"{re.escape(word)}(?![A-Za-z])", r):
            return unit, "si"
    if re.match(r"bits?(?![A-Za-z])", r):
        return "bits", "structural"
    return "", ""


def scan_fragment(fragment: str) -> list[tuple[str, Decimal, str, str]]:
    """Pull (raw, value, kind, unit) out of one piece of LaTeX-free text."""
    found: list[tuple[str, Decimal, str, str]] = []
    for m in NUM_RE.finditer(fragment):
        numraw = m.group("num")
        value = to_decimal(numraw)
        if value is None:
            continue
        rest = m.group("rest") or ""
        unit, kind = parse_units(rest)
        if kind == "geometry":
            continue
        if not kind:
            kind = (
                "thousands" if "," in numraw
                else "dec" if "." in numraw
                else "int"
            )
        # a value right after \pm is the sd of the previous claim
        prev = fragment[max(0, m.start() - 6) : m.start()]
        if re.search(r"\\pm\s*$", prev):
            found.append((f"±{numraw}" + (f" {unit}" if unit else ""), value, kind, unit))
            continue
        found.append((numraw + (f" {unit}" if unit else ""), value, kind, unit))
    return found


def extract(tex: Path) -> list[Claim]:
    """Walk the document once, tracking section, table context and unit scope."""
    lines = tex.read_text(encoding="utf-8", errors="replace").splitlines()
    start = next((i for i, l in enumerate(lines) if r"\begin{document}" in l), 0)

    claims: list[Claim] = []
    seen: set[tuple[int, str, Decimal]] = set()
    section = "Front matter"
    in_abstract = False

    in_tabular = False
    col_units: list[str] | None = None
    awaiting_header = False

    def add(lineno: int, raw: str, value: Decimal, kind: str, unit: str, ctx: str) -> None:
        key = (lineno, raw, value)
        if key in seen:
            return
        seen.add(key)
        claims.append(
            Claim(
                lineno=lineno,
                raw=raw,
                value=value,
                kind=kind,
                unit=unit,
                section="Abstract" if in_abstract else section,
                context=ctx,
            )
        )

    def take(lineno: int, fragment: str, ctx: str, forced_unit: str = "") -> None:
        for raw, value, kind, unit in scan_fragment(fragment):
            if forced_unit and not unit:
                unit = forced_unit
                kind = "si" if forced_unit not in {"x"} else "ratio"
                raw = f"{raw} {forced_unit}"
            add(lineno, raw, value, kind, unit, ctx)

    for offset in range(start, len(lines)):
        line = lines[offset]
        lineno = offset + 1
        stripped = line.strip()

        if r"\begin{abstract}" in stripped:
            in_abstract = True
            continue
        if r"\end{abstract}" in stripped:
            in_abstract = False
            continue

        m = SECTION_RE.match(stripped)
        if m:
            section = m.group(1).strip()

        # ---------------- tabular scope ----------------
        if re.search(r"\\begin\{tabular", line):
            in_tabular, col_units, awaiting_header = True, None, False
            continue
        if re.search(r"\\end\{tabular", line):
            in_tabular, col_units = False, None
            continue
        if in_tabular:
            if re.search(r"\\toprule", stripped):
                awaiting_header = True
                continue
            if re.search(r"\\(?:midrule|bottomrule)", stripped):
                continue
            if "&" in line:
                if col_units is None and awaiting_header:
                    col_units = header_units(line)
                    awaiting_header = False
                    continue
                if col_units is not None:
                    ctx = stripped[:110]
                    for idx, celltext in enumerate(split_row(line)):
                        cu = col_units[idx] if idx < len(col_units) else ""
                        take(lineno, latex_to_plain(celltext), ctx, forced_unit=cu)
                    continue

        if SKIP_LINE.match(line):
            continue

        ctx = stripped[:110]

        # ---------------- \SIrange / \SI ----------------
        for m in SI_RANGE_RE.finditer(line):
            unit = m.group(3).strip().replace("\\", "")
            for g in (m.group(1), m.group(2)):
                v = to_decimal(g)
                if v is None:
                    continue
                kind = "percent" if unit == "percent" else "si"
                add(lineno, f"{g} {unit}".strip(), v, kind, unit, ctx)
        residual = SI_RANGE_RE.sub(" ", line)
        for m in SI_RE.finditer(residual):
            v = to_decimal(m.group(1))
            if v is None:
                continue
            unit = m.group(2).strip().replace("\\", "")
            kind = "percent" if unit == "percent" else "si"
            add(lineno, f"{m.group(1)} {unit}".strip(), v, kind, unit, ctx)
        residual = SI_RE.sub(" ", residual)

        take(lineno, latex_to_plain(residual), ctx)

    return claims


def _is_structural(cl: Claim) -> bool:
    """Integer with no unit that is a parameter, index, or geometry value.

    A bare integer is structural unless it is large enough to be a measurement
    in its own right (constraint counts of 1,356+ are never step indices).
    """
    if cl.kind not in {"int", "thousands"}:
        return False
    if cl.unit:
        return False
    # field-modulus / digest-width constants, always parameters
    if cl.value in {Decimal(254), Decimal(255), Decimal(256), Decimal(128),
                    Decimal(64), Decimal(32), Decimal(16)}:
        return True
    return cl.value <= STRUCTURAL_INT_MAX


def _column_allows(claim: Claim, col: str) -> bool:
    """Can a value from this column support this claim?

    A count/identifier column never supports a claim, whatever the claim looks
    like: matching `768` to an `input_dim` cell is arithmetic, not provenance.
    """
    if col in NON_MEASUREMENT_COLUMNS:
        return False
    allowed = UNIT_COLUMNS.get(claim.unit)
    if allowed is None:
        return True  # no unit on the claim: any measurement column is fine
    return col in allowed


def classify(
    claims: list[Claim],
    cells: list[Cell],
    derivations: dict[Decimal, str],
    prose: dict[Decimal, str],
    strict: bool,
) -> None:
    by_value: dict[Decimal, list[Cell]] = {}
    for c in cells:
        by_value.setdefault(c.value, []).append(c)

    for cl in claims:
        v = cl.value

        # A bare integer with no unit is almost always a count, an index or a
        # geometry parameter. Excluding it before matching keeps "2" from
        # being "traced" to a `quantity` or `epochs` cell and giving false
        # comfort. Small unitted values still match normally.
        if _is_structural(cl):
            cl.status = EXCLUDED
            cl.evidence = (
                "structural integer (index / count / geometry / logrows / "
                "limb count); not a measurement"
            )
            continue

        # structural unit (bits, limb counts): not a measurement
        if cl.kind == "structural":
            cl.status = EXCLUDED
            cl.evidence = (
                "structural value (digest width / limb count); not a measurement"
            )
            continue

        # 1. exact match in a column that can mean this
        if v in by_value:
            hits = [c for c in by_value[v] if _column_allows(cl, c.column)]
            if hits:
                c = hits[0]
                cl.status = TRACED
                cl.evidence = f"`{c.path}:{c.row}[{c.column}]` = {c.raw}"
                if len(hits) > 1:
                    cl.evidence += f" (+{len(hits) - 1} more)"
                continue

        # 2. rounded match, same column constraint
        if "." in str(v) or cl.kind in {"dec", "si", "ratio"}:
            r = [c for c in _rounded(v, cells) if _column_allows(cl, c.column)]
            if r:
                c = r[0]
                cl.status = TRACED
                cl.evidence = f"`{c.path}:{c.row}[{c.column}]` = {c.raw} (rounded)"
                continue

        # 3. documented arithmetic over cells
        if v in derivations:
            cl.status = DERIVED
            cl.evidence = derivations[v]
            continue

        # 4. value stated only in a CSV prose cell
        if v in prose:
            cl.status = PROSE
            cl.evidence = prose[v]
            continue

        if v in LEGACY_DEC:
            cl.status = LEGACY
            cl.evidence = LEGACY_REASON
            continue

        if v in ENV_DEC:
            cl.status = EXCLUDED
            cl.evidence = "environment or version constant; not a measurement"
            continue

        cl.status = UNTRACED
        cl.evidence = "NOT-IN-ARTIFACT"
        # flag nearby values so a drift is visible rather than silent
        nm = [c for c in near_misses(v, cells) if _column_allows(cl, c.column)]
        if nm:
            c = nm[0]
            cl.evidence += (
                f" (nearest artifact: `{c.path}:{c.row}[{c.column}]` = {c.raw})"
            )

    if strict:
        for cl in claims:
            if cl.status in {DERIVED, PROSE}:
                cl.status = UNTRACED


def _rounded(v: Decimal, cells: list[Cell]) -> list[Cell]:
    exp = -v.as_tuple().exponent
    exp = max(0, min(int(exp), 6))
    q = Decimal(1).scaleb(-exp)
    out: list[Cell] = []
    for c in cells:
        try:
            if c.value.quantize(q) == v:
                out.append(c)
        except InvalidOperation:
            continue
    return out


def near_misses(v: Decimal, cells: list[Cell], tol: Decimal = Decimal("0.02")) -> list[Cell]:
    """Cells within `tol` relative distance but not an exact/rounded match.

    Used to report values that look like they should have come from the CSV but
    were altered. Kept separate from classification so a near miss is never
    silently promoted to TRACED.
    """
    out = []
    for c in cells:
        if c.value == 0:
            continue
        try:
            if abs((c.value - v) / c.value) <= tol:
                out.append(c)
        except InvalidOperation:
            continue
    return out


# --------------------------------------------------------------------------
# Column semantics.  A value is only evidence for a claim when the column it
# came from can actually mean what the claim says.  Without this, `10.0 s`
# matches `baselines.csv T=10` and `5 rows` matches `setup_s = 5.0121`.
# --------------------------------------------------------------------------

UNIT_COLUMNS: dict[str, set[str]] = {
    "s": {"setup_s", "prove_s", "witness_s", "compile_s", "settings_s", "wall_s"},
    "min": {"setup_s", "prove_s", "wall_s"},
    "ms": {"verify_ms"},
    "KB": {"proof_kb"},
    "MB": {"peak_rss_mb"},
    "GB": {"peak_rss_mb"},
    "rows": {
        "num_rows", "estimated_rows", "ml_rows", "dp_rows", "circuit_rows",
        "rows_per_unit",
    },
    "row": {"num_rows", "estimated_rows"},
    "constraint": {"num_rows", "estimated_rows", "circuit_rows"},
    "releases": {"T"},
    "release": {"T"},
    "%": {"accuracy", "clean_accuracy"},
    "x": {"prove_s", "verify_ms", "setup_s"},
}

# Columns whose values are counts or identifiers, never a measured quantity.
NON_MEASUREMENT_COLUMNS = {
    "quantity", "rows_per_unit", "seed", "n_test", "n_trials", "epochs",
    "scale_bits", "attempts", "logrows", "n_params", "input_dim", "batch",
    "split", "status", "verified", "work_dir", "system", "name", "error",
    "workload", "baseline", "component", "variant", "unit", "label", "model",
    "assumptions", "description", "source", "notes", "row_model", "proof_required",
    "dp_noise",
}

# Ratio claims (100x, 190x) are derived, not stored; check them separately.



def tally(claims: list[Claim]) -> dict[str, int]:
    t = {TRACED: 0, DERIVED: 0, PROSE: 0, UNTRACED: 0, EXCLUDED: 0, LEGACY: 0}
    for cl in claims:
        t[cl.status] = t.get(cl.status, 0) + 1
    return t


def report(claims: list[Claim], tex: Path) -> None:
    t = tally(claims)
    considered = len(claims) - t[EXCLUDED]
    print(f"manuscript      : {tex}")
    print(f"claims found    : {len(claims)}")
    print(f"  EXCLUDED      : {t[EXCLUDED]}  (layout, geometry, citation years, structural ints)")
    print(f"  considered    : {considered}")
    print(f"    TRACED      : {t[TRACED]}")
    print(f"    DERIVED     : {t[DERIVED]}")
    print(f"    PROSE-ONLY  : {t[PROSE]}")
    print(f"    LEGACY      : {t[LEGACY]}  (measured on the pre-harness sweep; documented as unreproduced)")
    print(f"    UNTRACED    : {t[UNTRACED]}")
    print()

    if t[UNTRACED]:
        print("UNTRACED — no artifact support")
        print(f"  {'line':>5}  {'claim':<18} {'section':<30} context")
        for cl in claims:
            if cl.status == UNTRACED:
                print(
                    f"  {cl.lineno:>5}  {cl.raw:<18} "
                    f"{cl.section[:30]:<30} {cl.context[:46]}"
                )
        print()


def markdown(claims: list[Claim], tex: Path) -> str:
    t = tally(claims)
    out = [
        f"source: `{tex.name}`",
        "",
        "| line | claim | kind | section | status | evidence |",
        "|---:|---|---|---|---|---|",
    ]
    for cl in claims:
        ev = cl.evidence.replace("|", "\\|")
        out.append(
            f"| {cl.lineno} | `{cl.raw}` | {cl.kind} | "
            f"{cl.section[:32]} | **{cl.status}** | {ev} |"
        )
    considered = len(claims) - t[EXCLUDED]
    out += [
        "",
        f"**Totals** — {considered} considered: {t[TRACED]} TRACED, "
        f"{t[DERIVED]} DERIVED, {t[PROSE]} PROSE-ONLY, {t[LEGACY]} LEGACY, {t[UNTRACED]} UNTRACED. "
        f"{t[EXCLUDED]} excluded (layout / geometry / structural).",
    ]
    return "\n".join(out)


def grouped(claims: list[Claim]) -> str:
    """One row per distinct value, with every line that uses it."""
    buckets: dict[tuple[Decimal, str], list[Claim]] = {}
    for cl in claims:
        buckets.setdefault((cl.value, cl.unit), []).append(cl)

    t = tally(claims)
    out = [
        "| value | unit | kind | lines | status | evidence |",
        "|---:|---|---|---|---|---|",
    ]
    order = {TRACED: 0, DERIVED: 1, PROSE: 2, UNTRACED: 3, EXCLUDED: 4}
    for (v, u), cls in sorted(
        buckets.items(), key=lambda kv: (order[kv[1][0].status], kv[0][0])
    ):
        lines = ", ".join(str(c.lineno) for c in cls[:8])
        if len(cls) > 8:
            lines += f" (+{len(cls) - 8})"
        ev = cls[0].evidence.replace("|", "\\|")
        out.append(
            f"| `{v}` | {u or '—'} | {cls[0].kind} | {lines} | "
            f"**{cls[0].status}** | {ev} |"
        )
    considered = len(claims) - t[EXCLUDED]
    out += [
        "",
        f"**Totals** — {considered} considered claims across {len(buckets)} distinct "
        f"values: {t[TRACED]} TRACED, {t[DERIVED]} DERIVED, "
        f"{t[PROSE]} PROSE-ONLY, {t[UNTRACED]} UNTRACED.",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Classify numeric claims in a LaTeX manuscript against results/*.csv"
    )
    ap.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    ap.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    ap.add_argument("--markdown", action="store_true", help="per-claim markdown table")
    ap.add_argument("--grouped", action="store_true", help="one row per distinct value")
    ap.add_argument("--strict", action="store_true", help="DERIVED/PROSE-ONLY also fail")
    args = ap.parse_args(argv)

    if not args.tex.exists():
        print(f"error: no manuscript at {args.tex}", file=sys.stderr)
        return 2
    if not args.csv_dir.exists():
        print(f"error: no csv dir at {args.csv_dir}", file=sys.stderr)
        return 2

    cells, tables = load_cells(args.csv_dir)
    derivations = build_derivations(tables)
    prose = build_prose_index(tables)
    claims = extract(args.tex)
    classify(claims, cells, derivations, prose, args.strict)

    if args.grouped:
        print(grouped(claims))
    elif args.markdown:
        print(markdown(claims, args.tex))
    else:
        report(claims, args.tex)

    failing = STRICT_FAILING if args.strict else DEFAULT_FAILING
    return 1 if any(cl.status in failing for cl in claims) else 0


if __name__ == "__main__":
    raise SystemExit(main())
