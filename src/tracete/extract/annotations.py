"""Repeat annotation parsing and a deterministic genomic interval index."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
from typing import Iterable, Iterator, TextIO


DEFAULT_TE_CLASSES = frozenset({"LINE", "SINE", "LTR", "DNA"})


def _open_text(path: str | Path) -> TextIO:
    resolved = Path(path)
    if resolved.suffix == ".gz":
        return gzip.open(resolved, "rt", encoding="utf-8")
    return resolved.open("r", encoding="utf-8")


@dataclass(frozen=True, slots=True, order=True)
class TEInterval:
    chrom: str
    start: int
    end: int
    strand: str
    subfamily: str
    family: str
    te_class: str

    def __post_init__(self) -> None:
        if not self.chrom:
            raise ValueError("chrom must be non-empty")
        if not isinstance(self.start, int) or not isinstance(self.end, int):
            raise ValueError("start and end must be integers")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("interval must satisfy 0 <= start < end")
        if self.strand not in {"+", "-", "."}:
            raise ValueError("strand must be '+', '-' or '.'")
        for name in ("subfamily", "family", "te_class"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")

    @property
    def locus_id(self) -> str:
        return (
            f"{self.chrom}:{self.start}-{self.end}:"
            f"{self.strand}:{self.subfamily}"
        )

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlap_bp(self, start: int, end: int) -> int:
        if end <= start:
            return 0
        return max(0, min(self.end, end) - max(self.start, start))


def parse_gtf_attributes(raw: str) -> dict[str, str]:
    attributes: dict[str, str] = {}
    for field in raw.strip().rstrip(";").split(";"):
        field = field.strip()
        if not field:
            continue
        parts = field.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"invalid GTF attribute: {field!r}")
        key, value = parts
        value = value.strip().strip('"')
        if not key or not value:
            raise ValueError(f"invalid GTF attribute: {field!r}")
        attributes[key] = value
    return attributes


def parse_repeat_gtf_line(
    line: str,
    *,
    allowed_classes: frozenset[str] = DEFAULT_TE_CLASSES,
) -> TEInterval | None:
    if not line.strip() or line.startswith("#"):
        return None
    fields = line.rstrip("\n").split("\t")
    if len(fields) != 9:
        raise ValueError("GTF line must contain exactly 9 columns")
    chrom, _source, feature, start, end, _score, strand, _frame, raw_attributes = fields
    if feature not in allowed_classes:
        return None
    attributes = parse_gtf_attributes(raw_attributes)
    if "gene_id" not in attributes or "transcript_id" not in attributes:
        raise ValueError("repeat GTF requires gene_id and transcript_id")
    # GTF is 1-based inclusive; internal coordinates are 0-based half-open.
    return TEInterval(
        chrom=chrom,
        start=int(start) - 1,
        end=int(end),
        strand=strand,
        subfamily=attributes["gene_id"],
        family=attributes["transcript_id"],
        te_class=feature,
    )


def parse_ucsc_rmsk_line(
    line: str,
    *,
    allowed_classes: frozenset[str] = DEFAULT_TE_CLASSES,
) -> TEInterval | None:
    """Parse a UCSC ``rmsk.txt`` row.

    Expected columns are ``bin, swScore, milliDiv, milliDel, milliIns,
    genoName, genoStart, genoEnd, genoLeft, strand, repName, repClass,
    repFamily, ...``. UCSC genomic coordinates are already 0-based
    half-open.
    """

    if not line.strip() or line.startswith("#"):
        return None
    fields = line.rstrip("\n").split("\t")
    if len(fields) < 13:
        raise ValueError("UCSC rmsk line must contain at least 13 columns")
    chrom = fields[5]
    start = int(fields[6])
    end = int(fields[7])
    strand = fields[9]
    subfamily = fields[10]
    te_class = fields[11]
    family = fields[12]
    if te_class not in allowed_classes:
        return None
    if strand == "C":
        strand = "-"
    return TEInterval(
        chrom=chrom,
        start=start,
        end=end,
        strand=strand,
        subfamily=subfamily,
        family=family,
        te_class=te_class,
    )


def iter_repeat_annotation(
    path: str | Path,
    *,
    fmt: str = "auto",
    allowed_classes: frozenset[str] = DEFAULT_TE_CLASSES,
) -> Iterator[TEInterval]:
    resolved = Path(path)
    normalized_format = fmt.lower()
    if normalized_format == "auto":
        name = resolved.name.lower()
        if name.endswith(".gtf") or name.endswith(".gtf.gz"):
            normalized_format = "gtf"
        elif "rmsk" in name or name.endswith(".txt") or name.endswith(".txt.gz"):
            normalized_format = "ucsc_rmsk"
        else:
            raise ValueError("could not infer repeat annotation format")
    if normalized_format not in {"gtf", "ucsc_rmsk"}:
        raise ValueError("format must be gtf, ucsc_rmsk or auto")

    parser = parse_repeat_gtf_line if normalized_format == "gtf" else parse_ucsc_rmsk_line
    with _open_text(resolved) as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                interval = parser(line, allowed_classes=allowed_classes)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid repeat annotation at {resolved}:{line_number}: {exc}"
                ) from exc
            if interval is not None:
                yield interval


class RepeatIndex:
    """Fixed-bin interval index with deterministic query output."""

    def __init__(self, intervals: Iterable[TEInterval], *, bin_size: int = 16_384) -> None:
        if not isinstance(bin_size, int) or isinstance(bin_size, bool) or bin_size <= 0:
            raise ValueError("bin_size must be a positive integer")
        self.bin_size = bin_size
        self._bins: dict[str, dict[int, list[TEInterval]]] = {}
        self._intervals: dict[str, TEInterval] = {}
        self._contigs: set[str] = set()
        for interval in intervals:
            if not isinstance(interval, TEInterval):
                raise ValueError("intervals must contain TEInterval objects")
            existing = self._intervals.get(interval.locus_id)
            if existing is not None and existing != interval:
                raise ValueError(f"conflicting duplicate locus_id: {interval.locus_id}")
            if existing is not None:
                continue
            self._intervals[interval.locus_id] = interval
            self._contigs.add(interval.chrom)
            chrom_bins = self._bins.setdefault(interval.chrom, {})
            first_bin = interval.start // bin_size
            last_bin = (interval.end - 1) // bin_size
            for bin_id in range(first_bin, last_bin + 1):
                chrom_bins.setdefault(bin_id, []).append(interval)
        for chrom_bins in self._bins.values():
            for values in chrom_bins.values():
                values.sort(key=lambda item: (item.start, item.end, item.locus_id))

    def __len__(self) -> int:
        return len(self._intervals)

    @property
    def contigs(self) -> frozenset[str]:
        return frozenset(self._contigs)

    def get(self, locus_id: str) -> TEInterval | None:
        return self._intervals.get(locus_id)

    def query(self, chrom: str, start: int, end: int) -> tuple[TEInterval, ...]:
        if not chrom:
            raise ValueError("chrom must be non-empty")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
            raise ValueError("query must satisfy 0 <= start < end")
        chrom_bins = self._bins.get(chrom)
        if chrom_bins is None:
            return ()
        hits: dict[str, TEInterval] = {}
        first_bin = start // self.bin_size
        last_bin = (end - 1) // self.bin_size
        for bin_id in range(first_bin, last_bin + 1):
            for interval in chrom_bins.get(bin_id, ()):
                if interval.start < end and start < interval.end:
                    hits[interval.locus_id] = interval
        return tuple(sorted(hits.values(), key=lambda item: (item.start, item.end, item.locus_id)))

    def query_blocks(
        self,
        chrom: str,
        blocks: Iterable[tuple[int, int]],
    ) -> tuple[TEInterval, ...]:
        hits: dict[str, TEInterval] = {}
        for start, end in blocks:
            for interval in self.query(chrom, start, end):
                hits[interval.locus_id] = interval
        return tuple(sorted(hits.values(), key=lambda item: (item.start, item.end, item.locus_id)))


def load_repeat_annotation(
    path: str | Path,
    *,
    fmt: str = "auto",
    allowed_classes: frozenset[str] = DEFAULT_TE_CLASSES,
    bin_size: int = 16_384,
) -> RepeatIndex:
    return RepeatIndex(
        iter_repeat_annotation(path, fmt=fmt, allowed_classes=allowed_classes),
        bin_size=bin_size,
    )
