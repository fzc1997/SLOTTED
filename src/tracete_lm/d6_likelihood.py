"""Closed-form error-aware k-mer likelihood used by SLOTTED-Joint."""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Sequence

PLATFORM_PROFILES = {
    # substitution rate, insertion rate, deletion rate (per base), 5' truncation shape
    "ONT": {"sub": 0.03, "ins": 0.015, "del": 0.015},
    "PACBIO": {"sub": 0.005, "ins": 0.002, "del": 0.002},
}


COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


DIST_CAP = 2  # words with no single-edit neighbour are treated at distance 2


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


def _forward_kmers(sequence: str, k: int) -> set[str]:
    return {sequence[i : i + k] for i in range(len(sequence) - k + 1)}


def two_strand_kmers(sequence: str, k: int) -> set[str]:
    forward = _forward_kmers(sequence, k)
    return forward | {reverse_complement(w) for w in forward}


def _clean(words: set[str]) -> set[str]:
    return {w for w in words if "N" not in w}


@dataclass(frozen=True, slots=True)
class D6Weights:
    """Closed-form error model constants for one platform profile."""

    platform: str
    k: int
    log_intact: float           # log[(1 - e)^k]
    log_p_mut: dict[int, float]  # d -> log[q^d (1-q)^(k-d)]

    @classmethod
    def from_platform(cls, platform: str, k: int = 31) -> "D6Weights":
        profile = PLATFORM_PROFILES[platform]
        sub, ins, dele = profile["sub"], profile["ins"], profile["del"]
        e = sub + ins + dele
        q = sub / 3.0 + ins + dele
        log_p = {d: d * math.log(q) + (k - d) * math.log1p(-q) for d in range(1, 8)}
        return cls(platform=platform, k=k, log_intact=k * math.log1p(-e), log_p_mut=log_p)

    def log_word(self, distance: int) -> float:
        if distance <= 0:
            return self.log_intact
        return self.log_p_mut[min(distance, DIST_CAP)]


def _single_edit_neighbours(word: str) -> list[str]:
    """All words at edit distance <= 1 (mismatch / deletion / insertion)."""
    bases = "ACGT"
    out: list[str] = []
    for i in range(len(word)):
        prefix, suffix = word[:i], word[i + 1 :]
        original = word[i]
        for b in bases:
            if b != original:
                out.append(prefix + b + suffix)
        out.append(prefix + suffix)  # delete position i
    for i in range(len(word) + 1):
        prefix, suffix = word[:i], word[i:]
        for b in bases:
            out.append(prefix + b + suffix)  # insert before position i
    return out


@dataclass(frozen=True, slots=True)
class CandidateIndex:
    """Membership oracle for one molecule's candidate two-strand k-mer sets."""

    n_candidates: int
    membership: dict[str, int]  # word -> bitmask of candidates containing it

    @classmethod
    def build(cls, candidate_sequences: Sequence[str], k: int) -> "CandidateIndex":
        sets = [two_strand_kmers(sequence, k) for sequence in candidate_sequences]
        return cls.from_kmer_sets(sets)

    @classmethod
    def from_kmer_sets(cls, kmer_sets: Sequence[set[str]]) -> "CandidateIndex":
        membership: dict[str, int] = {}
        for i, words in enumerate(kmer_sets):
            for word in words:
                if "N" in word:
                    continue
                membership[word] = membership.get(word, 0) | (1 << i)
        return cls(n_candidates=len(kmer_sets), membership=membership)

    def contains_mask(self, word: str) -> int:
        return self.membership.get(word, 0)

    def single_edit_mask(self, word: str) -> int:
        """Bitmask of candidates holding some word at edit distance <= 1."""
        mask = 0
        for neighbour in _single_edit_neighbours(word):
            mask |= self.membership.get(neighbour, 0)
        return mask


def score_read_indexed(
    read_sequence: str,
    index: CandidateIndex,
    weights: D6Weights,
    *,
    k: int = 31,
) -> list[float]:
    n = index.n_candidates
    totals = [0.0] * n
    full_mask = (1 << n) - 1
    for word in _clean(_forward_kmers(read_sequence, k)):
        exact = index.contains_mask(word)
        if exact == full_mask:
            continue  # symmetric contribution; cancels in every margin
        near = index.single_edit_mask(word) & ~exact
        far = full_mask & ~exact & ~near
        for i in range(n):
            bit = 1 << i
            if exact & bit:
                totals[i] += weights.log_word(0)
            elif near & bit:
                totals[i] += weights.log_word(1)
            elif far & bit:
                totals[i] += weights.log_word(DIST_CAP)
    return totals
