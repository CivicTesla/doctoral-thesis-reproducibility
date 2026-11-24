"""Independent transcription of C1's exact parity recursion for N=32 pairs.

It evaluates only the C1-audited diagonal-decomposition class used by the
three known collisions.  Unlike the broad direct backend, this never loops
over all 2^(3r) tuples; it bottoms out at N=8 and convolves integer counters.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from itertools import product

from census_core import polar_row, span_words


def split_even_odd(word: int, n: int) -> tuple[int, int]:
    left = right = 0
    for k in range(n // 2):
        left |= ((word >> (2 * k)) & 1) << k
        right |= ((word >> (2 * k + 1)) & 1) << k
    return left, right


def parity(indices: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    even = {j // 2 for j in indices if not j & 1}
    odd = {j // 2 for j in indices if j & 1}
    return tuple(sorted(even & odd)), tuple(sorted(even - odd)), tuple(sorted(odd - even))


def convolve(a: Counter, b: Counter) -> Counter:
    out: Counter = Counter()
    for (a2, a3), av in a.items():
        for (b2, b3), bv in b.items():
            out[(a2 + b2, a3 + b3)] += av * bv
    return out


@lru_cache(maxsize=None)
def base(rows8: tuple[int, ...], shifts: tuple[int, int, int]) -> Counter:
    shifts = tuple(sorted(shifts))
    code = span_words(tuple(polar_row(j, 8) for j in rows8))
    branches = tuple(tuple(x ^ shift for x in code) for shift in shifts)
    out: Counter = Counter()
    for x in branches[0]:
        for y in branches[1]:
            xy = x & y
            xy_or = x | y
            for z in branches[2]:
                c3 = (xy & z).bit_count()
                c2 = (xy | (xy_or & z)).bit_count() - c3
                out[(c2, c3)] += 1
    return out


class Parent16:
    def __init__(self, rows16: tuple[int, ...]):
        paired, even_only, diagonal = parity(rows16)
        if even_only:
            raise ValueError("not a C1 audited diagonal-decomposition state")
        self.paired = paired
        self.diagonal = span_words(tuple(polar_row(j, 8) for j in diagonal))
        self.cache: dict[tuple[int, int, int], Counter] = {}

    def psi3(self, shifts16: tuple[int, int, int]) -> Counter:
        if shifts16 in self.cache:
            return self.cache[shifts16]
        split = tuple(split_even_odd(s, 16) for s in shifts16)
        out: Counter = Counter()
        for hs in product(self.diagonal, repeat=3):
            left = tuple(split[q][0] ^ hs[q] for q in range(3))
            right = tuple(split[q][1] ^ hs[q] for q in range(3))
            for key, value in convolve(base(self.paired, left), base(self.paired, right)).items():
                out[key] += value
        self.cache[shifts16] = out
        return out


def psi3(i: int, js: tuple[int, ...], beta: tuple[int, int, int]) -> Counter:
    paired, even_only, diagonal = parity(js)
    if i % 2 != 1 or even_only:
        raise ValueError("object is outside the audited C1 N=32 recursion class")
    parent = Parent16(paired)
    diagonal16 = span_words(tuple(polar_row(j, 16) for j in diagonal))
    shift = polar_row(i // 2, 16)
    out: Counter = Counter()
    for hs in product(diagonal16, repeat=3):
        p = parent.psi3(tuple(hs[q] ^ (shift if beta[q] else 0) for q in range(3)))
        for key, value in convolve(p, p).items():
            out[key] += value
    if sum(out.values()) != 2 ** (3 * len(js)):
        raise AssertionError("tuple multiplicity mismatch")
    return out


def signature(i: int, js: tuple[int, ...]) -> tuple:
    labels = ((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1))
    return tuple(tuple(sorted(psi3(i, js, beta).items())) for beta in labels)
