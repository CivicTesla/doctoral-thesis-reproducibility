"""Exact small-code primitives for the marked quotient-Schur audit.

This module deliberately uses direct code/dual enumeration.  It is a theorem
checker for bounded controls, not an N=64/N=128 evaluator.
"""
from __future__ import annotations

from collections import Counter

from census_core import canonical_labels, collapsed_count, span_words
from psi2_trellis import gf2_rank


def _dot(left: int, right: int) -> int:
    return (left & right).bit_count() & 1


def dual_words(rows: tuple[int, ...], n: int) -> tuple[int, ...]:
    """The literal dual code; bounded controls only."""
    return tuple(h for h in range(1 << n) if all(_dot(h, row) == 0 for row in rows))


def _in_row_span(value: int, rows: list[int]) -> bool:
    return gf2_rank(rows + [value]) == gf2_rank(rows)


def form_rows(rows: tuple[int, ...], character: int) -> list[int]:
    """Rows of B_character in the supplied generator-coordinate basis."""
    return [sum((1 << j) for j, right in enumerate(rows)
                if _dot(character, left & right)) for left in rows]


def marked_character_data(rows: tuple[int, ...], a: int, character: int) -> dict:
    """Return (rho,A,chi) for the character induced by ``character in C^perp``."""
    matrix = form_rows(rows, character)
    linear = sum((1 << i) for i, row in enumerate(rows)
                 if _dot(character, a & row))
    return {"rho": gf2_rank(matrix), "A": _dot(character, a),
            "chi": _in_row_span(linear, matrix), "matrix_rows": matrix,
            "lambda": linear}


def marked_pencil_signature(rows: tuple[int, ...], a: int, n: int) -> dict:
    """Literal dual-character signatures and W values for small codes only."""
    r, s = len(rows), n - len(rows)
    signatures = Counter()
    weights: dict[tuple[int, int], int] = Counter()
    for character in dual_words(rows, n):
        item = marked_character_data(rows, a, character)
        key = (item["rho"], int(item["A"]), int(item["chi"]))
        signatures[key] += 1
        weights[(int(item["A"]), int(item["chi"]))] += 1 << (2 * r - item["rho"])
    w = {(A, chi): weights[(A, chi)] for A in (0, 1) for chi in (0, 1)}
    t = t_from_w(w, s)
    s0, s1, s2 = Counter(), Counter(), Counter()
    for (rho, A, chi), multiplicity in signatures.items():
        s0[rho] += multiplicity
        s1[(rho, A)] += multiplicity
        s2[(rho, chi)] += multiplicity
    return {"r": r, "s": s, "S0": _render(s0),
            "S1": _render(s1), "S2": _render(s2),
            "S3": _render(signatures), "W": _render_w(w), "T_from_characters": t}


def _render(counter: Counter) -> dict[str, int]:
    return {"/".join(map(str, key)) if isinstance(key, tuple) else str(key): int(value)
            for key, value in sorted(counter.items())}


def _render_w(w: dict[tuple[int, int], int]) -> dict[str, int]:
    return {f"{A}{chi}": int(w[(A, chi)]) for A in (0, 1) for chi in (0, 1)}


def t_from_w(w: dict[tuple[int, int], int], s: int) -> tuple[int, int, int, int]:
    """(T000,T001,T011,T111), in canonical branch order."""
    scale = 1 << s
    W00, W01, W10, W11 = (w[(0, 0)], w[(0, 1)], w[(1, 0)], w[(1, 1)])
    numerators = (W00 + W01 + W10 + W11,
                  W01 + W11,
                  W01 - W11,
                  W00 + W01 - W10 - W11)
    if any(value % scale for value in numerators):
        raise AssertionError("marked-Schur character normalization did not cancel")
    return tuple(value // scale for value in numerators)


def w_from_t(t: tuple[int, int, int, int], s: int) -> dict[tuple[int, int], int]:
    """Exact inverse of ``t_from_w``."""
    T000, T001, T011, T111 = t
    scale = 1 << (s - 1)
    values = {(0, 0): scale * (T000 + T111 - T001 - T011),
              (0, 1): scale * (T001 + T011),
              (1, 0): scale * (T000 - T111 - T001 + T011),
              (1, 1): scale * (T001 - T011)}
    if any(value < 0 for value in values.values()):
        raise AssertionError("invalid T tuple has negative W inverse")
    if t_from_w(values, s) != t:
        raise AssertionError("W/T inversion failure")
    return values


def direct_t_branches(rows: tuple[int, ...], a: int) -> tuple[int, int, int, int]:
    """Literal zero-overlap triple counts for (000,001,011,111)."""
    code = span_words(rows)
    coset = tuple(word ^ a for word in code)
    values = []
    for beta in canonical_labels(3):
        domains = tuple(coset if bit else code for bit in beta)
        values.append(sum(collapsed_count((x, y, z), a.bit_length() or 1) == (0, 0)
                          for x in domains for y in domains for z in domains))
    return tuple(values)


def direct_t_branches_n(rows: tuple[int, ...], a: int, n: int) -> tuple[int, int, int, int]:
    code = span_words(rows)
    coset = tuple(word ^ a for word in code)
    return tuple(sum(collapsed_count((x, y, z), n) == (0, 0)
                     for x in (coset if beta[0] else code)
                     for y in (coset if beta[1] else code)
                     for z in (coset if beta[2] else code))
                 for beta in canonical_labels(3))


def t_equation_counts(rows: tuple[int, ...], a: int) -> tuple[int, int, int, int]:
    """The four quotient equations, directly in the ambient word representation."""
    code = span_words(rows)
    cset = set(code)
    t000 = sum((z & q) in cset for z in code for q in code)
    t001 = sum(((z & v) ^ (a & z)) in cset for z in code for v in code)
    t011 = sum(((u & v) ^ (a & u) ^ (a & v) ^ a) in cset for u in code for v in code)
    t111 = sum(((z & q) ^ a) in cset for z in code for q in code)
    return t000, t001, t011, t111


def polar_dual_functional(index: int, n: int) -> int:
    """Physical word h_index with <h_index,g_j>=1{index=j}.

    The natural Polar matrix is involutory over F_2, so this is the index-th
    row of its inverse transpose.
    """
    return sum(1 << coordinate for coordinate in range(n)
               if index & ~coordinate == 0)


def polar_effective_signature(n: int, i: int, js: tuple[int, ...], gate: int = 20) -> dict:
    """Exact marked signature after removing character directions invisible to
    (mu_C,L_a,alpha).  It is intentionally fail-closed above ``gate``.
    """
    from census_core import polar_row
    from schur_active_pencil import active_meets

    jset = set(js)
    active = set(active_meets(js))
    marked = {i & u for u in js if (i & u) not in jset}
    visible = tuple(sorted(active | marked | {i}))
    if len(visible) > gate:
        raise ValueError(f"marked visible character dimension {len(visible)} exceeds gate {gate}")
    rows, a = tuple(polar_row(u, n) for u in js), polar_row(i, n)
    r, s = len(rows), n - len(rows)
    if len(visible) > s:
        raise AssertionError("visible quotient directions exceed codimension")
    signatures: Counter = Counter()
    weights: Counter = Counter()
    basis = tuple(polar_dual_functional(k, n) for k in visible)
    for mask in range(1 << len(visible)):
        character = 0
        for index, word in enumerate(basis):
            if mask >> index & 1:
                character ^= word
        item = marked_character_data(rows, a, character)
        key = (item["rho"], int(item["A"]), int(item["chi"]))
        signatures[key] += 1
        weights[(int(item["A"]), int(item["chi"]))] += 1 << (2 * r - item["rho"])
    invisible_factor = 1 << (s - len(visible))
    signatures = Counter({key: value * invisible_factor for key, value in signatures.items()})
    weights = Counter({key: value * invisible_factor for key, value in weights.items()})
    s0, s1, s2 = Counter(), Counter(), Counter()
    for (rho, A, chi), multiplicity in signatures.items():
        s0[rho] += multiplicity
        s1[(rho, A)] += multiplicity
        s2[(rho, chi)] += multiplicity
    w = {(A, chi): weights[(A, chi)] for A in (0, 1) for chi in (0, 1)}
    return {"r": r, "s": s, "visible_dimension": len(visible),
            "active_forbidden_meets": sorted(active), "marked_output_indices": sorted(marked),
            "visible_indices": list(visible), "invisible_character_factor": invisible_factor,
            "S0": _render(s0), "S1": _render(s1), "S2": _render(s2),
            "S3": _render(signatures), "W": _render_w(w), "T_from_characters": t_from_w(w, s)}
