#!/usr/bin/env python3
"""
Computational checks for
  "Higher-Order Reconstruction of Subtree Families from Intersection Data"

What this script DOES:
  * certifies every explicit finite object in the paper (witness families,
    Table 1, R10, the subdivision and spider constructions for small t);
  * re-derives the nine-vertex classification in its reduced (post-projection)
    form over all 47 nine-vertex trees, and the eight-vertex Helly barrier;
  * runs trade-theory lemmas (3.1, 3.2, 3.3, 3.4) and the pairwise
    classification as bounded brute-force sanity checks (small r only).

What this script does NOT do:
  * it does not prove any statement quantified over all r or all trees.
    Lemmas 3.1-3.4, Theorem 3.7, Theorem 4.1 (upper bound), Proposition 5.1,
    Lemma 7.1 and the reduction steps in Theorems 6.2 / 8.1 still need a
    human proof check.  A PASS in the bounded sections means "no
    counterexample in the searched range", nothing more.

Requirements:  python3, networkx, numpy
    pip install networkx numpy
Run:           python3 verify_c2.py            (all sections, a few minutes)
               python3 verify_c2.py --quick    (skip the slower bounded searches)
"""
import sys, itertools, time
from collections import Counter
import networkx as nx
import numpy as np

QUICK = "--quick" in sys.argv
RESULTS = []

def report(name, ok, detail=""):
    RESULTS.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))

def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)

# ---------------------------------------------------------------------------
# basic helpers: families as lists of vertex sets, rows as label bitmasks
# ---------------------------------------------------------------------------
def is_connected_set(T, S):
    S = set(S)
    return len(S) > 0 and nx.is_connected(T.subgraph(S))

def signature(F, k):
    """labelled intersection cardinalities |cap_{i in S} F_i| for 1<=|S|<=k"""
    r = len(F)
    sig = {}
    for j in range(1, min(k, r) + 1):
        for S in itertools.combinations(range(r), j):
            sig[S] = len(set.intersection(*[set(F[i]) for i in S]))
    return sig

def active_rows(F):
    """multiset of nonzero incidence rows = active incidence structure up to iso"""
    rows = Counter()
    for v in set().union(*map(set, F)):
        rows[frozenset(i for i, A in enumerate(F) if v in A)] += 1
    return rows

def isomorphic(F, G):
    return active_rows(F) == active_rows(G)

def families_from_rows(rowmap, r):
    """rowmap: vertex -> set of labels (subset of range(r))"""
    return [set(v for v, R in rowmap.items() if i in R) for i in range(r)]

def has_sdr(F):
    B = nx.Graph()
    L = [("s", i) for i in range(len(F))]
    B.add_nodes_from(L)
    for i, A in enumerate(F):
        for v in A:
            B.add_edge(("s", i), ("v", v))
    M = nx.bipartite.maximum_matching(B, top_nodes=L)
    return sum(1 for x in L if x in M) == len(F)

def parity_rows(m, parity, pad=0):
    R = [frozenset(c) for k in range(m + 1)
         for c in itertools.combinations(range(m), k) if k % 2 == parity]
    return Counter(R + [frozenset()] * pad)

def monotone_labelling(T, core, ms):
    """bijective labelling of T - core by multiset ms with child row <= parent row
    (Lemma 7.1 condition); returns dict or None"""
    rest = [v for v in T if v not in core]
    if len(rest) != sum(ms.values()):
        return None
    parent, order, seen, frontier = {}, [], set(core), list(core)
    while frontier:
        nf = []
        for u in frontier:
            for w in T[u]:
                if w not in seen:
                    seen.add(w); parent[w] = None if u in core else u
                    order.append(w); nf.append(w)
        frontier = nf
    lab, cnt = {}, Counter(ms)
    def bt(i):
        if i == len(order):
            return True
        v, p = order[i], parent[order[i]]
        for R in list(cnt):
            if cnt[R] == 0 or (p is not None and not R <= lab[p]):
                continue
            cnt[R] -= 1; lab[v] = R
            if bt(i + 1):
                return True
            cnt[R] += 1
        return False
    return dict(lab) if bt(0) else None

def core_collision_families(T, core, ms_even, ms_odd, m):
    """turn two monotone labellings around a common core into families A, B"""
    le = monotone_labelling(T, core, ms_even)
    lo = monotone_labelling(T, core, ms_odd)
    if le is None or lo is None:
        return None
    full = frozenset(range(m))
    re = {**le, **{c: full for c in core}}
    ro = {**lo, **{c: full for c in core}}
    return families_from_rows(re, m), families_from_rows(ro, m)

def certify_collision(T, A, B, t, name):
    conn = all(is_connected_set(T, S) for S in A + B)
    eq = signature(A, t) == signature(B, t)
    noniso = not isomorphic(A, B)
    report(name, conn and eq and noniso,
           f"connected={conn}, Sigma<= {t} equal={eq}, nonisomorphic={noniso}")
    return conn and eq and noniso

GM = lambda G, H: nx.algorithms.isomorphism.GraphMatcher(G, H).subgraph_is_monomorphic()

Q_EDGES = {
    "Q1": [(0,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8)],
    "Q2": [(0,1),(0,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8)],
    "Q3": [(0,1),(0,2),(1,3),(1,5),(1,6),(1,7),(1,8),(3,4)],
    "Q4": [(0,1),(0,2),(0,3),(1,4),(1,5),(1,6),(1,7),(1,8)],
    "Q5": [(0,1),(0,2),(0,3),(0,4),(1,5),(1,6),(1,7),(1,8)],
    "Q6": [(0,1),(0,2),(0,3),(0,4),(1,5),(1,7),(1,8),(5,6)],
    "Q7": [(0,1),(0,2),(0,3),(1,4),(1,6),(1,7),(1,8),(4,5)],
}
Q_DEG = {"Q1":[8]+[1]*8, "Q2":[7,2]+[1]*7, "Q3":[6,2,2]+[1]*6, "Q4":[6,3]+[1]*7,
         "Q5":[5,4]+[1]*7, "Q6":[4,4,2]+[1]*6, "Q7":[5,3,2]+[1]*6}
QG = {k: nx.Graph(v) for k, v in Q_EDGES.items()}

# ===========================================================================
section("1. Section 5: pairwise witnesses on F3 and K_{1,4}")
F3 = nx.Graph([(0,1),(0,2),(0,3),(3,4)])
K14 = nx.star_graph(4)
A3 = [{0},{0,1,2},{0,1,2},{0,1,3},{0,2,3}]; B3 = [{0},{0,1,3},{0,1,3},{0,2,3},{0,3,4}]
A4 = [{0,1,2},{0,1,3},{0,2,3},{0,2,3},{0,2,3}]; B4 = [{0,1,2},{0,1,3},{0,1,4},{0,1,4},{0,1,4}]
for nm, T, A, B, trip in [("F3", F3, A3, B3, (1,3,4)), ("K14", K14, A4, B4, (0,1,2))]:
    certify_collision(T, A, B, 2, f"{nm}: order-2 collision")
    s3a, s3b = signature(A, 3)[trip], signature(B, 3)[trip]
    report(f"{nm}: stated triple intersection 1 vs 2 (labels {tuple(i+1 for i in trip)})",
           (s3a, s3b) == (1, 2), f"got {s3a} vs {s3b}")
    report(f"{nm}: SDR  A has none, B has one", (not has_sdr(A)) and has_sdr(B))
M5 = [[1,1,1,1,1],[1,3,3,2,2],[1,3,3,2,2],[1,2,2,3,2],[1,2,2,2,3]]
M6 = [[3,2,2,2,2],[2,3,2,2,2],[2,2,3,3,3],[2,2,3,3,3],[2,2,3,3,3]]
pm = lambda F: [[len(F[i] & F[j]) for j in range(5)] for i in range(5)]
report("matrix (5) matches", pm(A3) == M5 and pm(B3) == M5)
report("matrix (6) matches", pm(A4) == M6 and pm(B4) == M6)
report("SDR (0,1,3,2,4) is valid for B^(3)",
       all(x in B3[i] for i, x in enumerate([0,1,3,2,4])) and len({0,1,3,2,4}) == 5)

# ===========================================================================
section("2. Table 1: the seven nine-vertex hosts")
trees9 = list(nx.nonisomorphic_trees(9))
report("number of nine-vertex trees is 47", len(trees9) == 47, str(len(trees9)))
for k, G in QG.items():
    ok = nx.is_tree(G) and G.number_of_nodes() == 9 and \
         sorted((d for _, d in G.degree()), reverse=True) == Q_DEG[k]
    report(f"{k}: tree on 9 vertices with degree sequence {Q_DEG[k][:3]}...", ok)
report("Q1..Q7 pairwise nonisomorphic",
       all(not nx.is_isomorphic(a, b) for a, b in itertools.combinations(QG.values(), 2)))

# ===========================================================================
section("3. Explicit r=4 collisions on Q1..Q7 (Lemma 7.2 / Theorem 8.1, converse)")
E4, O4 = parity_rows(4, 0), parity_rows(4, 1)
for k, G in QG.items():
    found = False
    for h in G:
        fam = core_collision_families(G, {h}, E4, O4, 4)
        if fam:
            A, B = fam
            found = certify_collision(G, A, B, 3, f"{k}: order-3 collision rooted at vertex {h}")
            break
    if not found:
        report(f"{k}: order-3 collision", False, "no common-root realization found")

# ===========================================================================
section("4. Theorem 8.1, reduced form, over all 47 nine-vertex trees")
indep, common = [], []
for T in trees9:
    oh = {h for h in T if monotone_labelling(T, {h}, O4)}
    eh = {h for h in T if monotone_labelling(T, {h}, E4)}
    if oh and eh: indep.append(T)
    if oh & eh: common.append(T)
isQ = lambda T: any(nx.is_isomorphic(T, q) for q in QG.values())
report("trees with O-root and E-root (roots may differ): exactly the seven Q",
       len(indep) == 7 and all(isQ(T) for T in indep), f"count={len(indep)}")
report("trees with a common O/E root: exactly the seven Q",
       len(common) == 7 and all(isQ(T) for T in common), f"count={len(common)}")

# ===========================================================================
section("5. Theorem 6.2: no eight-vertex tree carries the odd leg O on all vertices")
trees8 = list(nx.nonisomorphic_trees(8))
Orows = list(O4.elements())
bad = 0
for T in trees8:
    V = list(T)
    conn = set()
    for k in range(1, 9):
        for S in itertools.combinations(V, k):
            if T.subgraph(S).number_of_edges() == k - 1:
                conn.add(frozenset(S))
    for perm in itertools.permutations(Orows):
        if all(frozenset(V[j] for j in range(8) if i in perm[j]) in conn for i in range(4)):
            bad += 1; break
report(f"all {len(trees8)} eight-vertex trees: no connected realization of O", bad == 0,
       f"realizable trees={bad}")

# ===========================================================================
section("6. Proposition 8.2: R10")
R10 = nx.Graph([(0,1),(0,2),(1,5),(1,7),(1,8),(1,9),(2,3),(2,4),(5,6)])
ev = {2:"1234",3:"12",4:"13",5:"23",6:"",7:"14",8:"24",9:"34"}
od = {2:"123",3:"1",4:"2",5:"124",6:"4",7:"3",8:"134",9:"234"}
tor = lambda d: {**{v: {int(c)-1 for c in s} for v, s in d.items()}, 0: {0,1,2,3}, 1: {0,1,2,3}}
A, B = families_from_rows(tor(ev), 4), families_from_rows(tor(od), 4)
report("even/odd tables use exactly E and O",
       Counter(frozenset(x) for v, x in tor(ev).items() if v > 1) == E4 and
       Counter(frozenset(x) for v, x in tor(od).items() if v > 1) == O4)
certify_collision(R10, A, B, 3, "R10: order-3 collision")
report("R10 degree sequence (5,3,2,2,1^6)",
       sorted((d for _, d in R10.degree()), reverse=True) == [5,3,2,2]+[1]*6)
report("R10 contains none of Q1..Q7 as a subgraph", not any(GM(R10, q) for q in QG.values()))

# ===========================================================================
section("7. Proposition 8.5: subdivision jump, t = 2,3,4")
for t in (2, 3, 4):
    m = t + 1
    Tt = nx.star_graph(2**t - 1)          # centre 0, leaves 1..2^t-1
    Tt.remove_edge(0, 1); Tt.add_edges_from([(0, 2**t), (2**t, 1)])   # subdivide
    Et, Ot = parity_rows(m, 0), parity_rows(m, 1)
    fam = core_collision_families(Tt, {0}, Et, Ot, m)
    ok = fam is not None
    if ok:
        certify_collision(Tt, fam[0], fam[1], t, f"t={t}: order-{t} collision on T_t")
    else:
        report(f"t={t}: order-{t} collision on T_t", False)
    report(f"t={t}: |V(T_t)| = 2^t+1 so log bound gives <= t+1",
           Tt.number_of_nodes() == 2**t + 1 and int(np.floor(np.log2(2**t + 1))) + 1 == t + 1)

# ===========================================================================
section("8. Theorem 8.6: central binomial spiders, t = 2,3,4")
from math import comb, ceil
for t in (2, 3, 4):
    m = t + 1; q = comb(m, m // 2); L = ceil((m + 1) / 2)
    S = nx.Graph()
    arms = []
    nid = 1
    for a in range(q):
        path = list(range(nid, nid + L)); nid += L
        S.add_edge(0, path[0]); S.add_edges_from(zip(path, path[1:])); arms.append(path)
    fams = []
    for parity in (0, 1):
        elems = [frozenset(c) for k in range(m + 1)
                 for c in itertools.combinations(range(m), k) if k % 2 == parity]
        # minimum chain cover via bipartite matching (Dilworth)
        Bg = nx.DiGraph()
        for x in elems:
            for y in elems:
                if y < x:
                    Bg.add_edge(("L", x), ("R", y))
        Ug = nx.Graph(Bg.to_undirected()); Ug.add_nodes_from(("L", x) for x in elems)
        Ug.add_nodes_from(("R", x) for x in elems)
        M = nx.bipartite.maximum_matching(Ug, top_nodes=[("L", x) for x in elems])
        succ = {x: M[("L", x)][1] for x in elems if ("L", x) in M}
        has_pred = set(succ.values())
        chains = []
        for x in sorted(elems, key=len, reverse=True):
            if x in has_pred:
                continue
            ch = [x]
            while ch[-1] in succ:
                ch.append(succ[ch[-1]])
            chains.append(ch)
        ok = len(chains) <= q and max(map(len, chains)) <= L
        rows = {0: frozenset(range(m))}
        for a, path in enumerate(arms):
            ch = chains[a] if a < len(chains) else []
            for j, v in enumerate(path):
                rows[v] = ch[j] if j < len(ch) else frozenset()
        fams.append((families_from_rows(rows, m), ok, len(chains)))
    (A, okA, cA), (B, okB, cB) = fams
    report(f"t={t}: chain covers fit (q={q} arms, L={L}); chains used {cA},{cB}", okA and okB)
    report(f"t={t}: p = qL - 2^t >= 0", q * L - 2**t >= 0, f"p={q*L-2**t}")
    certify_collision(S, A, B, t, f"t={t}: order-{t} collision on spider S_t (leaves={q})")

# ===========================================================================
section("9. Theorem 4.1 lower bound: parity leaf rows on K_{1,d}, d = 4..16")
allok = True
for d in range(4, 17):
    s = int(np.floor(np.log2(d)))
    T = nx.star_graph(d)
    fam = core_collision_families(T, {0}, parity_rows(s+1, 0, d - 2**s),
                                  parity_rows(s+1, 1, d - 2**s), s + 1)
    A, B = fam
    ok = all(is_connected_set(T, X) for X in A + B) and \
         signature(A, s) == signature(B, s) and not isomorphic(A, B)
    allok &= ok
report("order-s collision with s = floor(log2 d) for every d in 4..16", allok)

# ===========================================================================
section("10. Pairwise layer, independent brute force (bounded r)")
def brute_collision(T, k, r):
    V = sorted(T)
    conn = []
    for size in range(1, len(V) + 1):
        for S in itertools.combinations(V, size):
            if T.subgraph(S).number_of_edges() == size - 1:
                conn.append(frozenset(S))
    seen = {}
    for tup in itertools.product(conn, repeat=r):
        F = list(tup)
        key = tuple(sorted(signature(F, k).items()))
        rows = active_rows(F)
        canon = frozenset(rows.items())
        if key in seen and seen[key] != canon:
            return True
        seen.setdefault(key, canon)
    return False
hosts = [("P4", nx.path_graph(4)), ("K13", nx.star_graph(3)), ("P5", nx.path_graph(5)),
         ("F3", F3), ("K14", K14)]
rmax = 3 if QUICK else 4
for nm, T in hosts:
    t0 = time.time()
    found = any(brute_collision(T, 2, r) for r in range(1, rmax + 1))
    expect = nm in ("F3", "K14")
    tag = "collision found" if found else "no collision"
    if found == expect:
        report(f"{nm}: order-2, r<={rmax}: {tag}", True, f"{time.time()-t0:.1f}s")
    elif expect and not found:
        report(f"{nm}: order-2, r<={rmax}: expected collision not found", False)
    else:
        report(f"{nm}: order-2, r<={rmax}: unexpected collision", False)

# ===========================================================================
section("11. Trade lemmas, exhaustive on small r (sanity only; lemmas are for all r)")
# Every integer [t] trade f on r coordinates is f = mu * h, where h(S) = sum_{C >= S} f(C)
# vanishes for |S| <= t and is an arbitrary integer for |S| > t.  If vol(f) <= V then
# |h(S)| <= V.  So enumerating h in [-V, V]^{#high sets} lists every trade of volume <= V.
def enumerate_trades(r, t, V):
    blocks = list(range(1 << r))
    pc = lambda x: bin(x).count("1")
    high = [S for S in blocks if pc(S) > t]
    MU = np.array([[(-1) ** (pc(S) - pc(C)) if S & C == C else 0 for S in high]
                   for C in blocks], dtype=np.int32)
    vals = np.arange(-V, V + 1, dtype=np.int32)
    lead = min(2, len(high))
    rest = np.array(list(itertools.product(vals, repeat=len(high) - lead)),
                    dtype=np.int32).reshape(-1, len(high) - lead)
    by_vol = {}
    for pre in itertools.product(vals, repeat=lead):
        H = np.hstack([np.tile(np.array(pre, dtype=np.int32), (len(rest), 1)), rest])
        Fm = H @ MU.T
        vol = np.clip(Fm, 0, None).sum(axis=1)
        keep = (vol > 0) & (vol <= V)
        for f, v in zip(Fm[keep], vol[keep]):
            by_vol.setdefault(int(v), []).append(f)
    return by_vol

for (r, t, V, forbidden, must) in [(3, 1, 4, {1}, {2, 3, 4}),
                                   (4, 2, 7, {1, 2, 3, 5, 7}, {4, 6}),
                                   (5, 3, 9, set(range(1, 8)) | {9}, {8})]:
    t0 = time.time()
    by_vol = enumerate_trades(r, t, V)
    spec = sorted(by_vol)
    ok = not (set(spec) & forbidden) and must <= set(spec)
    report(f"[{t}] trades on r={r}, volume <= {V}: volume spectrum {spec}", ok,
           f"forbidden {sorted(forbidden)} absent, {time.time()-t0:.0f}s")
    if (r, t) == (5, 3):
        tr8 = by_vol.get(8, [])
        def projects_to_parity(f):
            blocks = range(1 << r)
            for D in itertools.combinations(range(r), 4):
                Dm = sum(1 << i for i in D)
                pos, neg = Counter(), Counter()
                for C in blocks:
                    if f[C] > 0: pos[C & Dm] += int(f[C])
                    if f[C] < 0: neg[C & Dm] -= int(f[C])
                par = lambda X: {bin(x).count("1") % 2 for x in X}
                if len(pos) == 8 and len(neg) == 8 and max(pos.values()) == 1 and \
                   max(neg.values()) == 1 and len(par(pos)) == 1 and par(pos) != par(neg):
                    return True
            return False
        report(f"Lemma 3.4: all {len(tr8)} volume-8 [3] trades on r=5 project to E/O",
               len(tr8) > 0 and all(projects_to_parity(f) for f in tr8))
print("  note: r=4 is the largest r for which [2] volume 5 can be enumerated this way;")
print("        Lemma 3.2 for general r rests on its written proof.")

# ===========================================================================
section("SUMMARY")
nf = sum(1 for _, ok in RESULTS if not ok)
print(f"  {len(RESULTS) - nf} passed, {nf} failed")
for nm, ok in RESULTS:
    if not ok:
        print(f"  FAILED: {nm}")
sys.exit(1 if nf else 0)