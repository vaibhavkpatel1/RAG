"""Simplified HNSW: a single-layer proximity graph (no hierarchy).

Build:  insert vectors one at a time (random order). Each new vector runs a beam search over the graph
        built so far, links to its `M` closest finds, and each of those links back to it. A node keeps at
        most 2*M neighbours; when it overflows, its farthest neighbours are dropped.
Search: start at a fixed entry node and repeatedly expand the closest unexplored node, keeping the `ef`
        closest nodes seen. Larger `ef` explores more of the graph: slower but more accurate.

What is left out compared with real HNSW: the upper layers (long-range "express" links that get the search
near the answer in a few hops, so here the walk always starts from one node and must hop its way over), and
the diversity heuristic for choosing neighbours (here: simply the M closest).

Vectors are expected to be unit length, so distance = 1 - dot product (cosine distance).
"""
import heapq

import numpy as np


class HNSWIndex:
    def __init__(self, vecs, neighbors, entry, M):
        self.vecs, self.neighbors, self.entry, self.M = vecs, neighbors, entry, M
        self.last_evals = 0  # distance computations done by the most recent search

    @classmethod
    def build(cls, vecs, M=16, ef_construction=100, seed=0):
        order = np.random.default_rng(seed).permutation(len(vecs))
        g = cls(vecs, [[] for _ in vecs], int(order[0]), M)
        for i in map(int, order[1:]):
            found, _ = g._search(vecs[i], ef_construction)
            g.neighbors[i] = [j for _, j in found[:M]]
            for j in g.neighbors[i]:  # make the link two-way
                g.neighbors[j].append(i)
                if len(g.neighbors[j]) > 2 * M:  # over capacity: keep the closest 2*M
                    nb = np.array(g.neighbors[j])
                    g.neighbors[j] = nb[np.argsort(-(vecs[nb] @ vecs[j]))[:2 * M]].tolist()
        return g

    def _search(self, q, ef):
        """Beam search. Returns ([(distance, id), ...] closest first, number of nodes evaluated)."""
        d0 = 1 - float(self.vecs[self.entry] @ q)
        visited = {self.entry}
        frontier = [(d0, self.entry)]  # min-heap: closest unexplored node first
        best = [(-d0, self.entry)]     # max-heap (negated): the ef closest nodes found so far
        while frontier:
            d, c = heapq.heappop(frontier)
            if len(best) >= ef and d > -best[0][0]:
                break  # closest unexplored node is farther than our worst result: nothing left to gain
            new = [n for n in self.neighbors[c] if n not in visited]
            if not new:
                continue
            visited.update(new)
            for dn, n in zip((1 - self.vecs[new] @ q).tolist(), new):
                if len(best) < ef or dn < -best[0][0]:
                    heapq.heappush(frontier, (dn, n))
                    heapq.heappush(best, (-dn, n))
                    if len(best) > ef:
                        heapq.heappop(best)
        return sorted((-nd, n) for nd, n in best), len(visited)

    def candidates(self, q, ef):
        """Ids of the ~ef nodes the graph walk found closest to q."""
        found, self.last_evals = self._search(q, ef)
        return np.array([n for _, n in found])

    def reachable(self):
        """How many nodes can be reached from the entry node (should equal the number of vectors)."""
        seen, stack = {self.entry}, [self.entry]
        while stack:
            for n in self.neighbors[stack.pop()]:
                if n not in seen:
                    seen.add(n)
                    stack.append(n)
        return len(seen)

    def save(self, path):
        pad = np.full((len(self.neighbors), 2 * self.M), -1, dtype=np.int32)
        for i, nb in enumerate(self.neighbors):
            pad[i, :len(nb)] = nb
        np.savez(path, neighbors=pad, entry=self.entry, M=self.M)

    @classmethod
    def load(cls, path, vecs):
        d = np.load(path)
        return cls(vecs, [row[row >= 0].tolist() for row in d["neighbors"]], int(d["entry"]), int(d["M"]))
