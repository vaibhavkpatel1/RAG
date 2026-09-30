"""IVF (inverted file) index with a from-scratch k-means coarse quantizer.

Build:  k-means groups the chunk vectors into `nlist` clusters; each cluster keeps an inverted list of
        the ids of its members.
Search: compare the query with the `nlist` centroids only, pick the `nprobe` closest clusters, and scan
        just the vectors in those lists. nprobe == nlist scans everything (exact); smaller is faster
        but can miss a neighbour that sits in a cluster that was not probed.

Vectors are expected to be unit length (rag.embed normalizes), so the dot product is cosine similarity
and k-means is the spherical variant: assign by highest dot product, update = renormalized mean.
"""
import numpy as np


def kmeans(x, k, iters=25, seed=0):
    """Spherical k-means with k-means++ seeding. Returns (centroids [k, d], labels [n])."""
    rng = np.random.default_rng(seed)
    n = len(x)
    first = rng.integers(n)
    centers = [x[first]]
    d2 = 2 - 2 * (x @ x[first])  # squared L2 to the nearest chosen centre (unit vectors)
    for _ in range(1, k):
        p = np.clip(d2, 0, None)
        nxt = rng.choice(n, p=p / p.sum()) if p.sum() > 0 else rng.integers(n)
        centers.append(x[nxt])
        d2 = np.minimum(d2, 2 - 2 * (x @ x[nxt]))
    c = np.stack(centers).astype(np.float32)

    labels = None
    for _ in range(iters):
        new = np.argmax(x @ c.T, axis=1)
        if labels is not None and np.array_equal(new, labels):
            break  # converged
        labels = new
        for j in range(k):
            members = x[labels == j]
            if len(members):
                m = members.mean(axis=0)
                c[j] = m / max(np.linalg.norm(m), 1e-12)
            else:  # empty cluster: re-seed on the point worst served by its centroid
                c[j] = x[np.argmin(np.max(x @ c.T, axis=1))]
    return c, np.argmax(x @ c.T, axis=1)


class IVFIndex:
    def __init__(self, centroids, labels):
        self.centroids = centroids
        self.labels = labels
        self.lists = [np.flatnonzero(labels == j) for j in range(len(centroids))]

    @classmethod
    def build(cls, vecs, nlist=None, iters=25, seed=0):
        nlist = nlist or max(1, round(np.sqrt(len(vecs))))  # common rule of thumb: ~sqrt(N)
        return cls(*kmeans(vecs, nlist, iters, seed))

    def candidates(self, q, nprobe):
        """Ids of every vector in the `nprobe` clusters whose centroid is closest to q."""
        probe = np.argsort(-(self.centroids @ q))[:min(nprobe, len(self.centroids))]
        return np.concatenate([self.lists[j] for j in probe])

    def save(self, path):
        np.savez(path, centroids=self.centroids, labels=self.labels)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["centroids"], d["labels"])
