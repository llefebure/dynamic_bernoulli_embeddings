import numpy as np


class Embedding:
    """A class for performing analysis on the trained embeddings"""

    def __init__(self, embeddings, dictionary):
        self.embeddings = embeddings
        self.token_to_id = dictionary
        self.id_to_token = {v: k for k, v in dictionary.items()}

    def neighborhood(self, v, t, n=20):
        """Finds the neighborhood of terms around `v` in timestep `t`"""
        idx = self.token_to_id[v]
        rho_v = self.embeddings[t][idx].reshape((1, -1))
        rho = self.embeddings[t].T
        sim = np.dot(np.sign(rho_v), rho).flatten()
        sim /= np.sqrt((rho_v ** 2).sum())
        sim /= np.sqrt((rho ** 2).sum(axis=0).flatten())
        return [self.id_to_token[i] for i in np.argsort(sim)[::-1][:n]]

    def absolute_drift(self, n=50):
        """Find the top drifting terms"""
        change = np.sqrt(
            ((self.embeddings[-1] - self.embeddings[0]) ** 2).sum(axis=1)
        ).flatten()
        return sorted(list(zip(change, self.id_to_token.values())), reverse=True)[:n]

    def change_points(self, n=50):
        """Find the most variable terms"""
        change = np.sqrt(
            ((self.embeddings[1:] - self.embeddings[:-1]) ** 2).sum(axis=-1)
        )
        ordered = np.argsort(change, axis=None)[::-1]
        times = ordered // change.shape[1]
        terms = ordered // change.shape[0]
        return list(zip(times, [self.id_to_token[t] for t in terms]))[:n]
