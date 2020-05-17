"""Contains embedding model implementation"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class DynamicBernoulliEmbeddingModel(nn.Module):
    def __init__(
        self,
        V,
        T,
        m,
        dictionary,
        sampling_distribution,
        k=50,
        lambda_=1e4,
        lambda_0=1,
        ns=20,
    ):
        """
        Parameters
        ----------
        V : int
            Vocabulary size.
        T : int
            Number of timesteps.
        m : int
            The scaling factor to apply to the pseudo likelihood values.
        dictionary : dict
            Maps word to index.
        sampling_distribution : tensor (V,)
            The unigram distribution to use for negative sampling.
        k : int
            Embedding dimension.
        lambda_ : int
            Scaling factor on the time drift prior.
        lambda_0 : int
            Scaling factor on the embedding priors.
        ns : int
            Number of negative samples.
        """
        super().__init__()
        self.V = V  # Vocab size.
        self.T = T  # Number of timestepss.
        self.k = k  # Embedding dimension.
        self.m = m  # Scaling factor for pseudo LL
        self.lambda_ = lambda_  # Scaling factor on the time drift prior.
        self.lambda_0 = lambda_0  # Scaling factor on the embedding priors.
        self.sampling_distribution = Categorical(logits=sampling_distribution)
        self.negative_samples = ns  # Number of negative samples.
        self.dictionary = dictionary
        self.dictionary_reverse = {v: k for k, v in dictionary.items()}
        self.embeddings = None  # Will eventually be the trained embeddings for analysis

        # Embeddings parameters.
        self.rho = nn.Embedding(V * T, k)  # Stacked dynamic embeddings
        self.alpha = nn.Embedding(V, k)  # Time independent context embeddings
        with torch.no_grad():
            nn.init.normal_(self.rho.weight, 0, 0.01)
            nn.init.normal_(self.alpha.weight, 0, 0.01)

        # Transformations
        self.log_sigmoid = nn.LogSigmoid()
        self.sigmoid = nn.Sigmoid()

    def L_pos(self, eta):
        return self.log_sigmoid(eta).sum()

    def L_neg(self, batch_size, times, contexts_summed):
        neg_samples = self.sampling_distribution.sample(
            torch.Size([batch_size, self.negative_samples])
        )
        neg_samples = neg_samples + (times * self.V).reshape((-1, 1))
        neg_samples = neg_samples.T.flatten()
        context_flat = contexts_summed.repeat((self.negative_samples, 1))
        eta_neg = (self.rho(neg_samples) * context_flat).sum(axis=1)
        return (torch.log(1 - self.sigmoid(eta_neg) + 1e-7)).sum()

    def forward(self, targets, times, contexts, validate=False, dynamic=True):
        """Forward pass of the model

        Parameters
        ----------
        targets : (batch_size,)
        times : (batch_size,)
        contexts : (batch_size, 2 * context_size)
        dynamic : bool
            Indicates whether to include the drift component of the loss.

        Returns
        -------
        loss
        L_pos
        L_neg
        L_prior
        """
        batch_size = targets.shape[0]

        # Since the embeddings are stacked, adjust the indices for the targets.
        # In other words, word `i` in time slice `j` would be at position
        # `j * V + i` in the embedding matrix where V is the vocab size.
        targets_adjusted = times * self.V + targets

        # -1 indicates out of bounds for the context word, so mask these out so
        # they don't contribute to the context sum.
        context_mask = contexts == -1
        contexts[context_mask] = 0
        contexts = self.alpha(contexts)
        contexts[context_mask] = 0
        contexts_summed = contexts.sum(axis=1)
        eta = (self.rho(targets_adjusted) * contexts_summed).sum(axis=1)

        # Loss
        loss, L_pos, L_neg, L_prior = None, None, None, None

        L_pos = self.L_pos(eta)
        if not validate:
            L_neg = self.L_neg(batch_size, times, contexts_summed)
            loss = self.m * (L_pos + L_neg)
            L_prior = -self.lambda_0 / 2 * (self.alpha.weight ** 2).sum()
            L_prior += -self.lambda_0 / 2 * (self.rho.weight[0] ** 2).sum()
            if dynamic:
                rho_trans = self.rho.weight.reshape((self.T, self.V, self.k))
                L_prior += (
                    -self.lambda_ / 2 * ((rho_trans[1:] - rho_trans[:-1]) ** 2).sum()
                )
            loss += L_prior
            loss = -loss

        return loss, L_pos, L_neg, L_prior

    def get_embeddings(self):
        """Gets trained embeddings and reshapes them into (T, V, k)"""
        embeddings = self.rho.cpu().weight.data.reshape(
            (self.T, len(self.dictionary), self.k)
        )
        self.embeddings = embeddings
        return self.embeddings

    def neighborhood(self, v, t, n=20):
        """Finds the neighborhood of terms around `v` in timestep `t`"""
        if self.embeddings is None:
            self.get_embeddings()
        idx = self.dictionary[v]
        rho_v = self.embeddings[t][idx].reshape((1, -1))
        rho = self.embeddings[t].T
        sim = np.dot(np.sign(rho_v), rho).flatten()
        sim /= np.sqrt((rho_v ** 2).sum())
        sim /= np.sqrt((rho ** 2).sum(axis=0).flatten())
        return [self.dictionary_reverse[i] for i in np.argsort(sim)[::-1][:n]]

    def absolute_drift(self, n=50):
        """Find the top drifting terms"""
        if self.embeddings is None:
            self.get_embeddings()
        change = np.sqrt(
            ((self.embeddings[-1] - self.embeddings[0]) ** 2).sum(axis=1)
        ).flatten()
        return sorted(
            list(zip(change, self.dictionary_reverse.values())), reverse=True
        )[:n]
