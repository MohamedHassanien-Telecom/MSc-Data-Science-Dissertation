"""Communication losses of the joint training objective.

The task term (cross-entropy of the detector head) is added in src/train.py.
"""
import torch.nn.functional as F

def loss_rec(x, x_hat):
    """Reconstruction loss L_rec = mean squared error between x and x_hat."""
    return F.mse_loss(x_hat, x)

def loss_sem(z, z_prime):
    """Semantic loss L_sem = (1 - cos(z, z'))/2, averaged over the batch."""
    return ((1.0 - F.cosine_similarity(z, z_prime, dim=1))/2).mean()

def composite_loss(x, x_hat, z, z_prime, lam1, lam2):
    """lam1 * L_rec + lam2 * L_sem.

    Returns the weighted loss and the two unweighted terms as floats (for logging).
    """
    lr = loss_rec(x, x_hat)
    ls = loss_sem(z, z_prime)
    return lam1*lr + lam2*ls, lr.item(), ls.item()
