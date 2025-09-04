def accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    return (pred.argmax(dim=-1) == y).float().mean().item()


def effective_rank(z: torch.Tensor, eps: float = 1e-12) -> float:
    """Shannon‐entropy based effective rank (Roy & Vetterli, 2007).

    The implementation follows the definition

        erank = exp(H) / d,

    where ``H`` is the Shannon entropy of the normalised singular values and
    ``d`` is the dimensionality of the feature matrix (number of columns).

    A numerical epsilon is used to avoid ``log(0)``.
    """
    # Singular values of the feature matrix (GPU-friendly & differentiable).
    _, s, _ = torch.linalg.svd(z, full_matrices=False)

    # Normalised (probability) spectrum.
    p = (s / s.sum()).clamp_min(eps)

    # Shannon entropy – bring to Python float early to safely use math.exp.
    h: float = -(p * p.log()).sum().item()

    # Effective rank.
    return math.exp(h) / z.size(1)
