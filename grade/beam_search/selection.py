import torch


def select_topk(candidates, probs):
    candidate_prob_pair = [(candidates[idx], probs[idx]) for idx in range(len(candidates))]
    candidate_prob_pair = sorted(candidate_prob_pair, key=lambda x:x[1], reverse=True)

    return candidate_prob_pair


def select_indices(scores, top_k, temperature, direct_probabilities):
    score_tensor = torch.as_tensor(scores, dtype=torch.float64)
    if score_tensor.ndim != 1 or score_tensor.numel() == 0:
        raise ValueError("probs.json must contain a non-empty list of scores")
    if not torch.isfinite(score_tensor).all():
        raise ValueError("probs.json contains a non-finite score")
    count = min(top_k, score_tensor.numel())
    if count <= 0:
        return []
    if direct_probabilities:
        if (score_tensor < 0).any() or score_tensor.sum() <= 0:
            raise ValueError("3-node probabilities must be non-negative and sum to a positive value")
        weights = score_tensor / score_tensor.sum()
    else:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        weights = torch.softmax((score_tensor - score_tensor.max()) / temperature, dim=0)
    return torch.multinomial(weights, count, replacement=False).tolist()


