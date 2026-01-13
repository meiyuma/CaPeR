import torch.nn.functional as F
import torch
from torch.nn import BCELoss, BCEWithLogitsLoss
from itertools import product


class RankLoss:

    @staticmethod
    def softmax_ce_loss(y_pred, *args, **kwargs):
        return F.cross_entropy(y_pred, torch.zeros((y_pred.size(0),)).long().cuda())

    @staticmethod
    def list_net(y_pred, y_true=None, weights=None, padded_value_indicator=-100, eps=1e-10):
        """
        [MODIFIED] ListNet with Evidence Weighting
        weights: [batch_size, slate_length] containing confidence scores (0.0 - 1.0)
        """
        if y_true is None:
            y_true = torch.zeros_like(y_pred).to(y_pred.device)
            y_true[:, 0] = 1

        preds_smax = F.softmax(y_pred, dim=1)
        true_smax = F.softmax(y_true, dim=1)

        preds_smax = preds_smax + eps
        preds_log = torch.log(preds_smax)

        # Base Loss
        element_wise_loss = -torch.sum(true_smax * preds_log, dim=1)

        # [NEW] Apply Evidence Weights
        if weights is not None:
            # weights shape is [batch, slate_length]
            # We average the weights across the slate to get a batch-level weight
            # Or use the weight of the positive sample (index 0)
            batch_weights = weights.mean(dim=1)
            element_wise_loss = element_wise_loss * batch_weights

        return torch.mean(element_wise_loss)

    @staticmethod
    def rank_net(y_pred, y_true=None, weights=None, padded_value_indicator=-100, weight_by_diff=False,
                 weight_by_diff_powed=False):
        """
        [MODIFIED] RankNet with Evidence Weighting
        """
        if y_true is None:
            y_true = torch.zeros_like(y_pred).to(y_pred.device)
            y_true[:, 0] = 1

        document_pairs_candidates = list(product(range(y_true.shape[1]), repeat=2))

        pairs_true = y_true[:, document_pairs_candidates]
        selected_pred = y_pred[:, document_pairs_candidates]

        true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
        pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]

        the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))

        pred_diffs = pred_diffs[the_mask]

        weight = None
        if weight_by_diff:
            abs_diff = torch.abs(true_diffs)
            weight = abs_diff[the_mask]
        elif weight_by_diff_powed:
            true_pow_diffs = torch.pow(pairs_true[:, :, 0], 2) - torch.pow(pairs_true[:, :, 1], 2)
            abs_diff = torch.abs(true_pow_diffs)
            weight = abs_diff[the_mask]

        # [NEW] Apply Evidence Weights to pairs
        if weights is not None:
            # We need to map weights to pairs.
            # weights: [batch, slate_len] -> pairs_weights: [batch, slate_len^2]
            # We use the weight of the "winner" (left side) in the pair
            pairs_weights = weights[:, [x[0] for x in document_pairs_candidates]]

            # Mask weights same as diffs
            masked_weights = pairs_weights[the_mask]

            # If 'weight' was already set by diffs, multiply them
            if weight is None:
                weight = masked_weights
            else:
                weight = weight * masked_weights

        true_diffs = (true_diffs > 0).type(torch.float32)
        true_diffs = true_diffs[the_mask]

        return BCEWithLogitsLoss(weight=weight)(pred_diffs, true_diffs)

    # LambdaLoss can be modified similarly, but RankNet/ListNet are most commonly used here.
    # For now, we only patch ListNet/RankNet as per your request.
    @staticmethod
    def lambda_loss(y_pred, y_true=None, eps=1e-10, padded_value_indicator=-100, weighing_scheme=None, k=None,
                    sigma=1., mu=10., reduction="mean", reduction_log="binary"):
        # Original implementation remains...
        if y_true is None:
            y_true = torch.zeros_like(y_pred).to(y_pred.device)
            y_true[:, 0] = 1

        device = y_pred.device
        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true.sort(descending=True, dim=-1)
        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)

        if weighing_scheme != "ndcgLoss1_scheme":
            padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

        ndcg_at_k_mask = torch.zeros((y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
        ndcg_at_k_mask[:k, :k] = 1

        true_sorted_by_preds.clamp_(min=0.)
        y_true_sorted.clamp_(min=0.)

        pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
        D = torch.log2(1. + pos_idxs.float())[None, :]
        maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=eps)
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        if weighing_scheme is None:
            weights = 1.
        else:
            weights = globals()[weighing_scheme](G, D, mu, true_sorted_by_preds)

        scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
        scores_diffs.masked_fill(torch.isnan(scores_diffs), 0.)
        weighted_probas = (torch.sigmoid(sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)

        if reduction_log == "natural":
            losses = torch.log(weighted_probas)
        elif reduction_log == "binary":
            losses = torch.log2(weighted_probas)
        else:
            raise ValueError("Reduction logarithm base can be either natural or binary")

        if reduction == "sum":
            loss = -torch.sum(losses[padded_pairs_mask & ndcg_at_k_mask])
        elif reduction == "mean":
            loss = -torch.mean(losses[padded_pairs_mask & ndcg_at_k_mask])
        else:
            raise ValueError("Reduction method can be either sum or mean")

        return loss


def ndcgLoss1_scheme(G, D, *args): return (G / D)[:, :, None]


def ndcgLoss2_scheme(G, D, *args):
    pos_idxs = torch.arange(1, G.shape[1] + 1, device=G.device)
    delta_idxs = torch.abs(pos_idxs[:, None] - pos_idxs[None, :])
    deltas = torch.abs(torch.pow(torch.abs(D[0, delta_idxs - 1]), -1.) - torch.pow(torch.abs(D[0, delta_idxs]), -1.))
    deltas.diagonal().zero_()
    return deltas[None, :, :] * torch.abs(G[:, :, None] - G[:, None, :])


def lambdaRank_scheme(G, D, *args):
    return torch.abs(torch.pow(D[:, :, None], -1.) - torch.pow(D[:, None, :], -1.)) * torch.abs(
        G[:, :, None] - G[:, None, :])


def ndcgLoss2PP_scheme(G, D, *args): return args[0] * ndcgLoss2_scheme(G, D) + lambdaRank_scheme(G, D)


def rankNet_scheme(G, D, *args): return 1.


def rankNetWeightedByGTDiff_scheme(G, D, *args): return torch.abs(args[1][:, :, None] - args[1][:, None, :])


def rankNetWeightedByGTDiffPowed_scheme(G, D, *args): return torch.abs(
    torch.pow(args[1][:, :, None], 2) - torch.pow(args[1][:, None, :], 2))
