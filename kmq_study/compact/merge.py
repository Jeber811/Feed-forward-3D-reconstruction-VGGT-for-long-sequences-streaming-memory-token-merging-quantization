"""M: token merging of the frames STORED in memory (not of the frame being processed).

Merging the tokens a frame is computed with (as FastVGGT does) saves compute but not peak memory;
what grows with the frame count is the stored keys/values, so those are what get merged.

ToMe-style bipartite soft matching on the keys of one frame, one layer at a time: tokens are split
alternately into sets A and B, each A token is matched to its most similar B token (cosine of the
keys, all heads), and the best-matched A tokens are averaged into their match. Repeated until
(1 - ratio) of the patch tokens remain. The camera/register tokens are never merged.
A merged token of size s gets +log(s) on its attention logit (ToMe's proportional attention), so
it weighs as much as the s tokens it replaces.
"""
import torch


def merge_frame(k, v, ratio, n_special):
    """k, v: [1, H, P, D] of one frame (after RoPE). Returns k', v', logw [P'] (None if unmerged)."""
    n_patch = k.shape[2] - n_special
    target = max(1, round(n_patch * (1.0 - ratio)))
    if ratio <= 0 or target >= n_patch:
        return k, v, None
    dtype = k.dtype
    sk, sv = k[:, :, :n_special], v[:, :, :n_special]
    pk, pv = k[:, :, n_special:].float(), v[:, :, n_special:].float()
    size = torch.ones(n_patch, device=k.device)

    while pk.shape[2] > target:
        n = pk.shape[2]
        r = min(n - target, n // 2)
        metric = pk[0].transpose(0, 1).reshape(n, -1)                       # [n, H*D]
        metric = metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        a, b = metric[0::2], metric[1::2]
        best, match = (a @ b.T).max(-1)                                     # for each A: best B
        order = best.argsort(descending=True)
        src, keep = order[:r], order[r:]
        dst = match[src]

        ka, kb = pk[:, :, 0::2], pk[:, :, 1::2]
        va, vb = pv[:, :, 0::2], pv[:, :, 1::2]
        sa, sb = size[0::2], size[1::2]
        w = sa[src]
        new_sb = sb.index_add(0, dst, w)
        # size-weighted average; sizes broadcast over the head dim D
        kb = (kb * sb[:, None]).index_add(2, dst, ka[:, :, src] * w[:, None]) / new_sb[:, None]
        vb = (vb * sb[:, None]).index_add(2, dst, va[:, :, src] * w[:, None]) / new_sb[:, None]
        pk = torch.cat([ka[:, :, keep], kb], 2)
        pv = torch.cat([va[:, :, keep], vb], 2)
        size = torch.cat([sa[keep], new_sb])

    logw = torch.cat([size.new_zeros(n_special), size.log()])
    return torch.cat([sk, pk.to(dtype)], 2), torch.cat([sv, pv.to(dtype)], 2), logw.half()
