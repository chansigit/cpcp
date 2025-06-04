import torch
import ot
from tqdm.autonotebook import tqdm
import time


def normdist(M):
    return (M - M.min()) / (M.max() - M.min() +1e-8)


def mb_esrFGW(
    X_feat, X_geom, Y_feat, Y_geom,
    n_iter=100, batch_size=128,
    alpha=0.5, eps=1e-2, topk_ratio=1.0, min_cutoff = 1e-8,
    device="cpu"
):
    """
    Vectorized approximation of global FGW transport using mini-batches and sparse COO accumulation.
    """
    N, M = X_feat.shape[0], Y_feat.shape[0]
    X_feat, X_geom = X_feat.to(device), X_geom.to(device)
    Y_feat, Y_geom = Y_feat.to(device), Y_geom.to(device)
    assert not torch.isnan(X_feat).any(), "X_feat contains nan"
    assert not torch.isnan(Y_feat).any(), "Y_feat contains nan"
    assert not torch.isnan(X_geom).any(), "X_geom contains nan"
    assert not torch.isnan(Y_geom).any(), "Y_geom contains nan"
    assert 0.0 < topk_ratio <= 1.0, "topk_ratio must be in (0, 1]"

    # ======================= Mini-batch FGW Iterations =======================
    t0 = time.time()
    p = torch.full((batch_size,), 1.0 / batch_size, device=device) # marginal distribution
    all_indices = []
    all_values = []
    for rounds in tqdm(range(n_iter)):
        idx_i = torch.randint(0, N, (batch_size,), device=device)
        idx_j = torch.randint(0, M, (batch_size,), device=device)
        xi_feat, yi_feat = X_feat[idx_i], Y_feat[idx_j]
        xi_geom, yi_geom = X_geom[idx_i], Y_geom[idx_j]

        # Compute & normalize pairwise distances
        C1 = normdist( torch.cdist(xi_geom, xi_geom, p=2).pow(2) )
        C2 = normdist( torch.cdist(yi_geom, yi_geom, p=2).pow(2) )
        M_feat = normdist( torch.cdist(xi_feat, yi_feat, p=2).pow(2) )
        
        # Solve semi-relaxed entropic FGW
        T_batch = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein(
            M=M_feat, C1=C1, C2=C2, p=p, epsilon=eps, alpha=alpha, loss_fun="square_loss", log=False)
        if torch.isnan(T_batch).any():
            print(f"Error! T_batch contains nan at iteration {rounds}, returning this mini batch.")
            return T_batch

        # Vectorized index mapping: (batch_size x batch_size) => (global_i, global_j)
        I = idx_i.view(-1, 1).expand(-1, batch_size)  # shape (B, B)
        J = idx_j.view(1, -1).expand(batch_size, -1)  # shape (B, B)
        indices = torch.stack([I.reshape(-1), J.reshape(-1)], dim=0)  # shape (2, B*B)
        values = T_batch.reshape(-1)  # shape (B*B,)

        # Optional: apply top-k sparsification
        if topk_ratio < 1.0:
            k = int(values.numel() * topk_ratio)
            topk_vals, topk_idx = torch.topk(values, k)
            values = topk_vals
            indices = indices[:, topk_idx]
        
        # Add values to the full list
        all_indices.append(indices)
        all_values.append(values)

        
    print( f"Iterations time: { time.time()-t0 :.2f} secs")

    # ======================= Mini-batch FGW Iterations =======================
    t0 = time.time()
    # Concatenate all mini-batch triplets
    all_indices = torch.cat(all_indices, dim=1)  # shape (2, total_nnz)
    all_values  = torch.cat(all_values,  dim=0)  # shape (total_nnz,)

    # Build sparse matrix with repeated indices
    # Coalesce: sum duplicates (i, j), to be averaged later
    T_sparse = torch.sparse_coo_tensor(all_indices, all_values, size=(N, M), device=device).coalesce()

    # Normalize by occurrence count to estimate expected transport values (average instead of sum)
    # # The occurrence count is how many times each (i,j) appeared across mini-batches
    counts = torch.sparse_coo_tensor(all_indices, torch.ones_like(all_values), size=(N, M), device=device).coalesce()

    # Find the average T
    avg_values   = T_sparse.values() / (counts.values() + 1e-8)

    # min cut-off filtering
    new_indices = T_sparse.indices()
    new_values = avg_values
    if min_cutoff is not None:
        mask = avg_values >= min_cutoff  # Boolean mask for values to keep
        new_indices = new_indices[:, mask]
        new_values  = new_values[mask]

    T_avg_sparse = torch.sparse_coo_tensor(new_indices, new_values, size=(N, M), device=device).coalesce()
    print( f"Aggregation time: {time.time()-t0 :.2f} secs")

    return T_avg_sparse # shape: sparse [N, M], coalesced, with averaged values
