import torch
import ot
from tqdm.autonotebook import tqdm
import time
from typing import Sequence


def normdist(M):
    return (M - M.min()) / (M.max() - M.min() +1e-8)


def prepare_prohibited_mask_matrix(
    X_celltypes: 'Sequence[str]',
    Y_celltypes: 'Sequence[str]',
    prohibited_transitions,
    device
):
    """
    Precompute the prohibited mask matrix and integer-encoded celltype arrays.

    Args:
        X_celltypes: Sequence of cell type annotations for adata0 (e.g., list, numpy array, or pandas Series of str)
        Y_celltypes: Sequence of cell type annotations for adata1 (e.g., list, numpy array, or pandas Series of str)
        prohibited_transitions: List of (celltype0, celltype1) pairs to prohibit
        device: torch device

    Returns:
        X_celltypes_int: torch.LongTensor of shape (N,)
        Y_celltypes_int: torch.LongTensor of shape (M,)
        prohibited_mask_matrix: torch.BoolTensor of shape (num_types, num_types)
    """
    # Number of cells in each dataset
    N = len(X_celltypes)
    M = len(Y_celltypes)
    
    # Find all unique cell types across both datasets
    all_celltypes = sorted(list(set(X_celltypes) | set(Y_celltypes)))
    # Map each cell type string to a unique integer index
    celltype_to_int = {celltype: i for i, celltype in enumerate(all_celltypes)}
    
    # Convert the cell type annotations for each cell to integer indices
    X_celltypes_int = torch.tensor(
        [celltype_to_int[c] for c in X_celltypes], dtype=torch.long, device=device
    )
    Y_celltypes_int = torch.tensor(
        [celltype_to_int[c] for c in Y_celltypes], dtype=torch.long, device=device
    )
    
    # The number of unique cell types
    num_types = len(all_celltypes)
    # Initialize a boolean matrix of shape (num_types, num_types)
    # Entry (i, j) will be True if transition from cell type i to j is prohibited
    prohibited_mask_matrix = torch.zeros(
        (num_types, num_types), dtype=torch.bool, device=device
    )
    # Mark prohibited transitions in the mask matrix
    for ct1, ct2 in prohibited_transitions:
        if ct1 in celltype_to_int and ct2 in celltype_to_int:
            i, j = celltype_to_int[ct1], celltype_to_int[ct2]
            prohibited_mask_matrix[i, j] = True
    
    # Return the integer-encoded cell type arrays and the mask matrix
    return X_celltypes_int, Y_celltypes_int, prohibited_mask_matrix
    
def mb_esrFGW(
    X_feat, X_geom, Y_feat, Y_geom,
    X_celltypes=None, Y_celltypes=None, prohibited_transitions=None, penalty_strength=10.0,
    n_iter=100, batch_size=128,
    alpha=0.5, eps=1e-2, topk_ratio=1.0, min_cutoff = 1e-8,
    device="cpu"
):
    """
    mini-batch version of entropic semi-relaxed FGW with blacklist support,
    using approximation of global FGW transport using mini-batches and sparse COO accumulation.
    Args:
        X_feat: torch.Tensor of shape (N, d_feat)
        X_geom: torch.Tensor of shape (N, d_geom)
        Y_feat: torch.Tensor of shape (M, d_feat)
        Y_geom: torch.Tensor of shape (M, d_geom)
        X_celltypes: list of cell types for X, None if no constraints   
        Y_celltypes: list of cell types for Y, None if no constraints
        prohibited_transitions: list of prohibited transitions, None if no constraints
        penalty_strength: penalty strength for prohibited transitions
        n_iter: number of iterations
        batch_size: batch size
        alpha: alpha for entropic regularization
        eps: epsilon for entropic regularization
        topk_ratio: topk ratio for sparse approximation
        min_cutoff: min cutoff for sparse approximation
        device: device to use
    Returns:
        T_avg_sparse: torch.sparse.FloatTensor of shape (N, M)
        
    """
    N, M = X_feat.shape[0], Y_feat.shape[0]
    X_feat, X_geom = X_feat.to(device), X_geom.to(device)
    Y_feat, Y_geom = Y_feat.to(device), Y_geom.to(device)
    assert not torch.isnan(X_feat).any(), "X_feat contains nan"
    assert not torch.isnan(Y_feat).any(), "Y_feat contains nan"
    assert not torch.isnan(X_geom).any(), "X_geom contains nan"
    assert not torch.isnan(Y_geom).any(), "Y_geom contains nan"
    assert 0.0 < topk_ratio <= 1.0, "topk_ratio must be in (0, 1]"

    # ======================= Pre-computation for biological constraints =======================
    use_constraints = (
        X_celltypes is not None
        and Y_celltypes is not None
        and prohibited_transitions is not None
    )
    if use_constraints:
        assert len(X_celltypes) == N, "X_celltypes must have the same length as X_feat"
        assert len(Y_celltypes) == M, "Y_celltypes must have the same length as Y_feat"
        # X_celltypes_int: (N,), Y_celltypes_int: (M,), prohibited_mask_matrix: (num_types, num_types)
        X_celltypes_int, Y_celltypes_int, prohibited_mask_matrix = prepare_prohibited_mask_matrix(
            X_celltypes, Y_celltypes, prohibited_transitions, device
        )
    

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
        
        if use_constraints:
            batch_X_types_int = X_celltypes_int[idx_i]
            batch_Y_types_int = Y_celltypes_int[idx_j]
            batch_prohibited_mask = prohibited_mask_matrix[
                batch_X_types_int[:, None], batch_Y_types_int[None, :]
            ]
            M_feat[batch_prohibited_mask] *= penalty_strength
        
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


def fb_esrFGW(
    X_feat, X_geom, Y_feat, Y_geom,
    X_celltypes=None, Y_celltypes=None, prohibited_transitions=None, penalty_strength=10.0,
    alpha=0.5, eps=1e-2, min_cutoff=1e-8, device="cpu"
):
    """
    Non-mini-batch (full) version of entropic semi-relaxed FGW with blacklist support.
    Computes the full transport matrix in one go.
    Args:
        X_feat: torch.Tensor of shape (N, d_feat)
        X_geom: torch.Tensor of shape (N, d_geom)
        Y_feat: torch.Tensor of shape (M, d_feat)
        Y_geom: torch.Tensor of shape (M, d_geom)
        X_celltypes: list of cell types for X, None if no constraints
        Y_celltypes: list of cell types for Y, None if no constraints
        prohibited_transitions: list of prohibited transitions, None if no constraints
        penalty_strength: penalty strength for prohibited transitions
        alpha: alpha for entropic regularization
        eps: epsilon for entropic regularization
        min_cutoff: min cutoff for sparse approximation
        device: device to use
    Returns:
        T: torch.Tensor of shape (N, M)
    """
    N, M = X_feat.shape[0], Y_feat.shape[0]
    X_feat, X_geom = X_feat.to(device), X_geom.to(device)
    Y_feat, Y_geom = Y_feat.to(device), Y_geom.to(device)
    assert not torch.isnan(X_feat).any(), "X_feat contains nan"
    assert not torch.isnan(Y_feat).any(), "Y_feat contains nan"
    assert not torch.isnan(X_geom).any(), "X_geom contains nan"
    assert not torch.isnan(Y_geom).any(), "Y_geom contains nan"

    # ======================= Pre-computation for biological constraints =======================
    use_constraints = (
        X_celltypes is not None
        and Y_celltypes is not None
        and prohibited_transitions is not None
    )
    if use_constraints:
        assert len(X_celltypes) == N, "X_celltypes must have the same length as X_feat"
        assert len(Y_celltypes) == M, "Y_celltypes must have the same length as Y_feat"
        # X_celltypes_int: (N,), Y_celltypes_int: (M,), prohibited_mask_matrix: (num_types, num_types)
        X_celltypes_int, Y_celltypes_int, prohibited_mask_matrix = prepare_prohibited_mask_matrix(
            X_celltypes, Y_celltypes, prohibited_transitions, device
        )

    # Compute & normalize pairwise distances
    C1 = normdist(torch.cdist(X_geom, X_geom, p=2).pow(2))
    C2 = normdist(torch.cdist(Y_geom, Y_geom, p=2).pow(2))
    M_feat = normdist(torch.cdist(X_feat, Y_feat, p=2).pow(2))

    if use_constraints:
        # Broadcast to (N, M) mask
        prohibited_mask = prohibited_mask_matrix[
            X_celltypes_int[:, None], Y_celltypes_int[None, :]
        ]
        M_feat[prohibited_mask] *= penalty_strength

    # Marginal distribution for semi-relaxed (uniform over N)
    p = torch.full((N,), 1.0 / N, device=device)

    # Solve semi-relaxed entropic FGW
    T = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein(
        M=M_feat, C1=C1, C2=C2, p=p, epsilon=eps, alpha=alpha, loss_fun="square_loss", log=False
    )
    if torch.isnan(T).any():
        print("Error! T contains nan.")
        return T

    # min cut-off filtering (optional, for sparsity)
    if min_cutoff is not None:
        T[T < min_cutoff] = 0.0

    return T
