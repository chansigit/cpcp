import torch
import ot
from collections import defaultdict
from tqdm.autonotebook import tqdm

def approximate_sparse_semi_fgw_outdated(
    X_feat, X_geom, Y_feat, Y_geom,
    n_iter=100, batch_size=128,
    alpha=0.5, eps=1e-2, device="cpu"
):
    """
    Approximate global FGW transport matrix using mini-batches and sparse accumulation.
    """
    N, M = X_feat.shape[0], Y_feat.shape[0]
    X_feat, X_geom = X_feat.to(device), X_geom.to(device)
    Y_feat, Y_geom = Y_feat.to(device), Y_geom.to(device)
    assert not torch.isnan(X_feat).any(), "X_feat contains nan"
    assert not torch.isnan(Y_feat).any(), "Y_feat contains nan"
    assert not torch.isnan(X_geom).any(), "X_geom contains nan"
    assert not torch.isnan(Y_geom).any(), "Y_geom contains nan"


    # Accumulate sparse triplets
    t0= time.time()
    transport_sum = defaultdict(float)
    count_sum = defaultdict(int)

    for i in tqdm(range(n_iter)):
        idx_i = torch.randint(0, N, (batch_size,), device=device)
        idx_j = torch.randint(0, M, (batch_size,), device=device)

        xi_feat, yi_feat = X_feat[idx_i], Y_feat[idx_j]
        xi_geom, yi_geom = X_geom[idx_i], Y_geom[idx_j]

        # Compute cost matrices
        C1 = torch.cdist(xi_geom, xi_geom, p=2).pow(2)
        C2 = torch.cdist(yi_geom, yi_geom, p=2).pow(2)
        M_feat = torch.cdist(xi_feat, yi_feat, p=2).pow(2)
        p = torch.full((batch_size,), 1.0 / batch_size, device=device)
        
        # normalize the cost matrices
        M_feat = (M_feat - M_feat.min())/ (M_feat.max()-M_feat.min())
        C1 = (C1 - C1.min())/ (C1.max() - C1.min())
        C2 = (C2 - C2.min())/ (C2.max() - C2.min())
        
        # Call semi-relaxed entropic FGW solver
        T_batch_mb = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein(
            M=M_feat,C1=C1,C2=C2,p=p,
            loss_fun="square_loss",
            epsilon=eps,
            alpha=alpha,
            log=False
        )
        
        if torch.isnan(T_batch_mb).any():
            print(f"Error! T_batch_mb contains nan at iteration {i}, returning this mini batch.")
            return T_batch_mb

        # Sparse COO accumulation
        for m in range(batch_size):
            i = idx_i[m].item()
            for n in range(batch_size):
                j = idx_j[n].item()
                key = (i, j)
                transport_sum[key] += T_batch_mb[m, n].item()
                count_sum[key] += 1
    print( f"Iterations cost {time.time()-t0 :.2f} secs")
    
    t0= time.time()
    # Final sparse matrix
    coords, values = [], []
    for (i, j), val in transport_sum.items():
        avg_val = val / count_sum[(i, j)]
        coords.append([i, j])
        values.append(avg_val)

    indices = torch.tensor(coords, dtype=torch.long, device=device).T  # [2, nnz]
    values = torch.tensor(values, dtype=torch.float32, device=device)
    T_sparse = torch.sparse_coo_tensor(indices, values, size=(N, M), device=device)
    print( f"Aggregation cost {time.time()-t0 :.2f} secs")

    return T_sparse.coalesce()
