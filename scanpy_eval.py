import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def build_spatial_knn(adata, spatial_key="spatial", n_neighbors=10):
    """
    Build spatial KNN graph from AnnData object.

    Parameters
    ----------
    adata : AnnData
        AnnData object with spatial coordinates stored in .obsm[spatial_key].
    spatial_key : str
        Key for spatial coordinates in adata.obsm.
    n_neighbors : int
        Number of neighbors to use for KNN.

    Returns
    -------
    knn_pairs : list of tuple
        List of (i, j) pairs where j is one of the k nearest neighbors of i.
    """
    coords = adata.obsm[spatial_key]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='auto').fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    knn_pairs = []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  # skip self
            knn_pairs.append((i, int(j)))

    return knn_pairs


def locality_preservation_error(
    adata0, adata1,
    trans_plan,
    knn_pairs,
    spatial_key="spatial",
    norm='l2'
):
    """
    Compute local geometric distortion after applying hard-matching transport.

    Parameters
    ----------
    adata0 : AnnData
        Source AnnData object.
    adata1 : AnnData
        Target AnnData object.
    trans_plan : list, np.ndarray, or torch.Tensor
        A list or array where trans_plan[i] is the index in adata1 that source i maps to.
    knn_pairs : list of tuple
        List of (i, j) index pairs in source spatial graph.
    spatial_key : str
        Key to get spatial coordinates from .obsm.
    norm : str
        'l2' or 'l1' distance metric.

    Returns
    -------
    float
        Mean local distance error after mapping.
    """
    if 'torch' in str(type(trans_plan)):
        trans_plan = trans_plan.detach().cpu().numpy()

    X0 = adata0.obsm[spatial_key]
    X1 = adata1.obsm[spatial_key]

    # Use target coordinates for mapped source
    mapped_X0 = X1[trans_plan]  # shape: (n_source, 3)

    errors = []
    for i, j in knn_pairs:
        d0 = np.linalg.norm(X0[i] - X0[j], ord=2 if norm == 'l2' else 1)
        d1 = np.linalg.norm(mapped_X0[i] - mapped_X0[j], ord=2 if norm == 'l2' else 1)
        
        errors.append(  abs(d0 - d1))

    return np.mean(errors)


def contextual_label_consistency(adata0, adata1, trans_plan, label_key='Level1_250527', spatial_key='std_3D', k=10):
    """
    Adapted from Spateo paper developed by Yifan Lu. Contextual label consistency.
    
    Parameters
    ----------
    adata0 : AnnData. Source AnnData object.
    adata1 : AnnData. Target AnnData object.
    trans_plan : torch.Tensor. Transport plan. Either sparse or dense torch tensor.
    label_key : str. Key for label in adata0.obs.
    spatial_key : str. Key for spatial coordinates in adata0.obsm.
    k : int. Number of neighbors to use for k-NN.

    Returns
    -------
    final_score : float. Contextual label consistency score.    
    """
    total_score = 0.0
    N = adata0.n_obs

    # Convert transport plan to indices
    if trans_plan.is_sparse:
        trans_plan = trans_plan.to_dense().to('cpu')
    else:
        trans_plan = trans_plan.to('cpu')
    mapped_target_idx = trans_plan.argmax(dim=1).numpy()

    # Build k-NN graph using NearestNeighbors
    coords = adata0.obsm[spatial_key]
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(coords)
    _, indices = nbrs.kneighbors(coords)

    for i in  (range(N)):
        source_label = adata0.obs[label_key].iloc[i]
        target_idx = mapped_target_idx[i]
        target_label = adata1.obs[label_key].iloc[target_idx]

        if source_label == target_label:
            # Get k-NN indices for the source point, excluding itself
            knn_indices = indices[i][1:]  # skip self

            # Map k-NN to their targets
            knn_target_indices = [mapped_target_idx[j] for j in knn_indices]

            # Calculate distances
            source_target_coord = adata1.obsm[spatial_key][target_idx]
            knn_target_coords = adata1.obsm[spatial_key][knn_target_indices]

            # Debugging: Print coordinates and indices
            #print(f"Source index: {i}, Target index: {target_idx}")
            #print(f"Source target coord: {source_target_coord}")
            #print(f"k-NN target coords: {knn_target_coords}")

            distances = np.linalg.norm(knn_target_coords - source_target_coord, axis=1)

            # Calculate score for this point
            score = distances.sum() / k
            total_score += score

    # Calculate final score
    final_score = total_score / N
    return final_score


def transport_evaluation(adata0, adata1, trans_plan, label_key='Level1_250527', spatial_key='std_3D'):
    """
    Evaluate the transport plan.

    Parameters
    ----------
    adata0 : AnnData. Source AnnData object.
    adata1 : AnnData. Target AnnData object.
    trans_plan : torch.Tensor. Transport plan. Either sparse or dense torch tensor.
    label_key : str. Key for label in adata0.obs.
    spatial_key : str. Key for spatial coordinates in adata0.obsm.

    Returns
    -------
    spa_err : float. Mean local distance error after mapping.
    """
    # find top-k elements for the transplan matrix
    if trans_plan.is_sparse:
        trans_plan = trans_plan.to_dense().to('cpu')
    else:
        trans_plan = trans_plan.to('cpu')
    plan_topk = torch.topk(trans_plan, k=1, dim=1)

    # find mapped targets for source samples, and grab source/mapped labels
    mapped_target_idx =  plan_topk.indices.reshape(-1)
    source_labels  = adata0.obs[label_key]
    source_mapped_labels = adata1.obs.iloc[mapped_target_idx][label_key]
    
    knn_pairs = build_spatial_knn(adata0, spatial_key=spatial_key, n_neighbors=10)
    spa_err   = locality_preservation_error(adata0, adata1, trans_plan=plan_topk.indices.view(-1), knn_pairs=knn_pairs, spatial_key=spatial_key)
    print(f'Locality preservation error:\t{spa_err:>10.4f}')
    
    clc = contextual_label_consistency(adata0, adata1, trans_plan=trans_plan, label_key=label_key, spatial_key=spatial_key)
    print(f'Spateo-CLC error: \t{clc:>10.4f}')
    
    # consistency = (source_labels.values.astype(str) == source_mapped_labels.values.astype(str)).mean()    
    consistency = (source_labels.values.astype(str) == source_mapped_labels.values.astype(str)).mean()
    print(f'Cell type label Consistency: \t{consistency:>10.4f}')

    n_transported = torch.sum( torch.topk(trans_plan.to_dense(), k=1, dim=1).values.flatten() >0  )
    print(f'N_transported: \t{n_transported.item():>10}' )
    
    return spa_err, clc, consistency, n_transported 



def random_transport_baseline(adata0, adata1, label_key='Level1_250527', spatial_key='std_3D'):
    """
    Compute random transport baseline for locality preservation and label consistency.

    Parameters
    ----------
    adata0 : AnnData. Source AnnData object.
    adata1 : AnnData. Target AnnData object.
    label_key : str. Key for label in adata0.obs.
    spatial_key : str. Key for spatial coordinates in adata0.obsm.

    Returns
    -------
    rand_err : float. Mean local distance error after mapping.
        Mean local distance error after mapping.
    consistency : float
        Label consistency after mapping.
    """
    import random
    # Initialize a zero matrix
    trans_plan = torch.zeros((adata0.n_obs, adata1.n_obs))
    
    # Get random target indices for each source index
    random_indices = random.sample(range(adata1.n_obs), adata0.n_obs)
    
    # Set one-hot values
    for i, target_idx in enumerate(random_indices):
        trans_plan[i, target_idx] = 1
    
    return transport_evaluation(adata0, adata1, trans_plan, label_key=label_key, spatial_key=spatial_key)



def nn_transport_baseline(adata0, adata1, label_key='Level1_250527', spatial_key='std_3D', embedding_key='Z_dynode'):
    """
    Compute nearest neighbor transport baseline for locality preservation and label consistency.
    """
    # Initialize a zero matrix
    trans_plan = torch.zeros((adata0.n_obs, adata1.n_obs))
    
    # Get top-k indices (assuming k=1 for nearest neighbor)
    nn_indices = torch.topk(-torch.cdist(torch.from_numpy(adata0.obsm[embedding_key]), 
                                         torch.from_numpy(adata1.obsm[embedding_key])), 
                            k=1, dim=1).indices.view(-1)
    
    # Set one-hot values
    for i, target_idx in enumerate(nn_indices):
        trans_plan[i, target_idx] = 1
    
    return transport_evaluation(adata0, adata1, trans_plan, label_key=label_key, spatial_key=spatial_key)


