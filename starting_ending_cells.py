from sklearn.metrics import pairwise_distances
from collections import deque
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans

def generate_init_from_kmeans(p0, num_clusters=1000, seed=0):
    """
    Perform KMeans clustering on the normalized point cloud p0 and return the index set of each cluster.

    对标准化的点云 p0 进行 KMeans 聚类，返回每个 cluster 的索引集合。
    
    More information about this function:
    
    This function assists in generating a training plan for the dynode model. 
    
    In the training plan,
    we need to specify the source and target cells for the model to learn dynamics that fit them.
    The source/target batches should:
    1) Not be too large to avoid GPU memory overflow;
    2) Have good locality in the spatial space so the message passing mechanism can work effectively;
    3) Preferably have regular batch sizes to speed up training;
    
    Therefore, we first partition the point cloud into several k-means clusters, and 
    then expand the cluster to a batch of source cells. This function serves as a helper
    to partition the point cloud into clusters.
    
    Parameters:
        p0: np.ndarray, shape (num_points, 3), original point cloud coordinates
        p0: np.ndarray, shape (num_points, 3)，原始点云坐标
        num_clusters: int, number of clusters
        num_clusters: int，聚类数量
        seed: int, random seed
        seed: int，随机种子

    Returns:
        dict: key=cluster_id, value=np.ndarray, indices of points belonging to the cluster
    """
    np.random.seed(seed)
    p0_normalized = (p0 - np.min(p0, axis=0)) / (np.max(p0, axis=0) - np.min(p0, axis=0))
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed).fit(p0_normalized)
    labels = kmeans.labels_
    clusters = {}
    for cluster_id in range(num_clusters):
        clusters[cluster_id] = np.where(labels == cluster_id)[0]
    return clusters


def bfs_sample(p0, n, init_idx, k=10, seed=0):
    """
    Sample n strongly connected points from the normalized point cloud using BFS (breadth-first search), 
    supporting multiple starting points.

    使用 BFS 从标准化后的点云中采样 n 个强连接点，支持多个起始点。
    
    More information about this function:
    
    This function assists in generating a training plan for the dynode model.
    Specifically, it serves as a starting point builder, helping to expand 
    an initial set of points to a set of n strongly connected points.

    此函数用于为 dynode 模型生成训练计划。
    具体来说，它作为起始点构建器，帮助将初始点集扩展为 n 个强连接点的集合。

    Parameters:
        p0: np.ndarray, shape (num_points, 3), original point cloud coordinates, 原始点云坐标
        n: int, desired number of points to sample, 希望采样得到的点数量
        k: int, number of neighbors for constructing the k-NN graph, 构建 k-NN 图的邻居数
        seed: int, random seed
        init_idx: list of int, indices of multiple starting points, 多个起始点索引

    Returns:
        np.ndarray: indices of sampled points, shape (n), 采样点的索引，shape (n)
    """
    np.random.seed(seed)
    # standardize point cloud
    p0_normalized = (p0 - np.min(p0, axis=0)) / (np.max(p0, axis=0) - np.min(p0, axis=0))
    # compute distance matrix 
    dist_matrix = pairwise_distances(p0_normalized)
    # build k-NN connectivity graph
    knn_graph = np.argsort(dist_matrix, axis=1)[:, 1:k+1] # 1:k+1 because the first neighbor is itself

    # 初始化多个起点
    if init_idx is None:
        raise ValueError("init_idx must be provided as a list or array of indices.")
    if not hasattr(init_idx, '__iter__'):
        raise TypeError("init_idx must be a list or array, not a single integer.")
    if any(idx >= p0.shape[0] or idx < 0 for idx in init_idx):
        raise ValueError("init_idx contains out of bounds indices.")

    queue = deque(init_idx)
    visited = set(queue)

    while len(visited) < n and queue:
        current = queue.popleft()
        neighbors = knn_graph[current]
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    if len(visited) < n:
        raise ValueError(f"Only found {len(visited)} connected points, increase k or check standardization.")

    return np.array(list(visited)[:n])



def spatially_balanced_topM(points, values, M, K=30):
    """
    Sample exactly M points evenly from the point cloud `points`
    with high values in the `values` array.
    从点云 `points` 中均匀采样 M 个点，这些点在 `values` 数组中具有较高的值。
    
    The points are first partitioned into K clusters using KMeans, 
    and then M points are selected from each cluster.
    首先使用 KMeans 将点划分为 K 个簇，然后从每个簇中选择 M 个点。

    More information about this function:
    
        This is a helper function to find targets for the dynode model training plan.
        The target construction process will first propose a candidate set of targets with more than M points,
        and then use this function to select M points from the candidate set. This function ensures high-quality 
        targets are selected and distributed evenly across the point cloud.

        这是一个辅助函数，用于为 dynode 模型训练计划寻找目标。
        目标构建过程将首先提出一个包含超过 M 个点的候选目标集，
        然后使用此函数从候选集中选择 M 个点。此函数确保高质量的目标被选择并均匀分布在点云中。
    
    Parameters:
        points: np.ndarray, shape (num_points, 3), original point cloud coordinates
        values: np.ndarray, shape (num_points,), values of the points
        M: int, number of points to sample
        K: int, number of clusters to use for KMeans
        
    Returns:
        np.ndarray: indices of sampled points, shape (M)
    """
    K = K or M
    kmeans = KMeans(n_clusters=K, random_state=0, n_init=10).fit(points)
    labels = kmeans.labels_

    selected = [np.where(labels == k)[0][np.argmax(values[labels == k])]
                for k in range(K) if np.any(labels == k)]
    selected = np.array(selected)

    if len(selected) < M:
        remaining = np.setdiff1d(np.arange(len(values)), selected)
        extra_needed = M - len(selected)
        extra = remaining[np.argsort(-values[remaining])[:extra_needed]]
        selected = np.concatenate([selected, extra])
    elif len(selected) > M:
        selected = selected[np.argsort(-values[selected])[:M]]

    return selected


def topk_target_density_filter(plan, p1, idx0, k=5, 
                               plan_quantile=0.5, 
                               density_quantile=0.05, 
                               desired_target_num=400, 
                               target_part_K=30):
    """
    Find the target points based on starting points and transport plan. 
    Filter the candidate targets based on top-k, quantile cutoff, KDE density, and spatially balanced downsampling.
    这是生成targets points的主函数

    Parameters:
        plan: np.ndarray, shape (num_points, num_points), transport plan
        p1: np.ndarray, shape (num_points, 3), target point cloud coordinates
        idx0: np.ndarray, shape (num_points,), source point indices
        k: int, number of neighbors for top-k selection
        plan_quantile: float, quantile cutoff for plan values. The transport plan must be larger than this value to be considered.
        density_quantile: float, quantile cutoff for density filtering. The density of the target points must be larger than this value to be considered.
        desired_target_num: int, desired number of target points
        target_part_K: int, partition number for target points final selection. See docs of `spatially_balanced_topM` for more details.

    Returns:
        final_idx: np.ndarray，最终 target 索引
        final_values: np.ndarray，最终对应的 plan 值
    """
    # Step 1: based on the transport plan, find the top-k target points for each source point in `idx0`.
    partition_indices = np.argpartition(-plan[idx0], k-1, axis=1)[:, :k]
    topk_values = np.take_along_axis(plan[idx0], partition_indices, axis=1)
    
    # and also find the target points' transport plan values.
    row_indices = np.repeat(np.arange(len(idx0)), k)
    col_indices = partition_indices.ravel()
    values_flat = topk_values.ravel()

    # Step 2: drop those targets with zero transport plan values
    mask1 = values_flat > 0
    col_indices = col_indices[mask1]
    values_flat = values_flat[mask1]
    if len(values_flat) == 0:
        raise ValueError("No valid top-k values found above zero.")

    # Step 3: keep only those targets with transport plan values larger than the quantile cutoff.
    cutoff = np.quantile(values_flat, plan_quantile)
    mask2 = values_flat > cutoff
    col_indices = col_indices[mask2]
    values_flat = values_flat[mask2]
    if len(col_indices) == 0:
        raise ValueError("No candidates passed the quantile cutoff filtering.")

    # and make the target points unique by aggregating the transport plan values of the same target point.
    unique_idx, inverse_indices = np.unique(col_indices, return_inverse=True)
    aggregated_values = np.zeros(len(unique_idx))
    for i, idx in enumerate(inverse_indices):
        aggregated_values[idx] = max(aggregated_values[idx], values_flat[i])

    # Step 4: Spatial KDE filtering
    if len(unique_idx) > 0:
        density = gaussian_kde(p1[unique_idx].T)(p1[unique_idx].T)
        density_mask = density >= np.quantile(density, density_quantile)
        unique_idx = unique_idx[density_mask]
        aggregated_values = aggregated_values[density_mask]

    # Step 5: check if the number of target points is enough.
    if len(unique_idx) < desired_target_num:
        raise ValueError(f"Insufficient target candidates after filtering: "
                         f"found {len(unique_idx)}, required at least {desired_target_num}. "
                         f"Consider increasing `k` or lowering down `plan_quantile`. Current k={k}, plan_quantile={plan_quantile}")

    # Step 6: spatially balanced topM selection.
    selected_indices = spatially_balanced_topM(p1[unique_idx], aggregated_values, M=desired_target_num, K=target_part_K)
    final_idx = unique_idx[selected_indices]
    final_values = aggregated_values[selected_indices]

    return final_idx

