import random
import time

import numpy as np


def renormalize(weights, index):
    """
    :param weights: vector of non negative weights summing to 1.
    :type weights: numpy.array
    :param index: index of the weight to remove
    :type index: int
    """
    renormalized_weights = np.delete(weights, index)
    renormalized_weights /= renormalized_weights.sum()

    return renormalized_weights


def iid_divide(l, g):
    """
    https://github.com/TalwalkarLab/leaf/blob/master/data/utils/sample.py
    divide list `l` among `g` groups
    each group has either `int(len(l)/g)` or `int(len(l)/g)+1` elements
    returns a list of groups
    """
    num_elems = len(l)
    group_size = int(len(l) / g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l[group_size * i: group_size * (i + 1)])
    bi = group_size * num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
    return glist


def split_list_by_indices(l, indices):
    """
    divide list `l` given indices into `len(indices)` sub-lists
    sub-list `i` starts from `indices[i]` and stops at `indices[i+1]`
    returns a list of sub-lists
    """
    res = []
    current_index = 0
    for index in indices:
        res.append(l[current_index: index])
        current_index = index

    return res


def split_dataset_by_labels(
        dataset,
        n_classes,
        n_clients,
        n_clusters,
        alpha,
        frac,
        seed=1234
):
    """
    split classification dataset among `n_clients`. The dataset is split as follow:
        1) classes are grouped into `n_clusters`
        2) for each cluster `c`, samples are partitioned across clients using dirichlet distribution

    Inspired by the split in "Federated Learning with Matched Averaging"__(https://arxiv.org/abs/2002.06440)

    :param dataset:
    :type dataset: torch.utils.Dataset
    :param n_classes: number of classes present in `dataset`
    :param n_clients: number of clients
    :param n_clusters: number of clusters to consider; if it is `-1`, then `n_clusters = n_classes`
    :param alpha: parameter controlling the diversity among clients
    :param frac: fraction of dataset to use
    :param seed:
    :return: list (size `n_clients`) of subgroups, each subgroup is a list of indices.
    """
    if n_clusters == -1:
        n_clusters = n_classes

    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    all_labels = list(range(n_classes))
    rng.shuffle(all_labels)
    clusters_labels = iid_divide(all_labels, n_clusters)

    label2cluster = dict()  # maps label to its cluster
    for group_idx, labels in enumerate(clusters_labels):
        for label in labels:
            label2cluster[label] = group_idx

    # get subset
    n_samples = int(len(dataset) * frac)
    selected_indices = rng.sample(list(range(len(dataset))), n_samples)

    clusters_sizes = np.zeros(n_clusters, dtype=int)
    clusters = {k: [] for k in range(n_clusters)}
    for idx in selected_indices:
        _, label = dataset[idx]
        group_id = label2cluster[label]
        clusters_sizes[group_id] += 1
        clusters[group_id].append(idx)

    for _, cluster in clusters.items():
        rng.shuffle(cluster)

    clients_counts = np.zeros((n_clusters, n_clients), dtype=np.int64)  # number of samples by client from each cluster

    for cluster_id in range(n_clusters):
        weights = np.random.dirichlet(alpha=alpha * np.ones(n_clients))
        clients_counts[cluster_id] = np.random.multinomial(clusters_sizes[cluster_id], weights)

    clients_counts = np.cumsum(clients_counts, axis=1)

    clients_indices = [[] for _ in range(n_clients)]
    for cluster_id in range(n_clusters):
        cluster_split = split_list_by_indices(clusters[cluster_id], clients_counts[cluster_id])

        for client_id, indices in enumerate(cluster_split):
            clients_indices[client_id] += indices

    return clients_indices


def pachinko_allocation_split(
        dataset,
        n_clients,
        coarse_labels,
        n_fine_labels,
        n_coarse_labels,
        alpha,
        beta,
        frac,
        seed
):
    """
    split classification dataset among `n_clients` using pachinko allocation.
    This method works for datasets with coarse (super) labels, e.g. cifar-100.
    The dataset is split as follow:
        1) Each client  has a symmetric Dirichlet distribution with parameter `alpha` over the coarse labels.
        2) Each coarse label has a symmetric Dirichlet distribution with parameter `beta` over its fine labels.
        3) To generate a sample for the client, we first select  a coarse label by drawing from the coarse
         label multinomial distribution, and then draw a fine label using the coarse-to-fine multinomial
         distribution. We then randomly draw a sample from CIFAR-100 with that label (without replacement).
        4) If this exhausts the set of samples with this label, we remove the label from the coarse-to-fine
         multinomial and re-normalize the multinomial distribution.

    Implementation follows the description in "Adaptive Federated Optimization"__(https://arxiv.org/abs/2003.00295)

    :param dataset:
    :param coarse_labels:
    :param n_fine_labels:
    :param n_coarse_labels:
    :param n_clients:
    :param alpha:
    :param beta:
    :param frac:
    :param seed:
    :return:
    """
    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    n_samples = int(len(dataset) * frac)
    selected_indices = rng.sample(list(range(len(dataset))), n_samples)
    n_samples_by_client = n_samples // n_clients

    # map labels to fine/coarse labels
    indices_by_fine_labels = {k: list() for k in range(n_fine_labels)}
    indices_by_coarse_labels = {k: list() for k in range(n_coarse_labels)}

    for idx in selected_indices:
        _, fine_label = dataset[idx]
        coarse_label = coarse_labels[fine_label]

        indices_by_fine_labels[fine_label].append(idx)
        indices_by_coarse_labels[coarse_label].append(idx)

    available_coarse_labels = [ii for ii in range(n_coarse_labels)]

    fine_labels_by_coarse_labels = {k: list() for k in range(n_coarse_labels)}
    for fine_label, coarse_label in enumerate(coarse_labels):
        fine_labels_by_coarse_labels[coarse_label].append(fine_label)

    clients_indices = np.zeros(shape=(n_clients, n_samples_by_client), dtype=np.int64)

    for client_idx in range(n_clients):
        coarse_labels_weights =\
            np.random.dirichlet(alpha=alpha * np.ones(len(fine_labels_by_coarse_labels)))

        weights_by_coarse_labels = dict()
        for coarse_label, fine_labels in fine_labels_by_coarse_labels.items():
            weights_by_coarse_labels[coarse_label] =\
                np.random.dirichlet(alpha=beta * np.ones(len(fine_labels)))

        for ii in range(n_samples_by_client):
            coarse_label_idx =\
                int(np.argmax(np.random.multinomial(1, coarse_labels_weights)))

            coarse_label = available_coarse_labels[coarse_label_idx]

            fine_label_idx =\
                int(np.argmax(np.random.multinomial(1, weights_by_coarse_labels[coarse_label])))

            fine_label = fine_labels_by_coarse_labels[coarse_label][fine_label_idx]

            sample_idx = rng.choice(list(indices_by_fine_labels[fine_label]))
            clients_indices[client_idx, ii] = sample_idx

            indices_by_fine_labels[fine_label].remove(sample_idx)
            indices_by_coarse_labels[coarse_label].remove(sample_idx)

            if not indices_by_fine_labels[fine_label]:
                fine_labels_by_coarse_labels[coarse_label].remove(fine_label)

                weights_by_coarse_labels[coarse_label] =\
                    renormalize(
                        weights_by_coarse_labels[coarse_label],
                        fine_label_idx
                    )

                if not indices_by_coarse_labels[coarse_label]:
                    fine_labels_by_coarse_labels.pop(coarse_label, None)
                    available_coarse_labels.remove(coarse_label)

                    coarse_labels_weights =\
                        renormalize(
                            coarse_labels_weights,
                            coarse_label_idx
                        )

    return clients_indices


def pathological_non_iid_split(
        dataset,
        n_classes,
        n_clients,
        n_classes_per_client,
        frac=1,
        seed=1234
):
    """
    split classification dataset among `n_clients`. The dataset is split as follow:
        1) sort the data by label
        2) divide it into `n_clients * n_classes_per_client` shards, of equal size.
        3) assign each of the `n_clients` with `n_classes_per_client` shards

    Inspired by the split in
     "Communication-Efficient Learning of Deep Networks from Decentralized Data"__(https://arxiv.org/abs/1602.05629)

    :param dataset:
    :type dataset: torch.utils.Dataset
    :param n_classes: umber of classes present in `dataset`
    :param n_clients: number of clients
    :param n_classes_per_client:
    :param frac: fraction of dataset to use
    :param seed:
    :return: list (size `n_clients`) of subgroups, each subgroup is a list of indices.
    """
    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    # get subset
    n_samples = int(len(dataset) * frac)
    selected_indices = rng.sample(list(range(len(dataset))), n_samples)

    label2index = {k: [] for k in range(n_classes)}
    for idx in selected_indices:
        _, label = dataset[idx]
        label2index[label].append(idx)

    sorted_indices = []
    for label in label2index:
        sorted_indices += label2index[label]

    n_shards = n_clients * n_classes_per_client
    shards = iid_divide(sorted_indices, n_shards)
    random.shuffle(shards)
    tasks_shards = iid_divide(shards, n_clients)

    clients_indices = [[] for _ in range(n_clients)]
    for client_id in range(n_clients):
        for shard in tasks_shards[client_id]:
            clients_indices[client_id] += shard

    return clients_indices
