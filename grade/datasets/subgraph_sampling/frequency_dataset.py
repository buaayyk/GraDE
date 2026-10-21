import json
import pathlib

import networkx
import torch
import torch.nn.functional as F

from grade.utils.format_transform import tensor_to_networkx


def build_frequency_datasets(base_train_set, seed=42):
    """Build motif-level datasets from the complete sampled subgraph file.

    Each WL group contributes its first instance and its sample count. Labels
    are log1p counts standardized using the training motifs only. The split
    seed controls a local torch generator, independently of the training seed.
    """
    source_path = pathlib.Path(base_train_set.inter_data_path)
    if not source_path.exists():
        if not base_train_set.preprocessed_data_path.exists():
            base_train_set.preprocess()
        base_train_set.subgraph_sampling()

    source_stat = source_path.stat()
    source_metadata = {
        "source_path": str(source_path.resolve()),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "seed": seed,
    }
    cache_dir = pathlib.Path(str(base_train_set.processed_data_dirpath) + "-GNNRanker")
    metadata_path = cache_dir / "metadata.json"
    splits = ("train", "valid", "test")
    if metadata_path.exists() and all((cache_dir / f"{split}.pt").exists() for split in splits):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        if all(metadata.get(key) == value for key, value in source_metadata.items()):
            datasets = [
                torch.load(cache_dir / f"{split}.pt", map_location="cpu", weights_only=False)
                for split in splits
            ]
            return *datasets, metadata

    samples = torch.load(source_path, map_location="cpu", weights_only=False)
    motifs = {}
    counts = {}
    for sample in samples:
        graph = tensor_to_networkx(sample, base_train_set.directed_flag)
        graph_hash = networkx.weisfeiler_lehman_graph_hash(
            graph, node_attr="operator", edge_attr="edge_type"
        )
        if graph_hash not in motifs:
            motif = sample.clone()
            if not base_train_set.x_one_hot:
                motif.x = F.one_hot(motif.x.long(), num_classes=len(base_train_set.encoder)).float()
            motifs[graph_hash] = motif
            counts[graph_hash] = 0
        counts[graph_hash] += 1

    motif_data = []
    for graph_hash, motif in motifs.items():
        motif.frequency = torch.tensor([counts[graph_hash]], dtype=torch.float32)
        motif.log_frequency = torch.log1p(motif.frequency)
        motif_data.append(motif)

    if len(motif_data) < 3:
        raise ValueError("GNNRanker requires at least three sampled motif types for train/valid/test splits.")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(motif_data), generator=generator).tolist()
    valid_size = max(1, int(len(motif_data) * 0.05))
    test_size = max(1, int(len(motif_data) * 0.05))
    train_size = len(motif_data) - valid_size - test_size
    train_set = [motif_data[idx] for idx in indices[:train_size]]
    val_set = [motif_data[idx] for idx in indices[train_size:train_size + valid_size]]
    test_set = [motif_data[idx] for idx in indices[train_size + valid_size:]]

    train_log_frequency = torch.cat([motif.log_frequency for motif in train_set])
    target_mean = train_log_frequency.mean()
    target_std = train_log_frequency.std(unbiased=False).clamp_min(1e-6)
    for motif in motif_data:
        motif.frequency_target = (motif.log_frequency - target_mean) / target_std

    datasets = (train_set, val_set, test_set)
    metadata = {
        **source_metadata,
        "num_samples": len(samples),
        "num_unique_motifs": len(motif_data),
        "split_sizes": {split: len(dataset) for split, dataset in zip(splits, datasets)},
        "target": "standardized log1p empirical sample count",
        "target_mean": target_mean.item(),
        "target_std": target_std.item(),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    for split, dataset in zip(splits, datasets):
        torch.save(dataset, cache_dir / f"{split}.pt")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return *datasets, metadata
