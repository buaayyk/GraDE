import os
import pathlib
from typing import Literal

import torch
import torch.nn.functional as F
import torch_geometric
from overrides import overrides

from grade.datasets.loader.AbstractDataset import AbstractDataset


class PROTEINSDataset(AbstractDataset):
    FILE_CANDIDATES = {
        "A": ["raw/PROTEINS_A.txt", "PROTEINS_A.txt"],
        "graph_indicator": [
            "raw/PROTEINS_graph_indicator.txt",
            "PROTEINS_graph_indicator.txt",
        ],
        "graph_labels": ["raw/PROTEINS_graph_labels.txt", "PROTEINS_graph_labels.txt"],
        "node_labels": ["raw/PROTEINS_node_labels.txt", "PROTEINS_node_labels.txt"],
        "node_attributes": [
            "raw/PROTEINS_node_attributes.txt",
            "PROTEINS_node_attributes.txt",
        ],
    }

    @classmethod
    def _resolve_existing_path(
        cls, root: str | pathlib.Path, candidates: list[str]
    ) -> pathlib.Path | None:
        root_path = pathlib.Path(root)
        for rel_path in candidates:
            path = root_path / rel_path
            if path.exists():
                return path
        return None

    @classmethod
    def _infer_raw_file_name(cls, root: str | pathlib.Path) -> str:
        path = cls._resolve_existing_path(root, cls.FILE_CANDIDATES["graph_indicator"])
        if path is None:
            return cls.FILE_CANDIDATES["graph_indicator"][0]
        return str(path.relative_to(pathlib.Path(root)))

    @classmethod
    def _infer_encoder_decoder(cls, root: str | pathlib.Path):
        node_label_path = cls._resolve_existing_path(root, cls.FILE_CANDIDATES["node_labels"])
        if node_label_path is None:
            return {"0": 0}, ["0"]

        tokens = []
        with open(node_label_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                token = raw_line.strip()
                if len(token) == 0:
                    continue
                tokens.append(token)

        if len(tokens) == 0:
            return {"0": 0}, ["0"]

        decoder = sorted(set(tokens), key=lambda x: int(x) if x.lstrip("-").isdigit() else x)
        encoder = {decoder[i]: i for i in range(len(decoder))}
        return encoder, decoder

    def __init__(
        self,
        root: str,
        split: Literal["train", "valid", "test", "estimate"],
        subgraph_sampling_method: str = None,
        subgraph_size: int = None,
        sampling_density: float = None,
    ):
        raw_file_name = self._infer_raw_file_name(root)
        encoder, decoder = self._infer_encoder_decoder(root)
        directed_flag = False
        super().__init__(
            root,
            split,
            subgraph_sampling_method,
            subgraph_size,
            sampling_density,
            raw_file_name,
            encoder,
            decoder,
            directed_flag,
            x_one_hot=True,
            tau=1,
        )
        self.ground_truth_sampling_backend = "ESU"

    def prepare(self):
        required_keys = ["A", "graph_indicator", "graph_labels"]
        missing_paths = []
        for key in required_keys:
            if self._resolve_existing_path(self.root, self.FILE_CANDIDATES[key]) is None:
                missing_paths.append(self.FILE_CANDIDATES[key][0])
        if len(missing_paths) > 0:
            missing = "\n".join([f"- {self.root / rel_path}" for rel_path in missing_paths])
            raise FileNotFoundError(
                "Missing PROTEINS raw files:\n"
                f"{missing}\n"
                "Expected TU-format files under the dataset root or its `raw/` subdirectory."
            )
        super().prepare()

    def _read_int_file(self, path: pathlib.Path) -> list[int]:
        values = []
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if len(line) == 0:
                    continue
                values.append(int(line))
        return values

    def _read_edge_file(self, path: pathlib.Path) -> list[tuple[int, int]]:
        edges = []
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if len(line) == 0:
                    continue
                parts = [part.strip() for part in line.split(",")]
                if len(parts) != 2:
                    parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Malformed edge line in {path}: {raw_line!r}")
                src, dst = int(parts[0]), int(parts[1])
                edges.append((src, dst))
        return edges

    @overrides
    def preprocess(self):
        print("[preprocess]")

        edge_path = self._resolve_existing_path(self.root, self.FILE_CANDIDATES["A"])
        graph_indicator_path = self._resolve_existing_path(
            self.root, self.FILE_CANDIDATES["graph_indicator"]
        )
        graph_labels_path = self._resolve_existing_path(
            self.root, self.FILE_CANDIDATES["graph_labels"]
        )
        node_labels_path = self._resolve_existing_path(
            self.root, self.FILE_CANDIDATES["node_labels"]
        )

        assert edge_path is not None
        assert graph_indicator_path is not None
        assert graph_labels_path is not None

        graph_indicator = self._read_int_file(graph_indicator_path)
        if len(graph_indicator) == 0:
            raise RuntimeError(f"No node-to-graph assignments were parsed from {graph_indicator_path}.")

        graph_labels = self._read_int_file(graph_labels_path)
        if len(graph_labels) == 0:
            raise RuntimeError(f"No graph labels were parsed from {graph_labels_path}.")

        if node_labels_path is not None:
            raw_node_labels = self._read_int_file(node_labels_path)
            if len(raw_node_labels) != len(graph_indicator):
                raise RuntimeError(
                    "PROTEINS node label count does not match graph indicator count: "
                    f"{len(raw_node_labels)} vs {len(graph_indicator)}."
                )
            if len(self.encoder) == 0:
                decoder = sorted(set(str(label) for label in raw_node_labels), key=lambda x: int(x))
                self.decoder = decoder
                self.encoder = {decoder[i]: i for i in range(len(decoder))}
            node_type_ids = [self.encoder[str(label)] for label in raw_node_labels]
        else:
            self.encoder = {"0": 0}
            self.decoder = ["0"]
            node_type_ids = [0 for _ in graph_indicator]

        n_graph = max(graph_indicator)
        if len(graph_labels) != n_graph:
            raise RuntimeError(
                "PROTEINS graph label count does not match graph indicator max id: "
                f"{len(graph_labels)} vs {n_graph}."
            )

        graph_to_nodes = [[] for _ in range(n_graph + 1)]
        global_to_local = {}
        for global_node_idx, graph_id in enumerate(graph_indicator, start=1):
            local_idx = len(graph_to_nodes[graph_id])
            graph_to_nodes[graph_id].append(global_node_idx)
            global_to_local[global_node_idx] = local_idx

        graph_edges = [set() for _ in range(n_graph + 1)]
        for src, dst in self._read_edge_file(edge_path):
            if src == dst:
                continue
            src_graph = graph_indicator[src - 1]
            dst_graph = graph_indicator[dst - 1]
            if src_graph != dst_graph:
                continue
            src_local = global_to_local[src]
            dst_local = global_to_local[dst]
            if src_local == dst_local:
                continue
            graph_edges[src_graph].add((min(src_local, dst_local), max(src_local, dst_local)))

        self.preprocessed_data = []
        n_node_type = len(self.encoder)
        for graph_id in range(1, n_graph + 1):
            node_ids = graph_to_nodes[graph_id]
            if len(node_ids) == 0:
                continue

            local_type_ids = [
                node_type_ids[global_node_idx - 1] for global_node_idx in node_ids
            ]
            x = F.one_hot(
                torch.tensor(local_type_ids, dtype=torch.long),
                num_classes=n_node_type,
            ).float()

            row = []
            col = []
            for src, dst in sorted(graph_edges[graph_id]):
                row.extend([src, dst])
                col.extend([dst, src])

            if len(row) > 0:
                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_attr = torch.zeros((len(row), 2), dtype=torch.float)
                edge_attr[:, 1] = 1.0
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, 2), dtype=torch.float)

            # Keep graph labels available for future use, but preserve the
            # current unconditional pipeline contract via a scalar float y.
            y = torch.tensor([[0.0]], dtype=torch.float)
            datapoint = torch_geometric.data.Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
            )
            datapoint.graph_label = int(graph_labels[graph_id - 1])
            self.preprocessed_data.append(datapoint)

        print(f"saving [preprocessed_data] to {self.preprocessed_data_path}")
        torch.save(self.preprocessed_data, self.preprocessed_data_path)
        print("saving finsh")


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    root = pathlib.Path(cur_dir) / ".." / ".." / ".." / "data" / "PROTEINS"
    train_set = PROTEINSDataset(root, "train", "RESU", 4, 0.1)
    val_set = PROTEINSDataset(root, "valid", "RESU", 4, 0.1)
    test_set = PROTEINSDataset(root, "test", "RESU", 4, 0.1)
    all_set = train_set + val_set + test_set
    print(len(train_set))
    print(len(val_set))
    print(len(test_set))
    print(len(all_set))
