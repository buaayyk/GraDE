import gzip
import os
import pathlib
from typing import Literal

import torch
import torch.nn.functional as F
import torch_geometric
from overrides import overrides

from grade.datasets.loader.AbstractDataset import AbstractDataset


class SocioPatternsHighSchoolDataset(AbstractDataset):
    METADATA_CANDIDATES = [
        "raw/HighSchool2013_metadata.txt",
        "HighSchool2013_metadata.txt",
    ]
    FRIENDSHIP_CANDIDATES = [
        "raw/HighSchool2013_friendship_network.csv.gz",
        "HighSchool2013_friendship_network.csv.gz",
        "raw/HighSchool2013_friendship_network.csv",
        "HighSchool2013_friendship_network.csv",
    ]

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
        path = cls._resolve_existing_path(root, cls.METADATA_CANDIDATES)
        if path is None:
            return cls.METADATA_CANDIDATES[0]
        return str(path.relative_to(pathlib.Path(root)))

    @classmethod
    def _infer_encoder_decoder(cls, root: str | pathlib.Path):
        metadata_path = cls._resolve_existing_path(root, cls.METADATA_CANDIDATES)
        if metadata_path is None:
            return {}, []

        token_set = set()
        with open(metadata_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if len(line) == 0:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                token_set.add(f"class={parts[1]}|gender={parts[2]}")

        decoder = sorted(token_set)
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
            x_one_hot = True,
            tau = 1
        )
        self.ground_truth_sampling_backend = "ESU"

    def prepare(self):
        metadata_path = self._resolve_existing_path(self.root, self.METADATA_CANDIDATES)
        friendship_path = self._resolve_existing_path(
            self.root, self.FRIENDSHIP_CANDIDATES
        )
        missing_paths = []
        if metadata_path is None:
            missing_paths.append(self.METADATA_CANDIDATES[0])
        if friendship_path is None:
            missing_paths.append(self.FRIENDSHIP_CANDIDATES[0])
        if len(missing_paths) > 0:
            missing = "\n".join([f"- {self.root / rel_path}" for rel_path in missing_paths])
            raise FileNotFoundError(
                "Missing SocioPatterns High School raw files:\n"
                f"{missing}\n"
                "Download them from:\n"
                "- https://www.sociopatterns.org/datasets/high-school-contact-and-friendship-networks/\n"
                "Recommended commands:\n"
                f"mkdir -p {self.root / 'raw'}\n"
                f"curl -L https://www.sociopatterns.org/assets/data/HighSchool2013_metadata.txt "
                f"-o {self.root / 'raw' / 'HighSchool2013_metadata.txt'}\n"
                f"curl -L https://www.sociopatterns.org/assets/data/HighSchool2013_friendship_network.csv.gz "
                f"-o {self.root / 'raw' / 'HighSchool2013_friendship_network.csv.gz'}"
            )
        super().prepare()

    def _metadata_path(self) -> pathlib.Path:
        path = self._resolve_existing_path(self.root, self.METADATA_CANDIDATES)
        assert path is not None
        return path

    def _friendship_path(self) -> pathlib.Path:
        path = self._resolve_existing_path(self.root, self.FRIENDSHIP_CANDIDATES)
        assert path is not None
        return path

    @staticmethod
    def _open_text(path: pathlib.Path):
        if path.suffix == ".gz":
            return gzip.open(path, "rt", encoding="utf-8")
        return open(path, "r", encoding="utf-8")

    @overrides
    def preprocess(self):
        print("[preprocess]")

        metadata_path = self._metadata_path()
        friendship_path = self._friendship_path()

        node_attr = {}
        with self._open_text(metadata_path) as f:
            for raw_line in f:
                line = raw_line.strip()
                if len(line) == 0:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                node_id = int(parts[0])
                class_id = parts[1]
                gender_id = parts[2]
                node_attr[node_id] = f"class={class_id}|gender={gender_id}"

        if len(node_attr) == 0:
            raise RuntimeError(f"No metadata rows were parsed from {metadata_path}.")

        raw_node_ids = sorted(node_attr.keys())
        raw_to_contig = {raw_node_ids[i]: i for i in range(len(raw_node_ids))}
        node_type_ids = [self.encoder[node_attr[node_id]] for node_id in raw_node_ids]

        undirected_edges = set()
        with self._open_text(friendship_path) as f:
            for raw_line in f:
                line = raw_line.strip()
                if len(line) == 0:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                src_raw = int(parts[0])
                dst_raw = int(parts[1])
                if src_raw == dst_raw:
                    continue
                if src_raw not in raw_to_contig or dst_raw not in raw_to_contig:
                    continue
                src = raw_to_contig[src_raw]
                dst = raw_to_contig[dst_raw]
                if src == dst:
                    continue
                undirected_edges.add((min(src, dst), max(src, dst)))

        row = []
        col = []
        for src, dst in sorted(undirected_edges):
            row.extend([src, dst])
            col.extend([dst, src])

        if len(row) > 0:
            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_attr = torch.zeros((len(row), 2), dtype=torch.float)
            edge_attr[:, 1] = 1.0
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 2), dtype=torch.float)

        x = F.one_hot(
            torch.tensor(node_type_ids, dtype=torch.long),
            num_classes=len(self.encoder),
        ).float()
        y = torch.tensor([[0.0]], dtype=torch.float)

        self.preprocessed_data = [
            torch_geometric.data.Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
            )
        ]

        print(f"saving [preprocessed_data] to {self.preprocessed_data_path}")
        torch.save(self.preprocessed_data, self.preprocessed_data_path)
        print("saving finsh")


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    root = pathlib.Path(cur_dir) / ".." / ".." / ".." / "data" / "SocioPatternsHighSchool"
    train_set = SocioPatternsHighSchoolDataset(root, "train", "RESU", 5, 0.1)
    val_set = SocioPatternsHighSchoolDataset(root, "valid", "RESU", 5, 0.1)
    test_set = SocioPatternsHighSchoolDataset(root, "test", "RESU", 5, 0.1)
    all_set = train_set + val_set + test_set
    print(len(train_set))
    print(len(val_set))
    print(len(test_set))
    print(len(all_set))
