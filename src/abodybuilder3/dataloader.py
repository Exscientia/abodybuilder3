from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ABDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str,
        limit: Optional[int] = None,
        legacy: bool = False,
        rel_pos_dim: int = 16,
        edge_chain_feature: bool = False,
        use_plm_embeddings: bool = False,
    ) -> None:
        """Dataset of antibodies suitable for openfold structuremodules and loss functions.

        Args:
            path (str): root data folder
            split (str): "train", "valid" or "test"
        """
        super().__init__()
        self.path = path
        self.split = split
        self.limit = limit
        self.legacy = legacy
        self.rel_pos_dim = rel_pos_dim
        self.edge_chain_feature = edge_chain_feature
        self.use_plm_embeddings = use_plm_embeddings

        self.df = pd.read_csv(Path(self.path) / "split.csv", index_col=0)
        self.df = self.df.query(f"split=='{split}'")
        if self.legacy:
            self.df = self.df[self.df["in_legacy"]]

        if limit is not None:
            self.df = self.df.iloc[:limit]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        file_id = self.df.iloc[idx].name
        if self.use_plm_embeddings:
            fname = Path(self.path) / "structures" / "structures_plm" / f"{file_id}.pt"
        else:
            fname = Path(self.path) / "structures" / "structures" / f"{file_id}.pt"
        datapoint = torch.load(fname)
        datapoint.update(
            self.single_and_double_from_datapoint(
                datapoint,
                self.rel_pos_dim,
                self.edge_chain_feature,
                self.use_plm_embeddings,
            )
        )
        return datapoint

    @staticmethod
    def single_and_double_from_datapoint(
        datapoint: dict,
        rel_pos_dim: int,
        edge_chain_feature: bool = False,
        use_plm_embeddings: bool = False,
    ):
        """
        datapoint is a dict containing:
            aatype - [n,] tensor of ints for the amino acid (including unknown)
            is_heavy - [n,] tensor of ints where 1 is heavy chain and 0 is light chain.
            residue_index - [n,] tensor of ints assinging integer to each residue

        rel_pos_dim: integer determining edge feature dimension

        edge_chain_feature: boolean to add an edge feature z_ij that encodes what chain i and j are in.

        returns:
            A dictionary containing single a tensor of size (n, 23) and pair a tensor of size (n, n, 2 * rel_pos_dim + 1 + x) where x is 3 if edge_chain_feature and 0 otherwise.
        """
        if use_plm_embeddings:
            single = datapoint["plm_embedding"]
        else:
            single_aa = torch.nn.functional.one_hot(datapoint["aatype"], 21)
            single_chain = torch.nn.functional.one_hot(datapoint["is_heavy"].long(), 2)
            single = torch.cat((single_aa, single_chain), dim=-1)
        pair = datapoint["residue_index"]
        pair = pair[None] - pair[:, None]
        pair = pair.clamp(-rel_pos_dim, rel_pos_dim) + rel_pos_dim
        pair = torch.nn.functional.one_hot(pair, 2 * rel_pos_dim + 1)
        if edge_chain_feature:
            is_heavy = datapoint["is_heavy"]
            is_heavy = 2 * is_heavy.outer(is_heavy) + (
                (1 - is_heavy).outer(1 - is_heavy)
            )
            is_heavy = torch.nn.functional.one_hot(is_heavy.long())
            pair = torch.cat((is_heavy, pair), dim=-1)
        return {"single": single.float(), "pair": pair.float()}


def pad_square_tensors(tensors: list[torch.tensor]) -> torch.tensor:
    """Pads a list of tensors in the first two dimensions.

    Args:
        tensors (list[torch.tensor]): Input tensor are of shape (n_1, n_1, ...), (n_2, n_2, ...). where shape matches in the ... dimensions

    Returns:
        torch.tensor: A tensor of size (len(tensor), max(n_1,...), max(n_1,...), ...)
    """
    max_len = max(map(len, tensors))
    output = torch.zeros((len(tensors), max_len, max_len, *tensors[0].shape[2:]))
    for i, tensor in enumerate(tensors):
        output[i, : tensor.size(0), : tensor.size(1)] = tensor
    return output


pad_first_dim_keys = [
    "atom14_gt_positions",
    "atom14_alt_gt_positions",
    "atom14_atom_is_ambiguous",
    "atom14_gt_exists",
    "atom14_alt_gt_exists",
    "atom14_atom_exists",
    "single",
    "seq_mask",
    "aatype",
    "backbone_rigid_tensor",
    "backbone_rigid_mask",
    "rigidgroups_gt_frames",
    "rigidgroups_alt_gt_frames",
    "rigidgroups_gt_exists",
    "cdr_mask",
    "chi_mask",
    "chi_angles_sin_cos",
    "residue_index",
    "residx_atom14_to_atom37",
    "region_numeric",
]

pad_first_two_dim_keys = ["pair"]


def collate_fn(batch: dict):
    """A collate function so the ABDataset can be used in a torch dataloader.

    Args:
        batch (dict): A list of datapoints from ABDataset

    Returns:
        dict: A dictionary where the keys are the same as batch but map to a batched tensor where the batch is on the leading dimension.
    """
    batch = {key: [d[key] for d in batch] for key in batch[0]}
    output = {}
    for key in batch:
        if key in pad_first_dim_keys:
            output[key] = pad_sequence(batch[key], batch_first=True)
        elif key in pad_first_two_dim_keys:
            output[key] = pad_square_tensors(batch[key])
        elif key == "resolution":
            output[key] = torch.Tensor(batch["resolution"])
    return output


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = ABDataset("data", "test", edge_chain_feature=True)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    for batch in dataloader:
        break
    for key in batch:
        print(key, batch[key].shape)
