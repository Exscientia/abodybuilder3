import torch
from torch.utils.data import DataLoader

from abodybuilder3.dataloader import ABDataset, collate_fn


def aligned_fv_and_cdrh3_rmsd(
    coords_truth: torch.Tensor,
    coords_prediction: torch.Tensor,
    sequence_mask: torch.Tensor,
    cdrh3_mask: torch.Tensor,
    batch_average: bool = True,
) -> dict[str, torch.Tensor]:
    """Aligns positions_truth to positions_prediction in a batched way.

    Args:
        positions_truth (torch.Tensor): (B, n, 14/37, 3) ground truth coordinates
        positions_prediction (torch.Tensor): (B, n, 14/37, 3) predicted coordinates
        sequence_mask (torch.Tensor): (B, n) where [i, j] = 1 if a coordinate for sequence i at residue j exists.
        cdrh3_mask (torch.Tensor): (B, n) where [i, j] = 1 if a coordinate for sequence i at residue j is part of the cdrh3 loop.
        batch_average (bool): if True, average along the batch dimensions

    Returns:
        A dictionary[str, torch.Tensor] containing
            seq_rmsd: the RMSD of the backbone after backbone alignment
            cdrh3_rmsd: the RMSD of the CDRH3 backbone after backbone alignment
    """

    # extract backbones and mask and put in 3d point cloud shape
    backbone_truth = extract_backbone_coordinates(coords_truth)
    backbone_prediction = extract_backbone_coordinates(coords_prediction)
    backbone_sequence_mask = extract_backbone_mask(sequence_mask)

    # align backbones
    backbone_truth = batch_align(
        backbone_truth, backbone_prediction, backbone_sequence_mask
    )

    square_distance = (
        torch.linalg.norm(backbone_prediction - backbone_truth, dim=-1) ** 2
    )
    square_distance = square_distance * backbone_sequence_mask

    seq_msd = square_distance.sum(dim=-1) / backbone_sequence_mask.sum(dim=-1)
    seq_rmsd = torch.sqrt(seq_msd)

    backbone_cdrh3_mask = extract_backbone_mask(cdrh3_mask)
    square_distance = square_distance * (backbone_cdrh3_mask * backbone_sequence_mask)
    cdrh3_msd = torch.sum(square_distance, dim=-1) / backbone_cdrh3_mask.sum(dim=-1)
    cdrh3_rmsd = torch.sqrt(cdrh3_msd)

    if batch_average:
        seq_rmsd = seq_rmsd.mean()
        cdrh3_rmsd = cdrh3_rmsd.mean()

    return {"seq_rmsd": seq_rmsd, "cdrh3_rmsd": cdrh3_rmsd}


def extract_backbone_coordinates(positions: torch.Tensor) -> torch.Tensor:
    """(B, n, 14/37, 3) -> (B, n * 4, 3)"""
    batch_size = positions.size(0)
    backbone_positions = positions[:, :, :4, :]  # (B, n, 4, 3)
    backbone_positions_flat = backbone_positions.reshape(
        batch_size, -1, 3
    )  # (B, n * 4, 3)
    return backbone_positions_flat


def extract_backbone_mask(sequence_mask: torch.Tensor) -> torch.Tensor:
    """(B, n) -> (B, n * 4)"""
    batch_size = sequence_mask.size(0)
    return sequence_mask.unsqueeze(-1).repeat(1, 1, 4).view(batch_size, -1)


def batch_align(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Aligns 3-dimensional point clouds. Based on section 4 of https://igl.ethz.ch/projects/ARAP/svd_rot.pdf.

    Args:
        x (torch.Tensor): A tensor shape (B, n, 3)
        y (torch.Tensor): A tensor shape (B, n, 3)
        mask (torch.Tensor): A mask of shape (B, n) were mask[i, j]=1 indicates the presence of a point in sample i at location j of both sequences.

    Returns:
        torch.Tensor: a rototranslated x aligned to y.
    """

    # check inputs
    if x.ndim != 3:
        raise ValueError(f"Expected x.ndim=3. Instead got {x.ndim=}")
    if y.ndim != 3:
        raise ValueError(f"Expected y.ndim=3. Instead got {y.ndim=}")
    if mask.ndim != 2:
        raise ValueError(f"Expected mask.ndim=2. Instead got {mask.ndim=}")
    if x.size(-1) != 3:
        raise ValueError(f"Expected last dim of x to be 3. Instead got {x.size(-1)=}")
    if y.size(-1) != 3:
        raise ValueError(f"Expected last dim of y to be 3. Instead got {y.size(-1)=}")

    # (B, n) -> (B, n, 1)
    mask = mask.unsqueeze(-1)

    # zero masked coordinates (the below centroids computation relies on it).
    x = x * mask
    y = y * mask

    # centroids (B, 3)
    p_bar = x.sum(dim=1) / mask.sum(dim=1)
    q_bar = y.sum(dim=1) / mask.sum(dim=1)

    # centered points (B, n, 3)
    x_centered = x - p_bar.unsqueeze(1)
    y_centered = y - q_bar.unsqueeze(1)

    # compute covariance matrices (B, 3, 3)
    num_valid_points = mask.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)
    S = torch.bmm(x_centered.transpose(-1, -2), y_centered * mask) / num_valid_points
    S = S + 10e-6 * torch.eye(S.size(-1)).unsqueeze(0).to(S.device)

    # Compute U, V from SVD (B, 3, 3)
    U, _, Vh = torch.linalg.svd(S)
    V = Vh.transpose(-1, -2)
    Uh = U.transpose(-1, -2)

    # correction that accounts for reflection (B, 3, 3)
    correction = torch.eye(x.size(-1)).unsqueeze(0).repeat(x.size(0), 1, 1).to(x.device)
    correction[:, -1, -1] = torch.det(torch.bmm(V, Uh).float())

    # rotation (B, 3, 3)
    R = V.bmm(correction).bmm(Uh)

    # translation (B, 3)
    t = q_bar - R.bmm(p_bar.unsqueeze(-1)).squeeze()

    # translate x to align with y
    x_rotated = torch.bmm(R, x.transpose(-1, -2)).transpose(-1, -2)
    x_aligned = x_rotated + t.unsqueeze(1)

    return x_aligned


if __name__ == "__main__":
    dataset = ABDataset("data", "test", edge_chain_feature=True)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, shuffle=False)
    for batch in dataloader:
        coords = batch["atom14_gt_positions"]
        preds = coords + 10
        mask = batch["seq_mask"]
        cdrh3_mask = batch["region_numeric"] == 2
        print(aligned_fv_and_cdrh3_rmsd(coords, preds, mask, cdrh3_mask))
        break
