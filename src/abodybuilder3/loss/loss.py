# Copyright 2024 Exscientia
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Based on AlphaFoldLoss class from https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/loss.py#L1685

import loguru
import ml_collections
import torch

from abodybuilder3.loss.aligned_rmsd import aligned_fv_and_cdrh3_rmsd
from abodybuilder3.openfold.utils.loss import (
    compute_renamed_ground_truth,
    fape_loss,
    final_output_backbone_loss,
    find_structural_violations,
    lddt_loss,
    supervised_chi_loss,
    violation_loss_bondangle,
    violation_loss_bondlength,
    violation_loss_clash,
)


class ABB3Loss(torch.nn.Module):
    def __init__(self, config: ml_collections.config_dict.ConfigDict):
        super().__init__()
        self.config = config
        self.dist_and_angle_annealing = 0.0

    def forward(self, output: dict, batch: dict, finetune: bool = False):
        if finetune:
            output["violation"] = find_structural_violations(
                batch,
                output["positions"][-1],
                **self.config.violation,
            )

        if "renamed_atom14_gt_positions" not in output.keys():
            batch.update(
                compute_renamed_ground_truth(
                    batch,
                    output["positions"][-1],
                )
            )

        loss_fns = {
            "fape": lambda: fape_loss(
                {"sm": output},
                batch,
                self.config.fape,
            ),
            "supervised_chi": lambda: supervised_chi_loss(
                output["angles"],
                output["unnormalized_angles"],
                **{**batch, **self.config.supervised_chi},
            ),
            "final_output_backbone_loss": lambda: final_output_backbone_loss(
                output, batch
            ),
        }
        if "plddt" in output:
            loss_fns.update(
                {
                    "plddt": lambda: lddt_loss(
                        output["plddt"],
                        output["positions"][-1],
                        batch["atom14_gt_positions"],
                        batch["atom14_atom_exists"],
                        batch["resolution"],
                    ),
                }
            )

        if finetune:
            loss_fns.update(
                {
                    "violation_loss_bondlength": lambda: violation_loss_bondlength(
                        output["violation"]
                    ),
                    "violation_loss_bondangle": lambda: violation_loss_bondangle(
                        output["violation"]
                    ),
                    "violation_loss_clash": lambda: violation_loss_clash(
                        output["violation"], **batch
                    ),
                }
            )

        cum_loss = 0.0
        losses = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = self.config[loss_name].weight
            loss = loss_fn()
            if torch.isnan(loss) or torch.isinf(loss):
                loguru.warning(f"{loss_name} loss is NaN. Skipping...")
                loss = loss.new_tensor(0.0, requires_grad=True)
            if loss_name in ["violation_loss_bondlength", "violation_loss_bondangle"]:
                weight *= min(self.dist_and_angle_annealing / 50, 1)
            cum_loss = cum_loss + weight * loss
            losses[loss_name] = loss.detach().clone()
            losses[f"{loss_name}_weighted"] = weight * losses[loss_name]
            losses[f"{loss_name}_weight"] = weight

        # aligned_rmsd (not added to cum_loss)
        with torch.no_grad():
            losses.update(
                aligned_fv_and_cdrh3_rmsd(
                    coords_truth=batch["atom14_gt_positions"],
                    coords_prediction=output["positions"][-1],
                    sequence_mask=batch["seq_mask"],
                    cdrh3_mask=batch["region_numeric"] == 2,
                )
            )

        losses["loss"] = cum_loss.detach().clone()

        return cum_loss, losses
