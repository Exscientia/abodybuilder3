from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from lightning import Trainer
from lightning.pytorch import LightningModule, Trainer
from torch.utils.data import DataLoader
from transformers import (
    BertModel,
    BertTokenizer,
    PreTrainedTokenizer,
    T5EncoderModel,
    T5Tokenizer,
)
from transformers.tokenization_utils_base import BatchEncoding


@dataclass
class ProtTrans:
    weights_dir: Path
    model_type: str
    paired: bool
    batch_size: int = 1
    device_map: str = 'auto'
    tokeniser: PreTrainedTokenizer = field(init=False)
    seperator_token: Optional[str] = field(init=False)
    trainer: Optional[Trainer] = field(init=False)

    def __post_init__(self):
        if self.model_type not in ["bert", "t5"]:
            raise ValueError(f"Model should be 'bert' or 't5'. Got {self.model_type=}.")

        self.embedding_module = ProtTransEmbedder(
            weights_dir=self.weights_dir, model_type=self.model_type, device_map=self.device_map
        )

        if self.model_type == "bert":
            Tokeniser = BertTokenizer
        elif self.model_type == "t5":
            Tokeniser = T5Tokenizer
        self.tokeniser = Tokeniser.from_pretrained(
            self.weights_dir,
            do_lower_case=False,
        )

        if self.paired and self.model_type == "bert":
            self.seperator_token = "[SEP]"
        elif self.paired and self.model_type == "t5":
            self.seperator_token = "</s>"

        self.trainer = Trainer(num_nodes=1, devices=1)

    def collate_fn(self, batch: list[str]) -> BatchEncoding:
        sequences = [list(seq) for seq in batch]
        if self.paired:
            sequences = [
                [char if char != "/" else self.seperator_token for char in sequence]
                for sequence in sequences
            ]
        batch = self.tokeniser(
            sequences,
            padding="longest",
            return_special_tokens_mask=True,
            is_split_into_words=True,
            return_tensors="pt",
        )
        return batch

    def get_embeddings(
        self, heavy_sequences: list[str], light_sequences: list[str]
    ) -> npt.NDArray:
        if self.paired:
            sequences = [x + "/" + y for x, y in zip(heavy_sequences, light_sequences)]
            embeddings = self._embed_sequences(sequences)
        else:
            embeddings_heavy = self._embed_sequences(heavy_sequences)
            embeddings_light = self._embed_sequences(light_sequences)
            embeddings = [
                torch.cat((x, y), dim=1)
                for x, y in zip(embeddings_heavy, embeddings_light)
            ]
        embeddings = [x.squeeze(0) for x in embeddings]
        return embeddings

    def _embed_sequences(self, sequences: list[str]) -> list[torch.Tensor]:
        loader = DataLoader(
            sequences, collate_fn=self.collate_fn, batch_size=self.batch_size
        )
        embeddings = self.trainer.predict(self.embedding_module, loader)
        return embeddings


class ProtTransEmbedder(LightningModule):
    def __init__(
        self,
        weights_dir: Path,
        model_type: str,
        device_map: str,
    ) -> None:
        super().__init__()
        if model_type not in ["bert", "t5"]:
            raise ValueError(f"Model should be 'bert' or 't5'. Got {model_type=}.")
        self.weights_dir = weights_dir
        self.model_type = model_type

        if model_type == "bert":
            self.model = BertModel.from_pretrained(
                self.weights_dir, add_pooling_layer=False, device_map=device_map
            )
        elif model_type == "t5":
            self.model = T5EncoderModel.from_pretrained(self.weights_dir, device_map=device_map)

        if self.model_type == "bert":
            self.seperator_token_id = 3
        elif self.model_type == "t5":
            self.seperator_token_id = 1

    def predict_step(self, batch: BatchEncoding, batch_idx: int) -> torch.Tensor:
        # embed all tokens
        if batch["input_ids"].size(0) != 1:
            raise ValueError("Not implemented for batchsize > 1.")
        embedding_matrix = self.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )[
            "last_hidden_state"
        ]  # (B, n, d)
        special_token_mask = (
            batch["special_tokens_mask"].bool() | batch["input_ids"]
            == self.seperator_token_id
        )
        masked_embedding_matrix = embedding_matrix[~special_token_mask].unsqueeze(
            0
        )  # (B, n, d)
        return masked_embedding_matrix


@dataclass
class ProtT5(ProtTrans):
    batch_size: int = 1
    paired: bool = False
    weights_dir: Path = Path("Rostlab/prot_t5_xl_uniref50")
    model_type: str = "t5"


@dataclass
class PairedIgT5(ProtTrans):
    batch_size: int = 1
    paired: bool = True
    weights_dir: Path = Path("Exscientia/IgT5")
    model_type: str = "t5"


@dataclass
class ProtBert(ProtTrans):
    batch_size: int = 1
    paired: bool = False
    weights_dir: Path = Path("Rostlab/prot_bert_bfd")
    model_type: str = "bert"
