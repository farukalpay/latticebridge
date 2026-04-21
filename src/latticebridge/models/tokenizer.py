from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<src>", "<tgt>"]


@dataclass
class LatticeTokenizer:
    tokenizer: Tokenizer

    @classmethod
    def train(cls, corpus_files: list[str], save_path: Path, vocab_size: int = 4096, min_frequency: int = 2) -> "LatticeTokenizer":
        tokenizer = Tokenizer(BPE(unk_token="<pad>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        tokenizer.post_processor = ByteLevelProcessor(trim_offsets=True)
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
        )
        tokenizer.train(corpus_files, trainer)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(save_path))
        return cls(tokenizer)

    @classmethod
    def load(cls, path: Path) -> "LatticeTokenizer":
        return cls(Tokenizer.from_file(str(path)))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

    def token_surface(self, token_id: int) -> str:
        token = self.tokenizer.id_to_token(token_id)
        if token in SPECIAL_TOKENS or token is None:
            return ""
        return self.decode([token_id])

    def id_for(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise KeyError(token)
        return token_id

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
