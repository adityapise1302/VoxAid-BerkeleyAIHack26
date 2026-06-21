import json
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from peft import PeftModel
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)


def resolve_device(device_setting: str) -> torch.device:
    if device_setting != "auto":
        return torch.device(device_setting)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_processor(adapter_path: Path, base_model_name: str) -> Wav2Vec2Processor:
    """
    Your adapter folder has:
    - vocab.json
    - tokenizer_config.json
    - processor_config.json

    It may not have the standard Hugging Face preprocessor_config.json.
    This function first tries the standard loader, then falls back to manually
    constructing the processor.
    """

    try:
        return Wav2Vec2Processor.from_pretrained(str(adapter_path))
    except Exception:
        pass

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(str(adapter_path))

    processor_config_path = adapter_path / "processor_config.json"

    if processor_config_path.exists():
        with processor_config_path.open("r", encoding="utf-8") as f:
            processor_config = json.load(f)

        feature_config = processor_config.get("feature_extractor", {})
        feature_extractor = Wav2Vec2FeatureExtractor(**feature_config)
    else:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(base_model_name)

    return Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )


class LocalWav2Vec2LoRASTT:
    def __init__(
        self,
        base_model_name: str,
        adapter_path: str,
        sample_rate: int = 16000,
        device: str = "auto",
    ) -> None:
        self.base_model_name = base_model_name
        self.adapter_path = Path(adapter_path).resolve()
        self.sample_rate = sample_rate
        self.device = resolve_device(device)
        self.lock = threading.Lock()

        if not self.adapter_path.exists():
            raise FileNotFoundError(
                f"STT adapter path does not exist: {self.adapter_path}"
            )

        self.processor = load_processor(
            adapter_path=self.adapter_path,
            base_model_name=self.base_model_name,
        )

        vocab_size = len(self.processor.tokenizer)
        pad_token_id: Optional[int] = self.processor.tokenizer.pad_token_id

        base_model = Wav2Vec2ForCTC.from_pretrained(
            self.base_model_name,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            ctc_loss_reduction="mean",
            ignore_mismatched_sizes=True,
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            str(self.adapter_path),
        )

        self.model.to(self.device)
        self.model.eval()

    def transcribe(self, waveform: np.ndarray) -> str:
        if waveform is None or waveform.size == 0:
            return ""

        inputs = self.processor(
            waveform,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }

        with self.lock:
            with torch.no_grad():
                logits = self.model(**inputs).logits
                predicted_ids = torch.argmax(logits, dim=-1)

        transcript = self.processor.batch_decode(predicted_ids)[0]
        return transcript.strip()