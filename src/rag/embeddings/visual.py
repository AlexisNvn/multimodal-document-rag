from pathlib import Path

import numpy as np
from PIL import Image


class ColQwen2Encoder:
    """Late-interaction page embeddings; loads weights only when explicitly configured."""

    def __init__(self, model: str, device: str = "cpu"):
        import torch
        from colpali_engine.models import ColQwen2, ColQwen2Processor

        self.torch = torch
        self.model = (
            ColQwen2.from_pretrained(
                model,
                torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
            )
            .to(device)
            .eval()
        )
        self.processor = ColQwen2Processor.from_pretrained(model)

    def _encode(self, batch):
        with self.torch.inference_mode():
            return self.model(**batch.to(self.model.device))[0].float().cpu().numpy()

    def encode_image(self, path: str | Path) -> np.ndarray:
        with Image.open(path) as image:
            return self._encode(self.processor.process_images([image.convert("RGB")]))

    def encode_query(self, query: str) -> np.ndarray:
        return self._encode(self.processor.process_queries([query]))


def maxsim(query: np.ndarray, document: np.ndarray) -> float:
    if query.ndim != 2 or document.ndim != 2 or query.shape[1] != document.shape[1]:
        raise ValueError("Expected token matrices with matching embedding dimensions")
    if len(query) == 0 or len(document) == 0:
        return 0.0
    return float((query @ document.T).max(axis=1).sum())
