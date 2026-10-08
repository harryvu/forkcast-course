"""Lesson 15. Serving: a trained model becomes a service.

``Order``, ``latency_stats`` helpers' signatures and ``create_app``'s routes are fixed by the tests; the
preprocessing, the artifact, the export and the endpoint bodies are the exercises. ``deploy/app.py`` runs the
service with uvicorn, ``deploy/Dockerfile`` packages it.
"""

import statistics
import time
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor, nn

from forkcast.model import TabularNet
from forkcast.preprocess import Normalizer
from forkcast.trips import CATEGORICAL, CONTINUOUS, ZONE_VOCAB

try:
    from pydantic import BaseModel, Field
except ImportError:                                   # the web extras are optional for the rest of the course
    BaseModel = object

    def Field(*args, **kwargs):
        return None


class Order(BaseModel):
    """One delivery as the app sends it. pydantic rejects anything outside these ranges before the model sees it."""
    trip_distance: float = Field(gt=0, le=50)
    pickup_hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    passenger_count: int = Field(ge=1, le=6)
    pu_zone: int = Field(ge=0, lt=ZONE_VOCAB)
    do_zone: int = Field(ge=0, lt=ZONE_VOCAB)


class EtaRequest(BaseModel):
    orders: list[Order] = Field(min_length=1, max_length=256)


class TextRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=256)


class ImageRequest(BaseModel):
    pixels: list        # (3, H, W) nested lists of floats in [0, 1]


class EtaPredictor:
    """Everything the ETA endpoint needs, in one object: the network and the normalizer statistics it was trained with.

    An artifact file holds ``{"state_dict", "mean", "std", "hidden"}``; loading rebuilds the model with no training code.
    """

    def __init__(self, model: TabularNet, normalizer: Normalizer, hidden=(64, 64)):
        self.model, self.normalizer, self.hidden = model.eval(), normalizer, tuple(hidden)

    def save(self, path: str | Path) -> None:
        """torch.save a dict with the model's state_dict, the normalizer's mean and std, and the hidden sizes."""
        torch.save({"state_dict": self.model.state_dict(), "mean": self.normalizer.mean, "std": self.normalizer.std, "hidden": list(self.hidden)}, path)

    @classmethod
    def load(cls, path: str | Path) -> "EtaPredictor":
        """Rebuild a TabularNet(4, zone_vocab=ZONE_VOCAB, hidden=...) and a Normalizer from the artifact, on the CPU."""
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        model = TabularNet(4, zone_vocab=ZONE_VOCAB, hidden=tuple(bundle["hidden"]))
        model.load_state_dict(bundle["state_dict"])
        normalizer = Normalizer()
        normalizer.mean, normalizer.std = bundle["mean"], bundle["std"]
        return cls(model, normalizer, tuple(bundle["hidden"]))

    def features(self, orders: list[dict]) -> tuple[Tensor, Tensor]:
        """Orders as dicts with the CONTINUOUS and CATEGORICAL keys -> (normalized continuous (N, 4), zone ids (N, 2) int64).

        Build a DataFrame from the dicts so the column order matches training exactly.
        """
        frame = pd.DataFrame(orders)
        cont = torch.tensor(frame[CONTINUOUS].values, dtype=torch.float32)        # the training column order, always
        cat = torch.tensor(frame[CATEGORICAL].values, dtype=torch.long)
        return self.normalizer.transform(cont), cat

    def predict(self, orders: list[dict]) -> list[float]:
        """Minutes for a batch of orders, under torch.inference_mode(), as plain Python floats."""
        with torch.inference_mode():                                            # no autograd bookkeeping at all
            return self.model(*self.features(orders)).tolist()


def export_program(model: nn.Module, example_inputs: tuple, path: str | Path) -> None:
    """Freeze the model into a torch.export program with a dynamic batch dimension and save it as .pt2.

    ``batch = torch.export.Dim("batch")``; dynamic_shapes marks dimension 0 of every example input as ``batch``;
    ``torch.export.export(model, example_inputs, dynamic_shapes=...)`` then ``torch.export.save(program, path)``.
    """
    batch = torch.export.Dim("batch")
    dynamic = tuple({0: batch} for _ in example_inputs)                          # any batch size, fixed everything else
    program = torch.export.export(model.eval(), example_inputs, dynamic_shapes=dynamic)
    torch.export.save(program, str(path))


def load_exported(path: str | Path) -> nn.Module:
    """Load a .pt2 program and return a callable module: ``torch.export.load(path).module()``."""
    return torch.export.load(str(path)).module()


def latency_stats(fn, payloads: list, repeats: int = 3) -> dict:
    """Call fn(payload) for every payload ``repeats`` times; return {"p50_ms", "p95_ms", "mean_ms", "n"}.

    Use time.perf_counter around each call; statistics.quantiles(times, n=20)[9] is the median and [18] the 95th percentile.
    """
    times = []
    for _ in range(repeats):
        for payload in payloads:
            start = time.perf_counter()
            fn(payload)
            times.append((time.perf_counter() - start) * 1000.0)
    q = statistics.quantiles(times, n=20) if len(times) >= 2 else [times[0]] * 19
    return {"p50_ms": q[9], "p95_ms": q[18], "mean_ms": statistics.mean(times), "n": len(times)}


def create_app(eta: EtaPredictor, sentiment=None, dish=None):
    """The Forkcast ML service: GET /health, POST /eta, POST /sentiment, POST /dish.

    ``sentiment`` is (model, vocab) from Lesson 11 or None; ``dish`` is (model, class_names) from Lesson 7 or None.
    FastAPI validates request bodies against the pydantic models above and answers 422 when they do not match.
    """
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="Forkcast ML service")

    @app.get("/health")
    def health():
        return {"status": "ok", "torch": torch.__version__}

    @app.post("/eta")
    def predict_eta(request: EtaRequest):
        """Return {"minutes": [...]} with one float per order, in order."""
        return {"minutes": eta.predict([o.model_dump() for o in request.orders])}

    @app.post("/sentiment")
    def predict_sentiment(request: TextRequest):
        """Return {"results": [{"label": "positive"|"negative", "p_positive": float}, ...]} or 503 if no model."""
        if sentiment is None:
            raise HTTPException(status_code=503, detail="sentiment model not loaded")
        from forkcast.text import pad_batch, tokenize
        model, vocab = sentiment
        ids, mask = pad_batch([torch.tensor(vocab.encode(tokenize(t))[:200] or [1], dtype=torch.long) for t in request.texts])
        with torch.inference_mode():
            probs = torch.softmax(model.eval()(ids, mask), dim=1)[:, 1]
        return {"results": [{"label": "positive" if p >= 0.5 else "negative", "p_positive": float(p)} for p in probs]}

    @app.post("/dish")
    def predict_dish(request: ImageRequest):
        """Return {"label": str, "confidence": float} for one (3, H, W) image, 400 if the shape is wrong, 503 if no model."""
        if dish is None:
            raise HTTPException(status_code=503, detail="dish model not loaded")
        model, class_names = dish
        try:
            x = torch.tensor(request.pixels, dtype=torch.float32)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="pixels must be a rectangular (3, H, W) array of numbers")
        if x.dim() != 3 or x.shape[0] != 3:
            raise HTTPException(status_code=400, detail=f"expected shape (3, H, W), got {tuple(x.shape)}")
        with torch.inference_mode():
            probs = torch.softmax(model.eval()(x.unsqueeze(0)), dim=1)[0]
        conf, idx = probs.max(dim=0)
        return {"label": class_names[int(idx)], "confidence": float(conf)}

    return app
