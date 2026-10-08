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
        raise NotImplementedError("Lesson 15, step 3: torch.save({...}, path)")

    @classmethod
    def load(cls, path: str | Path) -> "EtaPredictor":
        """Rebuild a TabularNet(4, zone_vocab=ZONE_VOCAB, hidden=...) and a Normalizer from the artifact, on the CPU."""
        raise NotImplementedError("Lesson 15, step 3: torch.load(path, map_location=\"cpu\", weights_only=True); rebuild model and normalizer; return cls(...)")

    def features(self, orders: list[dict]) -> tuple[Tensor, Tensor]:
        """Orders as dicts with the CONTINUOUS and CATEGORICAL keys -> (normalized continuous (N, 4), zone ids (N, 2) int64).

        Build a DataFrame from the dicts so the column order matches training exactly.
        """
        raise NotImplementedError("Lesson 15, step 2: frame = pd.DataFrame(orders); continuous via self.normalizer.transform; zones as long")

    def predict(self, orders: list[dict]) -> list[float]:
        """Minutes for a batch of orders, under torch.inference_mode(), as plain Python floats."""
        raise NotImplementedError("Lesson 15, step 2: with torch.inference_mode(): self.model(*self.features(orders)).tolist()")


def export_program(model: nn.Module, example_inputs: tuple, path: str | Path) -> None:
    """Freeze the model into a torch.export program with a dynamic batch dimension and save it as .pt2.

    ``batch = torch.export.Dim("batch")``; dynamic_shapes marks dimension 0 of every example input as ``batch``;
    ``torch.export.export(model, example_inputs, dynamic_shapes=...)`` then ``torch.export.save(program, path)``.
    """
    raise NotImplementedError("Lesson 15, step 4: Dim, dynamic_shapes, torch.export.export, torch.export.save")


def load_exported(path: str | Path) -> nn.Module:
    """Load a .pt2 program and return a callable module: ``torch.export.load(path).module()``."""
    raise NotImplementedError("Lesson 15, step 4: torch.export.load(path).module()")


def latency_stats(fn, payloads: list, repeats: int = 3) -> dict:
    """Call fn(payload) for every payload ``repeats`` times; return {"p50_ms", "p95_ms", "mean_ms", "n"}.

    Use time.perf_counter around each call; statistics.quantiles(times, n=20)[9] is the median and [18] the 95th percentile.
    """
    raise NotImplementedError("Lesson 15, step 6: time each call with perf_counter, then quantiles")


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
        raise NotImplementedError("Lesson 15, step 5: eta.predict([o.model_dump() for o in request.orders])")

    @app.post("/sentiment")
    def predict_sentiment(request: TextRequest):
        """Return {"results": [{"label": "positive"|"negative", "p_positive": float}, ...]} or 503 if no model."""
        raise NotImplementedError("Lesson 15, step 5: tokenize, encode, pad_batch, softmax; 503 when sentiment is None")

    @app.post("/dish")
    def predict_dish(request: ImageRequest):
        """Return {"label": str, "confidence": float} for one (3, H, W) image, 400 if the shape is wrong, 503 if no model."""
        raise NotImplementedError("Lesson 15, step 5: torch.tensor(pixels); check 3 dims and 3 channels; softmax; class name")

    return app
