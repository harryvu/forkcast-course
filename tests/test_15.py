"""Lesson 15 definition of done.

Run:  pytest tests/test_15.py -q
Expected: 7 passed.   (The service is exercised in-process with FastAPI's TestClient: no server, no network.)
"""

import pytest
import torch
from torch import nn

from conftest import lesson_module

pytest.importorskip("fastapi", reason="pip install fastapi httpx (present in Colab)")
serve = lesson_module("serve", "EtaPredictor", "export_program", "load_exported", "latency_stats", "create_app", lesson="Lesson 15")
trips = lesson_module("trips", "make_loaders", lesson="Lesson 6")
train = lesson_module("train", "fit_loader", lesson="Lesson 6")
text = lesson_module("text", "BagOfEmbeddings", lesson="Lesson 11")
vision = lesson_module("vision", "SmallCNN", lesson="Lesson 7")
TabularNet = lesson_module("model", "TabularNet", lesson="Lesson 6").TabularNet
Normalizer = lesson_module("preprocess", "Normalizer", lesson="Lesson 3").Normalizer

fast = pytest.mark.fast


@pytest.fixture(scope="module")
def predictor():
    frame = trips.load_trips(verbose=False)
    tr, va, _ = trips.split_frame(frame, seed=0)
    norm = Normalizer().fit(torch.tensor(tr[trips.CONTINUOUS].values, dtype=torch.float32))
    tl, vl = trips.make_loaders(tr, va, norm, batch_size=256)
    torch.manual_seed(0)
    model = TabularNet(4, hidden=(32, 32))
    train.fit_loader(model, tl, vl, epochs=3, lr=3e-3)
    return serve.EtaPredictor(model, norm, hidden=(32, 32)), tr


@pytest.fixture(scope="module")
def orders(predictor):
    _, tr = predictor
    return tr.head(8)[trips.CONTINUOUS + trips.CATEGORICAL].astype({"pickup_hour": int, "weekday": int, "passenger_count": int}).to_dict(orient="records")


@fast
def test_predict_runs_batched_in_inference_mode(predictor, orders):
    eta, _ = predictor
    minutes = eta.predict(orders)
    assert len(minutes) == 8 and all(isinstance(m, float) for m in minutes)
    singles = [eta.predict([o])[0] for o in orders]
    assert torch.allclose(torch.tensor(minutes), torch.tensor(singles), atol=1e-4), "batching must not change the answer"
    cont, cat = eta.features(orders)
    assert cont.shape == (8, 4) and cat.shape == (8, 2) and cat.dtype == torch.long
    with torch.inference_mode():
        assert not eta.model(cont, cat).requires_grad


@fast
def test_artifact_round_trip_needs_no_training_data(predictor, orders, tmp_path):
    eta, _ = predictor
    path = tmp_path / "eta.pt"
    eta.save(path)
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    assert {"state_dict", "mean", "std", "hidden"} <= set(bundle), f"the artifact must carry the normalizer statistics: {sorted(bundle)}"
    loaded = serve.EtaPredictor.load(path)
    assert torch.allclose(torch.tensor(loaded.predict(orders)), torch.tensor(eta.predict(orders)), atol=1e-5)


@fast
def test_exported_program_matches_the_model_for_any_batch_size(predictor, orders, tmp_path):
    eta, _ = predictor
    cont, cat = eta.features(orders)
    path = tmp_path / "eta.pt2"
    serve.export_program(eta.model, (cont[:4], cat[:4]), path)
    exported = serve.load_exported(path)
    with torch.inference_mode():
        for n in (1, 3, 8):
            assert torch.allclose(exported(cont[:n], cat[:n]), eta.model(cont[:n], cat[:n]), atol=1e-5), f"batch of {n} must match"


@fast
def test_eta_endpoint_and_validation(predictor, orders):
    from fastapi.testclient import TestClient
    eta, _ = predictor
    client = TestClient(serve.create_app(eta))
    assert client.get("/health").status_code == 200
    response = client.post("/eta", json={"orders": orders})
    assert response.status_code == 200
    minutes = response.json()["minutes"]
    assert len(minutes) == 8 and torch.allclose(torch.tensor(minutes), torch.tensor(eta.predict(orders)), atol=1e-5)
    bad = dict(orders[0]); del bad["trip_distance"]
    assert client.post("/eta", json={"orders": [bad]}).status_code == 422, "a missing field must be rejected before it reaches the model"
    assert client.post("/eta", json={"orders": [dict(orders[0], pu_zone=999)]}).status_code == 422
    assert client.post("/eta", json={"orders": []}).status_code == 422


@fast
def test_sentiment_endpoint(predictor):
    from fastapi.testclient import TestClient
    eta, _ = predictor
    frame = text.load_reviews()
    vocab = text.Vocab([text.tokenize(t) for t in frame["text"]])
    tl, vl = text.make_text_loaders(frame["text"][:640], frame["label"][:640], frame["text"][640:], frame["label"][640:], vocab, batch_size=32)
    torch.manual_seed(0)
    model = text.BagOfEmbeddings(len(vocab), dim=32, hidden=32)
    train.fit_v2(model, tl, vl, epochs=6, lr=3e-3, loss_fn=nn.CrossEntropyLoss())
    client = TestClient(serve.create_app(eta, sentiment=(model, vocab)))
    r = client.post("/sentiment", json={"texts": ["Delicious and the rider was friendly. Will order again.", "Cold, soggy and an hour late. Never again."]})
    assert r.status_code == 200
    results = r.json()["results"]
    assert [x["label"] for x in results] == ["positive", "negative"] and all(0.0 <= x["p_positive"] <= 1.0 for x in results)
    assert TestClient(serve.create_app(eta)).post("/sentiment", json={"texts": ["hi"]}).status_code == 503, "no model loaded: say so, do not crash"


@fast
def test_dish_endpoint_checks_the_image_shape(predictor):
    from fastapi.testclient import TestClient
    eta, _ = predictor
    tl, vl = vision.shape_loaders(640, 128)
    torch.manual_seed(0)
    model = vision.SmallCNN()
    train.fit_v2(model, tl, vl, epochs=3, lr=1e-3, loss_fn=nn.CrossEntropyLoss())
    client = TestClient(serve.create_app(eta, dish=(model, vision.SHAPE_CLASSES)))
    images, labels = vision.make_shapes(2, seed=3)
    r = client.post("/dish", json={"pixels": images[0].tolist()})
    assert r.status_code == 200 and r.json()["label"] in vision.SHAPE_CLASSES and 0.0 <= r.json()["confidence"] <= 1.0
    assert client.post("/dish", json={"pixels": images[0].permute(1, 2, 0).tolist()}).status_code == 400, "channels-last input must be refused with a clear message"
    assert client.post("/dish", json={"pixels": [[1, 2], [3]]}).status_code == 400


@fast
def test_latency_stats_and_batching(predictor, orders):
    eta, _ = predictor
    stats = serve.latency_stats(lambda o: eta.predict([o]), orders, repeats=3)
    assert set(stats) == {"p50_ms", "p95_ms", "mean_ms", "n"} and stats["n"] == 24
    assert 0 < stats["p50_ms"] <= stats["p95_ms"]
    batched = serve.latency_stats(lambda os: eta.predict(os), [orders * 8], repeats=3)      # 64 orders per call
    assert batched["mean_ms"] / 64 < stats["mean_ms"], "one batched call must cost less per order than single calls"
