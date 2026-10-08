"""Run the Forkcast ML service:  uvicorn deploy.app:app --host 0.0.0.0 --port 8000

Loads the artifacts written by the Lesson 15 notebook from ./artifacts. Endpoints: /health, /eta, /sentiment, /dish.
"""

from pathlib import Path

import torch

from forkcast.serve import EtaPredictor, create_app

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
torch.set_num_threads(max(1, torch.get_num_threads() // 2))     # leave CPU for the web server

eta = EtaPredictor.load(ARTIFACTS / "eta.pt")
sentiment = dish = None
if (ARTIFACTS / "sentiment.pt").exists():
    from forkcast.text import BagOfEmbeddings, Vocab
    bundle = torch.load(ARTIFACTS / "sentiment.pt", map_location="cpu", weights_only=False)
    vocab = Vocab([]); vocab.itos = bundle["itos"]; vocab.stoi = {t: i for i, t in enumerate(vocab.itos)}
    model = BagOfEmbeddings(len(vocab), dim=bundle["dim"], hidden=bundle["hidden"]); model.load_state_dict(bundle["state_dict"])
    sentiment = (model.eval(), vocab)
if (ARTIFACTS / "dish.pt").exists():
    from forkcast.vision import SmallCNN
    bundle = torch.load(ARTIFACTS / "dish.pt", map_location="cpu", weights_only=False)
    model = SmallCNN(n_classes=len(bundle["classes"]), size=bundle["size"]); model.load_state_dict(bundle["state_dict"])
    dish = (model.eval(), bundle["classes"])

app = create_app(eta, sentiment, dish)
