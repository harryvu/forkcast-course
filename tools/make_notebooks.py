"""Generate the thin Colab notebooks from the lesson definitions below.

The notebooks are deliberately small: clone, install, explore, write the
lesson's file, run the definition-of-done tests, push. The code cells mirror
the lesson pages, and each %%writefile cell is generated from that lesson's
starter in starters/lessonNN/forkcast/, so the two can never drift apart.

Usage:  python tools/make_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")}


def notebook(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


SETUP = '''
GITHUB_USER = "your-github-username"   # edit this line
REPO = "forkcast-course"

!git clone https://github.com/{GITHUB_USER}/{REPO}.git
%cd {REPO}
!pip install -e .
'''

PUSH = '''
from getpass import getpass
TOKEN = getpass("GitHub token with access to your repo: ")

!git config user.name "{GITHUB_USER}"
!git config user.email "{GITHUB_USER}@users.noreply.github.com"
!git add -A
!git commit -m "{COMMIT_MESSAGE}"
!git push https://{TOKEN}@github.com/{GITHUB_USER}/{REPO}.git HEAD:main
'''


def lesson00() -> dict:
    starter = (ROOT / "starters/lesson00/forkcast/tensors.py").read_text(encoding="utf-8")
    cells = [
        md("""
# Lesson 0: Setup and tensors

This notebook goes with the Lesson 0 page of the Forkcast course. Read the page first; the cells here are the same code, in order, ready to run.

Runtime: CPU is enough for this lesson. To see a GPU show up in the environment check, choose **Runtime → Change runtime type** and pick a GPU.
"""),
        md("## 1. Clone your repository and install the package\n\nEdit `GITHUB_USER` first. The clone fails with a clear message if the name is wrong."),
        code(SETUP),
        md("## 2. Environment check\n\nThis is the first definition-of-done command."),
        code("!python -m forkcast.env"),
        md("## 3. Meet tensors\n\nFor every cell, say the shape out loud *before* you run it."),
        code('''
import torch
torch.manual_seed(0)                  # same random numbers every run

orders = torch.randn(32, 4)           # 32 orders, 4 features each
orders.shape, orders.dtype, orders.device
'''),
        code('''
print(orders[0].shape)        # the first order
print(orders[:, 1].shape)     # the second feature of every order
print(orders[:3, :2].shape)   # first three orders, first two features
'''),
        code('''
print(torch.tensor([1, 2, 3]).dtype)          # whole numbers give int64
print(torch.tensor([1.0, 2.0, 3.0]).dtype)    # the dtype models use
print(torch.zeros(2, 3).dtype)                # zeros/ones/randn default to float32
'''),
        md("### Rearranging without changing the numbers"),
        code('''
one = orders[0]
print(one.unsqueeze(0).shape)            # a batch of one order

images = torch.zeros(8, 1, 28, 28)       # 8 grayscale images
print(images.flatten(1).shape)           # keep dim 0, merge the rest
print(images.permute(0, 2, 3, 1).shape)  # channels last
print(images.reshape(8, -1).shape)       # -1 means "work this one out"

x = torch.arange(6).reshape(2, 3)
print(x)
print(x.view(3, 2))        # same memory order, regrouped
print(x.permute(1, 0))     # rows become columns: a different reading order
'''),
        md("### Broadcasting\n\nThe last cell reproduces the classic mistake on purpose, inside a `try` so the notebook keeps running."),
        code('''
bias = torch.tensor([0.0, 1.0, 2.0, 3.0])       # one bias per feature
print((orders + bias).shape)                     # stretched over 32 orders

col_means = orders.mean(dim=0)                   # one mean per feature
print(col_means.shape, (orders - col_means).shape)

row_means = orders.mean(dim=1)                   # one mean per order, shape (32,)
try:
    orders - row_means
except RuntimeError as err:
    print("RuntimeError:", err)

row_means = orders.mean(dim=1, keepdim=True)     # shape (32, 1) broadcasts
print((orders - row_means).shape)
'''),
        md("### Devices"),
        code('''
device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_orders = orders.to(device)       # .to() returns a new tensor; orders itself stays on the CPU
print(orders.device, gpu_orders.device)
'''),
        md("""
## 4. Write the six functions

The cell below writes `forkcast/tensors.py` when you run it. Replace every `raise NotImplementedError(...)` with your code, run the cell to save the file, then run the tests in the next section. (You can also edit the file directly in Colab's file browser on the left, under `forkcast-course/forkcast/`.)

Hints are on the lesson page under step 6.
"""),
        code("%%writefile forkcast/tensors.py\n" + starter),
        md("Quick manual check before the tests. Shapes first:"),
        code('''
import importlib
from forkcast import tensors
importlib.reload(tensors)   # pick up the file you just wrote

print(tensors.to_batch(torch.randn(4)).shape)                       # (1, 4)
print(tensors.flatten_images(torch.zeros(8, 3, 28, 28)).shape)      # (8, 2352)
print(tensors.channels_last(torch.zeros(8, 3, 28, 28)).shape)       # (8, 28, 28, 3)
print(tensors.normalize_columns(torch.randn(32, 4)).mean(dim=0))    # four numbers close to 0
print(tensors.pairwise_distance(torch.randn(5, 3), torch.randn(7, 3)).shape)   # (5, 7)
print(tensors.move_to(torch.randn(3), device).device)
'''),
        md("## 5. Definition of done\n\nExpected: `7 passed, 1 skipped, 1 deselected` on a CPU runtime, `8 passed, 1 deselected` on a GPU runtime (the deselected test is the stretch goal). A failing test prints the shape it got and the shape it expected."),
        code("!pytest tests/test_00.py -q"),
        md("### Stretch: `one_hot`\n\nOptional. Build it from `torch.arange`, a comparison and broadcasting, then:"),
        code("!pytest tests/test_00.py -m stretch -q"),
        md("""
## 6. Commit and push

Create a fine-grained personal access token on GitHub (Settings → Developer settings → Personal access tokens) for your `forkcast-course` repository with **Contents: read and write**. The cell asks for it with `getpass`, so it never appears in the notebook output. After the push, open the **Actions** tab of your repository: the *DoD checks* workflow runs the same fast tests and turns green.
"""),
        code(PUSH.replace("{COMMIT_MESSAGE}", "Lesson 0: tensors")),
    ]
    return notebook(cells)


def lesson01() -> dict:
    starter = (ROOT / "starters/lesson01/forkcast/linear.py").read_text(encoding="utf-8")
    cells = [
        md("""
# Lesson 1: Linear regression by hand

Goes with the Lesson 1 page of the Forkcast course. Predict delivery minutes from distance, measure how wrong the prediction is, and improve it with gradient descent that you write yourself.

Runtime: CPU is enough.
"""),
        md("## 1. Clone your repository and install the package"),
        code(SETUP),
        md("## 2. Generate deliveries and look at them\n\nThe data is synthetic, made from `minutes = 8 + 3.5 * km + noise`, so later you can compare what the model learned with the truth."),
        code('''
from forkcast.data import make_deliveries, TRUE_SLOPE, TRUE_INTERCEPT
x, y = make_deliveries(200, seed=0)
print(x.shape, y.shape)

import matplotlib.pyplot as plt
plt.scatter(x, y, s=10); plt.xlabel("km"); plt.ylabel("minutes");
'''),
        md("""
## 3. Write the five functions

Run the cell below to write `forkcast/linear.py`. Replace each `raise NotImplementedError(...)`, in the order the lesson page builds them: `predict` and `mse_loss` first, then `gradients`, then `gradient_step` and `fit`. Re-run the cell every time you change it.
"""),
        code("%%writefile forkcast/linear.py\n" + starter),
        md("### Try a guess by hand\n\nWith `w = 1, b = 2` the line is far too shallow. The loss should be about 389, and both slopes should be negative, meaning both knobs need to grow."),
        code('''
import importlib, torch
from forkcast import linear
importlib.reload(linear)

w, b = torch.tensor(1.0), torch.tensor(2.0)
pred = linear.predict(x, w, b)
print(pred.shape)                                   # torch.Size([200]) -- not (200, 200)!
print(linear.mse_loss(pred, y).item())              # about 389
print(linear.gradients(x, y, w, b))                 # about (-221.6, -36.7)
'''),
        md("## 4. Fit, and compare with the truth"),
        code('''
importlib.reload(linear)
w, b, losses = linear.fit(x, y, lr=0.01, steps=3000)
print(f"learned  w={w.item():.3f}  b={b.item():.3f}")
print(f"truth    w={TRUE_SLOPE}  b={TRUE_INTERCEPT}")
print(f"loss went from {losses[0]:.1f} to {losses[-1]:.2f}")

plt.plot(losses); plt.yscale("log"); plt.xlabel("step"); plt.ylabel("loss (log scale)");
'''),
        md("### Break it on purpose\n\nTen times the learning rate. Watch the loss grow every step until it overflows to `nan`. When a model's loss turns into `nan`, this is the first thing to suspect."),
        code('''
_, _, bad = linear.fit(x, y, lr=0.1, steps=60)
print([round(v, 1) for v in bad[:5]])
print(bad[-1])
'''),
        md("## 5. Definition of done\n\nExpected: `7 passed`, then `14 passed` with no failures for everything so far (skipped tests belong to lessons you have not started)."),
        code("!pytest tests/test_01.py -q"),
        code("!pytest -m fast -q"),
        md("## 6. Commit and push"),
        code(PUSH.replace("{COMMIT_MESSAGE}", "Lesson 1: linear regression by hand")),
    ]
    return notebook(cells)


def lesson02() -> dict:
    starter = (ROOT / "starters/lesson02/forkcast/regression.py").read_text(encoding="utf-8")
    cells = [
        md("""
# Lesson 2: Autograd and many features

Goes with the Lesson 2 page of the Forkcast course. PyTorch computes the slopes; the model grows from one feature to four.

Runtime: CPU is enough.
"""),
        md("## 1. Clone your repository and install the package"),
        code(SETUP),
        md("## 2. Autograd on Lesson 1's data\n\nSame guess as last lesson, `w = 1, b = 2`, so the numbers are comparable."),
        code('''
import torch
from forkcast.data import make_deliveries
from forkcast import linear
x, y = make_deliveries(200, seed=0)

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)
print("before backward:", w.grad)
loss = ((w * x + b - y) ** 2).mean()
print(loss)
loss.backward()
print("autograd:", w.grad, b.grad)
print("lesson 1:", linear.gradients(x, y, w.detach(), b.detach()))
'''),
        md("### Gradients add up\n\nCall `backward()` three times without zeroing. The data and the loss never change; only `.grad` does."),
        code('''
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)
for i in range(3):
    ((w * x + b - y) ** 2).mean().backward()
    print(i + 1, w.grad)
w.grad.zero_()
print("after zero_():", w.grad)
'''),
        md("""
## 3. Write the three functions

Run the cell to write `forkcast/regression.py`. Build them in order: `autograd_gradients` (step 3 on the page), `predict_many` (step 5), `fit_autograd` (step 6). Re-run after every change.
"""),
        code("%%writefile forkcast/regression.py\n" + starter),
        md("## 4. Four features"),
        code('''
import importlib
from forkcast import regression
importlib.reload(regression)
from forkcast.data import make_orders, FEATURES, TRUE_WEIGHTS, TRUE_BIAS

X, y = make_orders(400, seed=0)
print(X.shape, y.shape)
print(FEATURES)
print(X[:3])
print(y[:3])

w4 = torch.tensor([3.5, 1.5, 4.0, 6.0]); b4 = torch.tensor(8.0)
pred = regression.predict_many(X, w4, b4)
print(pred.shape)
print(torch.allclose(pred, (X * w4).sum(dim=1) + b4))     # @ is the loop you would have written
'''),
        md("## 5. Fit, and read what it learned"),
        code('''
importlib.reload(regression)
w, b, losses = regression.fit_autograd(X, y, lr=0.005, steps=4000)
for name, learned, true in zip(FEATURES, w.tolist(), TRUE_WEIGHTS.tolist()):
    print(f"{name:12s} learned {learned:5.2f}   truth {true}")
print(f"bias         learned {b.item():5.2f}   truth {TRUE_BIAS}")
print(f"loss went from {losses[0]:.1f} to {losses[-1]:.2f}")

import matplotlib.pyplot as plt
plt.plot(losses); plt.yscale("log"); plt.xlabel("step"); plt.ylabel("loss (log scale)");
'''),
        md("### Break it on purpose\n\nThe same loop with the two `zero_()` lines deleted. Nothing crashes; the loss just stops making sense."),
        code('''
w_ = torch.zeros(4, requires_grad=True); b_ = torch.zeros((), requires_grad=True)
bad = []
for step in range(30):
    loss = ((X @ w_ + b_ - y) ** 2).mean()
    bad.append(round(loss.item(), 1))
    loss.backward()
    with torch.no_grad():
        w_ -= 0.005 * w_.grad
        b_ -= 0.005 * b_.grad
    # no zero_() here, on purpose
print(bad)
'''),
        md("## 6. Definition of done\n\nExpected: `6 passed`, then `20 passed` with no failures for everything so far (skipped tests belong to lessons you have not started)."),
        code("!pytest tests/test_02.py -q"),
        code("!pytest -m fast -q"),
        md("## 7. Commit and push"),
        code(PUSH.replace("{COMMIT_MESSAGE}", "Lesson 2: autograd and many features")),
    ]
    return notebook(cells)


def lesson03() -> dict:
    model_src = (ROOT / "starters/lesson03/forkcast/model.py").read_text(encoding="utf-8")
    prep_src = (ROOT / "starters/lesson03/forkcast/preprocess.py").read_text(encoding="utf-8")
    train_src = (ROOT / "starters/lesson03/forkcast/train.py").read_text(encoding="utf-8")
    cells = [
        md("""
# Lesson 3: nn.Module, nn.Linear and optimizers

Goes with the Lesson 3 page of the Forkcast course. The model becomes an `nn.Module`, the update becomes an optimizer, the features get normalized, and the training loop moves into `forkcast/train.py` for good.

Runtime: CPU is enough.
"""),
        md("## 1. Clone your repository and install the package"),
        code(SETUP),
        md("## 2. Meet nn.Linear"),
        code('''
import torch
from torch import nn
from forkcast.data import make_orders, FEATURES, TRUE_WEIGHTS, TRUE_BIAS
X, y = make_orders(400, seed=0)

layer = nn.Linear(4, 1)
print(layer.weight.shape, layer.bias.shape)
print(layer.weight)
print(layer(X).shape)                                   # note the trailing 1
print(torch.allclose(layer(X), X @ layer.weight.T + layer.bias))
'''),
        md("## 3. The model: `forkcast/model.py`\n\nTwo lines in `__init__`, one in `forward`. Re-run the cell after editing."),
        code("%%writefile forkcast/model.py\n" + model_src),
        code('''
import importlib
from forkcast import model as model_module
importlib.reload(model_module)
from forkcast.model import LinearRegressor

m = LinearRegressor(4)
print(m(X).shape)                                        # torch.Size([400]) -- squeezed
print(torch.allclose(m(X), X @ m.linear.weight[0] + m.linear.bias[0]))
print(sum(p.numel() for p in m.parameters()), "parameters")
'''),
        md("## 4. The normalizer: `forkcast/preprocess.py`"),
        code("%%writefile forkcast/preprocess.py\n" + prep_src),
        code('''
from forkcast import preprocess
importlib.reload(preprocess)
from forkcast.preprocess import Normalizer

norm = Normalizer().fit(X)
print(norm.mean, norm.std)
Xn = norm.transform(X)
print(Xn.mean(dim=0), Xn.std(dim=0, correction=0))      # zeros and ones
'''),
        md("## 5. The toolkit: `forkcast/train.py`\n\n`train_one_epoch`, `evaluate`, `fit`, `save`, `load`. The docstrings give the order of the lines."),
        code("%%writefile forkcast/train.py\n" + train_src),
        md("## 6. Fit on normalized features, and read what it learned"),
        code('''
from forkcast import train
importlib.reload(train)

torch.manual_seed(0)
model = LinearRegressor(4)
losses = train.fit(model, Xn, y, epochs=300, lr=0.1)
print(f"loss {losses[0]:.1f} -> {losses[-1]:.2f}")
print("normalized-scale weights:", model.linear.weight[0].detach(), "bias:", model.linear.bias.detach())

w, b = model.linear.weight[0].detach(), model.linear.bias[0].detach()
print("original-scale weights:", w / norm.std, "   truth", TRUE_WEIGHTS)
print("original-scale bias:   ", (b - (w * norm.mean / norm.std).sum()).item(), "   truth", TRUE_BIAS)

import matplotlib.pyplot as plt
plt.plot(losses); plt.yscale("log"); plt.xlabel("epoch"); plt.ylabel("loss (log scale)");
'''),
        md("### Break it on purpose\n\nThe same learning rate on raw features, with SGD and then with Adam."),
        code('''
torch.manual_seed(0)
raw = train.fit(LinearRegressor(4), X, y, epochs=60, lr=0.1)
print("raw features, SGD lr=0.1:", [round(v, 1) for v in raw[:4]], "...", raw[-1])

torch.manual_seed(0)
adam = train.fit(LinearRegressor(4), X, y, epochs=300, lr=0.1, optimizer_cls=torch.optim.Adam)
print("raw features, Adam lr=0.1, 300 epochs:", round(adam[-1], 2))
'''),
        md("## 7. Save and load"),
        code('''
train.save(model, "linear.pt")
fresh = train.load(LinearRegressor(4), "linear.pt")
print(torch.equal(fresh(Xn), model(Xn)))
print(list(model.state_dict().keys()))
'''),
        md("## 8. Definition of done\n\nExpected: `7 passed`, then `27 passed` with no failures for everything so far (skipped tests belong to lessons you have not started)."),
        code("!pytest tests/test_03.py -q"),
        code("!pytest -m fast -q"),
        md("## 9. Commit and push"),
        code(PUSH.replace("{COMMIT_MESSAGE}", "Lesson 3: nn.Module, optimizers, normalization")),
    ]
    return notebook(cells)


def lesson04() -> dict:
    model_src = (ROOT / "starters/lesson04/forkcast/model.py").read_text(encoding="utf-8")
    cells = [
        md("""
# Lesson 4: The first neural network

Goes with the Lesson 4 page of the Forkcast course. The deliveries stop following a straight line; you watch the linear model fail, then build an MLP that bends.

Runtime: CPU is enough.
"""),
        md("## 1. Clone your repository and install the package"),
        code(SETUP),
        md("## 2. The new deliveries, and the old model\n\n`make_orders(nonlinear=True)`: rain slows every kilometre, and past 7 km riders take the bike-lane detour."),
        code('''
import torch
from torch import nn
from forkcast.data import make_orders, FEATURES
from forkcast.model import LinearRegressor
from forkcast.preprocess import Normalizer
from forkcast import train

X, y = make_orders(2000, seed=0, nonlinear=True)
norm = Normalizer().fit(X)
Xn = norm.transform(X)

torch.manual_seed(0)
lin = LinearRegressor(4)
lin_losses = train.fit(lin, Xn, y, epochs=300, lr=0.1)
print(f"linear model: loss {lin_losses[-1]:.2f}   (noise floor is about 4)")
'''),
        md("### The residual plot\n\nPrediction minus actual, against distance, coloured by rain. A model that captured everything leaves a shapeless band around zero."),
        code('''
import matplotlib.pyplot as plt

def residual_plot(model, title):
    with torch.no_grad():
        residual = model(Xn) - y
    plt.figure(figsize=(7, 3.5))
    plt.scatter(X[:, 0], residual, s=6, c=X[:, 2], cmap="coolwarm")
    plt.axhline(0, color="k", ls="--"); plt.xlabel("km"); plt.ylabel("prediction - actual"); plt.title(title)

residual_plot(lin, "LinearRegressor: the bend at 7 km and the rain drift are both visible")
'''),
        md("## 3. Write `count_parameters` and `MLP`\n\nIn the cell below, under `LinearRegressor`. Build a list of layers in a loop over `hidden`, append the output layer, wrap with `nn.Sequential(*layers)`."),
        code("%%writefile forkcast/model.py\n" + model_src),
        code('''
import importlib
from forkcast import model as model_module
importlib.reload(model_module)
from forkcast.model import MLP, count_parameters

model = MLP(4, (32, 32))
print(model.net)
print(model(Xn).shape)                      # torch.Size([2000])
print(count_parameters(model))              # 1249
print(count_parameters(MLP(4, ())))         # 5: no hidden layers is a straight line
'''),
        md("## 4. Train it with the Lesson 3 toolkit, unchanged"),
        code('''
torch.manual_seed(0)
model = MLP(4, (32, 32))
losses = train.fit(model, Xn, y, epochs=300, lr=0.01, optimizer_cls=torch.optim.Adam)
print(f"MLP: loss {losses[0]:.1f} -> {losses[-1]:.2f}")
plt.plot(lin_losses, label="linear"); plt.plot(losses, label="MLP(4, (32, 32))")
plt.yscale("log"); plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend();
residual_plot(model, "MLP: a band around zero, no trend")
'''),
        md("### Capacity\n\nSame recipe, different widths. Training loss only; Lesson 5 adds the loss on unseen deliveries."),
        code('''
for hidden in [(8,), (32,), (32, 32), (64, 64)]:
    torch.manual_seed(0)
    m = MLP(4, hidden)
    l = train.fit(m, Xn, y, epochs=300, lr=0.01, optimizer_cls=torch.optim.Adam)
    print(f"MLP(4, {hidden!s:10s}) {count_parameters(m):5d} parameters   loss {l[-1]:.2f}")
'''),
        md("### Break it on purpose, twice\n\nNo activation: 1249 parameters, the loss of a straight line. SGD at Lesson 3's learning rate: nan."),
        code('''
torch.manual_seed(0)
flat = train.fit(MLP(4, (32, 32), activation=nn.Identity), Xn, y, epochs=300, lr=0.01, optimizer_cls=torch.optim.Adam)
print("no activation:", round(flat[-1], 2), "  (linear model was", round(lin_losses[-1], 2), ")")

torch.manual_seed(0)
sgd = train.fit(MLP(4, (32, 32)), Xn, y, epochs=300, lr=0.1)
print("SGD lr=0.1:", sgd[-1])
'''),
        md("## 5. Definition of done\n\nExpected: `7 passed`, then `34 passed` with no failures for everything so far (skipped tests belong to lessons you have not started)."),
        code("!pytest tests/test_04.py -q"),
        code("!pytest -m fast -q"),
        md("## 6. Commit and push"),
        code(PUSH.replace("{COMMIT_MESSAGE}", "Lesson 4: the first neural network")),
    ]
    return notebook(cells)


def lesson05() -> dict:
    split_src = (ROOT / "starters/lesson05/forkcast/split.py").read_text(encoding="utf-8")
    train_src = (ROOT / "starters/lesson05/forkcast/train.py").read_text(encoding="utf-8")
    model_src = (ROOT / "starters/lesson05/forkcast/model.py").read_text(encoding="utf-8")
    cells = [
        md("""
# Lesson 5: Generalization

Goes with the Lesson 5 page of the Forkcast course. Splits, overfitting on purpose, early stopping, weight decay and dropout, and the two classic leaks.

Runtime: CPU is enough. The `%%writefile` cells for `train.py` and `model.py` contain the reference solutions of Lessons 3 and 4 with the new parts marked; keep your own versions if you prefer and add the new parts by hand.
"""),
        md("## 1. Clone your repository and install the package"),
        code(SETUP),
        md("## 2. Split: `forkcast/split.py`"),
        code("%%writefile forkcast/split.py\n" + split_src),
        code('''
import importlib, torch
from torch import nn
import matplotlib.pyplot as plt
from forkcast.data import make_orders
from forkcast import split as split_module
importlib.reload(split_module)
from forkcast.split import train_val_test_split
from forkcast.preprocess import Normalizer

X, y = make_orders(2000, seed=0, nonlinear=True)
(X_tr, y_tr), (X_val, y_val), (X_te, y_te) = train_val_test_split(X, y, seed=0)
print(len(X_tr), len(X_val), len(X_te))
norm = Normalizer().fit(X_tr)                       # training part only
Xn_tr, Xn_val, Xn_te = norm.transform(X_tr), norm.transform(X_val), norm.transform(X_te)
'''),
        md("## 3. `fit_with_validation`: `forkcast/train.py`\n\nBelow `load`. One `train_one_epoch`, one `evaluate`, keep the best `state_dict` with `copy.deepcopy`, stop after `patience` epochs without improvement, load the best weights back before returning."),
        code("%%writefile forkcast/train.py\n" + train_src),
        code('''
from forkcast import train
importlib.reload(train)
from forkcast.model import MLP

torch.manual_seed(0)
model = MLP(4, (32, 32))
h = train.fit_with_validation(model, Xn_tr, y_tr, Xn_val, y_val, epochs=300, lr=0.01)
print("baseline on 1400 deliveries: train", round(h["train"][-1], 2), "val", round(h["val"][-1], 2))
'''),
        md("## 4. Overfit on purpose\n\n40 deliveries, 4545 parameters, 800 epochs."),
        code('''
small = Normalizer().fit(X_tr[:40])
Xs, ys, Xs_val = small.transform(X_tr[:40]), y_tr[:40], small.transform(X_val)
torch.manual_seed(0)
h = train.fit_with_validation(MLP(4, (64, 64)), Xs, ys, Xs_val, y_val, epochs=800, lr=0.01)
print("train", round(h["train"][-1], 2), "val", round(h["val"][-1], 2), "best val", round(h["best_val"], 2), "at epoch", h["best_epoch"])
plt.plot(h["train"], label="train"); plt.plot(h["val"], label="validation")
plt.axvline(h["best_epoch"], ls="--", c="k"); plt.yscale("log"); plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend();
'''),
        md("## 5. Stop early, then regularize\n\nAdd `dropout` to `MLP` in the cell below: when `dropout > 0`, append `nn.Dropout(dropout)` after each activation."),
        code("%%writefile forkcast/model.py\n" + model_src),
        code('''
from forkcast import model as model_module
importlib.reload(model_module)
from forkcast.model import MLP
print(MLP(4, (64, 64), dropout=0.2).net)

torch.manual_seed(0)
m = MLP(4, (64, 64))
h = train.fit_with_validation(m, Xs, ys, Xs_val, y_val, epochs=800, lr=0.01, patience=30)
print("early stopping: ran", len(h["val"]), "epochs, best", round(h["best_val"], 2), "at", h["best_epoch"],
      "| model carries best weights:", round(train.evaluate(m, Xs_val, y_val, nn.MSELoss()), 2))

torch.manual_seed(0)
h = train.fit_with_validation(MLP(4, (64, 64), dropout=0.2), Xs, ys, Xs_val, y_val, epochs=800, lr=0.01, weight_decay=1e-2)
print("dropout 0.2 + weight decay 1e-2: best val", round(h["best_val"], 2), "at", h["best_epoch"],
      "| training loss with dropout on:", round(h["train"][-1], 1))
'''),
        md("## 6. The learning curve"),
        code('''
for n in [50, 100, 200, 400, 800, 1400]:
    nm = Normalizer().fit(X_tr[:n]); torch.manual_seed(0)
    h = train.fit_with_validation(MLP(4, (32, 32)), nm.transform(X_tr[:n]), y_tr[:n], nm.transform(X_val), y_val,
                                  epochs=300, lr=0.01, patience=30)
    print(f"{n:5d} training deliveries -> best validation loss {h['best_val']:.2f}")
'''),
        md("### Break it on purpose: cut without shuffling"),
        code('''
order = X[:, 0].argsort()                            # deliveries sorted by distance
Xs_, ys_ = X[order], y[order]
X_a, y_a, X_b, y_b = Xs_[:1400], ys_[:1400], Xs_[1700:], ys_[1700:]
nm = Normalizer().fit(X_a); torch.manual_seed(0); m = MLP(4, (32, 32))
train.fit_with_validation(m, nm.transform(X_a), y_a, nm.transform(Xs_[1400:1700]), ys_[1400:1700], epochs=300, lr=0.01)
print("test loss on the farthest 15%:", round(train.evaluate(m, nm.transform(X_b), y_b, nn.MSELoss()), 2),
      "| those deliveries start at", round(X_b[:, 0].min().item(), 1), "km")
'''),
        md("## 7. The test loss, once"),
        code("print(round(train.evaluate(model, Xn_te, y_te, nn.MSELoss()), 2))"),
        md("## 8. Definition of done\n\nExpected: `7 passed`, then `41 passed` with no failures for everything so far (skipped tests belong to lessons you have not started)."),
        code("!pytest tests/test_05.py -q"),
        code("!pytest -m fast -q"),
        md("## 9. Commit and push"),
        code(PUSH.replace("{COMMIT_MESSAGE}", "Lesson 5: generalization")),
    ]
    return notebook(cells)


def lesson06() -> dict:
    trips_src = (ROOT / "starters/lesson06/forkcast/trips.py").read_text(encoding="utf-8")
    model_src = (ROOT / "starters/lesson06/forkcast/model.py").read_text(encoding="utf-8")
    train_src = (ROOT / "starters/lesson06/forkcast/train.py").read_text(encoding="utf-8")
    cells = [
        md("""
# Lesson 6: Dataset, DataLoader and categorical embeddings

Goes with the Lesson 6 page of the Forkcast course. Real taxi trips as a stand-in for deliveries: cleaning, a `Dataset`, a `DataLoader`, mini-batches, and `nn.Embedding` for zone ids.

Runtime: CPU is enough for the bundled sample; a GPU makes the real month faster at the end.
"""),
        md("## 1. Clone your repository and install the package"),
        code(SETUP),
        md("## 2. Load, clean, split\n\n`clean_trips` and `load_trips` are provided. Read them: they print what they remove."),
        code("%%writefile forkcast/trips.py\n" + trips_src),
        code('''
import importlib, torch
from torch import nn
import matplotlib.pyplot as plt
from forkcast import trips
importlib.reload(trips)
from forkcast.preprocess import Normalizer

frame = trips.load_trips()
print(frame.head(3))
print("minutes: mean", round(frame["minutes"].mean(), 1), "std", round(frame["minutes"].std(), 1))
train_frame, val_frame, test_frame = trips.split_frame(frame, seed=0)
print(len(train_frame), len(val_frame), len(test_frame))
norm = Normalizer().fit(torch.tensor(train_frame[trips.CONTINUOUS].values, dtype=torch.float32))
print(norm.mean)
'''),
        md("## 3. `TripsDataset` and `make_loaders`\n\nFill in the four `raise NotImplementedError` lines in the `trips.py` cell above and re-run it, then check:"),
        code('''
importlib.reload(trips)
ds = trips.TripsDataset(train_frame, norm)
print(len(ds)); print(ds[0])
train_loader, val_loader = trips.make_loaders(train_frame, val_frame, norm, batch_size=256)
print(len(train_loader), len(val_loader))
cont, cat, minutes = next(iter(train_loader))
print(cont.shape, cat.shape, minutes.shape, cat.dtype)
'''),
        md("## 4. `TabularNet`: `forkcast/model.py`"),
        code("%%writefile forkcast/model.py\n" + model_src),
        code('''
from forkcast import model as model_module
importlib.reload(model_module)
from forkcast.model import TabularNet, MLP, count_parameters
model = TabularNet(4)
print(model(cont, cat).shape)             # torch.Size([256])
print(count_parameters(model))            # 9825
print(model.pu.weight.shape)              # torch.Size([266, 8])
'''),
        md("## 5. The toolkit learns to take loaders: `forkcast/train.py`"),
        code("%%writefile forkcast/train.py\n" + train_src),
        code('''
from forkcast import train
importlib.reload(train)
torch.manual_seed(0)
model = TabularNet(4)
history = train.fit_loader(model, train_loader, val_loader, epochs=25, lr=3e-3)
print([round(v, 1) for v in history["val"][:5]])
print("best validation", round(history["best_val"], 2), "at epoch", history["best_epoch"])
plt.plot(history["train"], label="train"); plt.plot(history["val"], label="validation"); plt.yscale("log"); plt.legend();
'''),
        md("### Do the zones carry information?"),
        code('''
class ContinuousOnly(nn.Module):           # same MLP, zone ids ignored
    def __init__(self):
        super().__init__()
        self.mlp = MLP(4, (64, 64))
    def forward(self, x_cont, x_cat):
        return self.mlp(x_cont)

torch.manual_seed(0)
without = train.fit_loader(ContinuousOnly(), train_loader, val_loader, epochs=25, lr=3e-3)["best_val"]
print(f"continuous features only:  best validation loss {without:.2f}")
print(f"with zone embeddings:      best validation loss {history['best_val']:.2f}")
'''),
        md("### Break it on purpose\n\nZone ids are indices. Feed them as floats and the embedding refuses."),
        code('''
try:
    model(cont, cat.float())
except RuntimeError as err:
    print("RuntimeError:", str(err)[:120])
'''),
        md("## 6. A real month\n\nDownloads about 50 MB and samples 200,000 trips. Everything above runs unchanged on it; compare the cleaning report and the curves."),
        code('''
!python tools/get_taxi.py --month 2024-01 --rows 200000
real = trips.load_trips("data/taxi_2024-01.csv")
r_train, r_val, r_test = trips.split_frame(real, seed=0)
r_norm = Normalizer().fit(torch.tensor(r_train[trips.CONTINUOUS].values, dtype=torch.float32))
r_tl, r_vl = trips.make_loaders(r_train, r_val, r_norm, batch_size=512)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
real_model = TabularNet(4)
h = train.fit_loader(real_model, r_tl, r_vl, epochs=10, lr=3e-3, device=device)
print("best validation", round(h["best_val"], 2), "at epoch", h["best_epoch"])
emb = real_model.pu.weight.detach().cpu()
print("zones nearest to JFK (132):", torch.cdist(emb[132:133], emb).squeeze(0).argsort()[:6].tolist())
'''),
        md("## 7. Definition of done\n\nExpected: `7 passed`, then `48 passed` with no failures for everything so far (skipped tests belong to lessons you have not started)."),
        code("!pytest tests/test_06.py -q"),
        code("!pytest -m fast -q"),
        md("## 8. Commit and push"),
        code(PUSH.replace("{COMMIT_MESSAGE}", "Lesson 6: datasets, loaders, embeddings")),
    ]
    return notebook(cells)


def lesson07() -> dict:
    vision_src = (ROOT / "starters/lesson07/forkcast/vision.py").read_text(encoding="utf-8")
    cells = [
        md("""
# Lesson 7: Images and the first CNN

Goes with the Lesson 7 page of the Forkcast course. Images as (N, C, H, W) tensors, convolution, pooling, a small CNN, and cross-entropy.

Runtime: pick a **GPU** (Runtime → Change runtime type). The synthetic shapes train on a CPU; the dish photos at the end want the GPU.
"""),
        md("## 1. Clone your repository and install the package"),
        code(SETUP),
        md("## 2. `forkcast/vision.py`\n\nThe shape generator and loaders are provided. Fill in `conv_output_size`, `SmallCNN`, `accuracy` and `cross_entropy_by_hand` as the page walks through them, re-running the cell after each edit."),
        code("%%writefile forkcast/vision.py\n" + vision_src),
        md("## 3. Look at the pictures"),
        code('''
import importlib, torch
from torch import nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from forkcast import vision, train
importlib.reload(vision)
from forkcast.model import count_parameters

images, labels = vision.make_shapes(16, seed=0)
print(images.shape, images.dtype, labels)
print("per-channel mean:", images.mean(dim=(0, 2, 3)))
plt.figure(figsize=(10, 3)); plt.axis("off"); plt.imshow(make_grid(images, nrow=8).permute(1, 2, 0))
print([vision.SHAPE_CLASSES[k] for k in labels])
'''),
        md("## 4. Shape arithmetic, checked against a real layer"),
        code('''
importlib.reload(vision)
for size, k, s, p in [(32, 3, 1, 1), (32, 3, 2, 1), (32, 5, 1, 0), (64, 7, 2, 3)]:
    real = nn.Conv2d(1, 1, k, stride=s, padding=p)(torch.zeros(1, 1, size, size)).shape[-1]
    print(size, k, s, p, "->", vision.conv_output_size(size, k, s, p), real)
'''),
        md("### Slide a kernel by hand\n\nThe vertical-edge kernel from the page, applied to the red channel of a square."),
        code('''
edge = nn.Conv2d(1, 1, kernel_size=3, bias=False)
with torch.no_grad():
    edge.weight[:] = torch.tensor([[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]])
square = images[labels == 1][:1, :1]              # one image, red channel only: (1, 1, 32, 32)
response = edge(square)
fig, ax = plt.subplots(1, 2, figsize=(6, 3))
ax[0].imshow(square[0, 0], cmap="gray"); ax[0].set_title("input"); ax[1].imshow(response[0, 0].detach(), cmap="coolwarm"); ax[1].set_title("vertical edges")
print(square.shape, "->", response.shape)
'''),
        md("## 5. The network, shape by shape"),
        code('''
importlib.reload(vision)
model = vision.SmallCNN()
print(model.features(images).shape)          # (16, 32, 8, 8)
print(model(images).shape)                   # (16, 4)
print(count_parameters(model))               # 136580
'''),
        md("## 6. `accuracy` and `cross_entropy_by_hand`"),
        code('''
importlib.reload(vision)
logits = torch.tensor([[2.0, 0.1, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 0.5, 0.4], [1.0, 0.0, 0.0, 0.0]])
y = torch.tensor([0, 1, 2, 3])
print(vision.accuracy(logits, y))
print(vision.cross_entropy_by_hand(logits, y), nn.CrossEntropyLoss()(logits, y))
'''),
        md("## 7. Train on the shapes"),
        code('''
train_loader, val_loader = vision.shape_loaders(n_train=1000, n_val=400)
torch.manual_seed(0)
model = vision.SmallCNN()
history = train.fit_loader(model, train_loader, val_loader, epochs=6, lr=1e-3, loss_fn=nn.CrossEntropyLoss())
print([round(v, 2) for v in history["val"]])
print("validation accuracy:", vision.accuracy_loader(model, val_loader))
out = model(images[:1]); print(out, torch.softmax(out, dim=1), "true:", vision.SHAPE_CLASSES[labels[0]])
'''),
        md("### Break it on purpose, twice"),
        code('''
try:
    model(images.permute(0, 2, 3, 1))                # channels last
except RuntimeError as err:
    print("RuntimeError:", str(err)[:140])

torch.manual_seed(0)
wrong = vision.SmallCNN()
train.fit_loader(wrong, train_loader, val_loader, epochs=6, lr=1e-3, loss_fn=lambda logits, yy: nn.MSELoss()(logits[:, 0], yy.float()))
print("MSE on the class index: accuracy", vision.accuracy_loader(wrong, val_loader))
'''),
        md("## 8. Real dish photos\n\nTen classes of Food-101, streamed from the Hugging Face Hub so nothing large is downloaded, resized to 64×64. A few minutes on the GPU."),
        code('''
!pip install -q datasets
from datasets import load_dataset
from torchvision.transforms import v2
from torch.utils.data import DataLoader, TensorDataset

KEEP = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]                 # ten of the 101 classes
stream = load_dataset("ethz/food101", split="train", streaming=True)
names = stream.features["label"].names
to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize((64, 64), antialias=True)])
xs, ys = [], []
for ex in stream:
    if ex["label"] in KEEP:
        xs.append(to_tensor(ex["image"].convert("RGB"))); ys.append(KEEP.index(ex["label"]))
    if len(xs) >= 2000:
        break
X = torch.stack(xs); Y = torch.tensor(ys)
print(X.shape, [names[k] for k in KEEP])
mean, std = X.mean(dim=(0, 2, 3)), X.std(dim=(0, 2, 3))
Xn = (X - mean[None, :, None, None]) / std[None, :, None, None]
food_train = DataLoader(TensorDataset(Xn[:1600], Y[:1600]), batch_size=64, shuffle=True, drop_last=True)
food_val = DataLoader(TensorDataset(Xn[1600:], Y[1600:]), batch_size=64)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
food_model = vision.SmallCNN(n_classes=10, size=64)
h = train.fit_loader(food_model, food_train, food_val, epochs=15, lr=1e-3, loss_fn=nn.CrossEntropyLoss(), device=device, patience=5)
print("validation accuracy on real dishes:", round(vision.accuracy_loader(food_model, food_val, device), 3), "(chance is 0.10)")
plt.plot(h["train"], label="train"); plt.plot(h["val"], label="validation"); plt.legend();
'''),
        md("## 9. Definition of done\n\nExpected: `7 passed`, then `55 passed` with no failures for everything so far (skipped tests belong to lessons you have not started)."),
        code("!pytest tests/test_07.py -q"),
        code("!pytest -m fast -q"),
        md("## 10. Commit and push"),
        code(PUSH.replace("{COMMIT_MESSAGE}", "Lesson 7: images and the first CNN")),
    ]
    return notebook(cells)


def lesson08() -> dict:
    train_src = (ROOT / "starters/lesson08/forkcast/train.py").read_text(encoding="utf-8")
    vision_src = (ROOT / "starters/lesson08/forkcast/vision.py").read_text(encoding="utf-8")
    cells = [
        md("""
# Lesson 8: GPU training for real

Goes with the Lesson 8 page of the Forkcast course. Devices, mixed precision, augmentation, checkpoints and resume, timing and logging: training loop v2.

Runtime: pick a **GPU**. Every cell also runs on a CPU (autocast falls back to bfloat16), just slower.
"""),
        md("## 1. Clone your repository and install the package"),
        code(SETUP),
        md("## 2. Meet the device"),
        code('''
import importlib, time, torch
from torch import nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from forkcast import trips, train, vision
from forkcast.model import TabularNet
from forkcast.preprocess import Normalizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, torch.cuda.get_device_name(0) if device.type == "cuda" else "(no GPU: everything below still runs)")
x = torch.randn(1000, 1000); y = x.to(device)
print("tensor .to() copies:", x.device, "->", y.device)
model = TabularNet(4).to(device)
print("module .to() moves in place:", next(model.parameters()).device)

frame = trips.load_trips(verbose=False)
train_frame, val_frame, _ = trips.split_frame(frame, seed=0)
norm = Normalizer().fit(torch.tensor(train_frame[trips.CONTINUOUS].values, dtype=torch.float32))
train_loader, val_loader = trips.make_loaders(train_frame, val_frame, norm, batch_size=512)
cont, cat, minutes = next(iter(train_loader))
try:
    model(cont, cat)                                     # batch still on the CPU
except RuntimeError as err:
    print("RuntimeError:", str(err)[:110])
'''),
        md("## 3. `forkcast/train.py`: training loop v2\n\nFill in `make_scaler`, `train_one_epoch_amp`, `save_checkpoint`, `load_checkpoint` and `fit_v2` as the page builds them. Re-run the cell after each edit."),
        code("%%writefile forkcast/train.py\n" + train_src),
        md("### Time an epoch on each device"),
        code('''
importlib.reload(train)
shape_train, shape_val = vision.shape_loaders(1000, 400)
for dev in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
    torch.manual_seed(0)
    h = train.fit_v2(vision.SmallCNN(), shape_train, shape_val, epochs=2, lr=1e-3, loss_fn=nn.CrossEntropyLoss(), device=dev)
    print(f"{dev}: {h['epoch_seconds'][1]:.2f} s per epoch")
'''),
        md("### Mixed precision"),
        code('''
importlib.reload(train)
print("scaler on", device, ":", train.make_scaler(device, amp=True))
def fresh():
    torch.manual_seed(0); return TabularNet(4, hidden=(32,))
with torch.autocast(device_type=device.type, dtype=train.autocast_dtype(device)):
    print("output dtype under autocast:", fresh().to(device)(cont.to(device), cat.to(device)).dtype)
fp32 = train.fit_v2(fresh(), train_loader, val_loader, epochs=3, lr=3e-3, device=device)
amp = train.fit_v2(fresh(), train_loader, val_loader, epochs=3, lr=3e-3, device=device, amp=True)
print("float32:", [round(v, 2) for v in fp32["train"]], "| mixed:", [round(v, 2) for v in amp["train"]])
print("seconds per epoch: float32", round(fp32["epoch_seconds"][-1], 3), "| mixed", round(amp["epoch_seconds"][-1], 3))
'''),
        md("## 4. Augmentation: `forkcast/vision.py`\n\n`train_transform` is provided; write `AugmentedTensorDataset`."),
        code("%%writefile forkcast/vision.py\n" + vision_src),
        code('''
importlib.reload(vision)
images, labels = vision.make_shapes(8, seed=0)
aug = vision.AugmentedTensorDataset(images, labels, vision.train_transform(32))
print("two fetches equal?", torch.equal(aug[0][0], aug[0][0]))
plt.figure(figsize=(10, 1.6)); plt.axis("off")
plt.imshow(make_grid(torch.stack([aug[0][0] for _ in range(8)]), nrow=8).permute(1, 2, 0))

small_train, small_val = vision.shape_loaders(200, 400, batch_size=32)
small_aug, _ = vision.augmented_shape_loaders(200, 400, batch_size=32)
torch.manual_seed(0); plain = vision.SmallCNN(); train.fit_v2(plain, small_train, small_val, epochs=15, lr=1e-3, loss_fn=nn.CrossEntropyLoss(), device=device)
torch.manual_seed(0); augm = vision.SmallCNN(); train.fit_v2(augm, small_aug, small_val, epochs=15, lr=1e-3, loss_fn=nn.CrossEntropyLoss(), device=device)
print("200 shapes, 15 epochs: plain", round(vision.accuracy_loader(plain, small_val, device), 3), "| augmented", round(vision.accuracy_loader(augm, small_val, device), 3))
'''),
        md("## 5. Checkpoints: resume is exact"),
        code('''
importlib.reload(train)
full = train.fit_v2(fresh(), train_loader, val_loader, epochs=5, lr=3e-3, ckpt_path="full.pt")
train.fit_v2(fresh(), train_loader, val_loader, epochs=3, lr=3e-3, ckpt_path="r.pt")                 # "crash" after epoch 2
resumed = train.fit_v2(fresh(), train_loader, val_loader, epochs=5, lr=3e-3, ckpt_path="r.pt", resume=True)
print("uninterrupted:", [round(v, 2) for v in full["train"]])
print("resumed:      ", [round(v, 2) for v in resumed["train"]])
print(sorted(torch.load("full.pt", weights_only=True).keys()))
'''),
        md("### Break it on purpose: weights only"),
        code('''
train.fit_v2(fresh(), train_loader, val_loader, epochs=3, lr=3e-3, ckpt_path="r2.pt")
ckpt = torch.load("r2.pt", weights_only=True)
ckpt["optimizer"] = torch.optim.Adam(fresh().parameters(), lr=3e-3).state_dict()     # throw Adam's memory away
torch.save(ckpt, "r2.pt")
weights_only = train.fit_v2(fresh(), train_loader, val_loader, epochs=5, lr=3e-3, ckpt_path="r2.pt", resume=True)
print("weights only: ", [round(v, 2) for v in weights_only["train"]])
'''),
        md("## 6. TensorBoard"),
        code('''
%load_ext tensorboard
%tensorboard --logdir runs
h = train.fit_v2(TabularNet(4), train_loader, val_loader, epochs=20, lr=3e-3, device=device, amp=True, log_dir="runs/tabular")
print("best validation", round(h["best_val"], 2), "at epoch", h["best_epoch"], "|", round(sum(h["epoch_seconds"]), 1), "s total")
'''),
        md("## 7. Definition of done\n\nExpected: `7 passed`, then `62 passed` with no failures for everything so far (skipped tests belong to lessons you have not started)."),
        code("!pytest tests/test_08.py -q"),
        code("!pytest -m fast -q"),
        md("## 8. Commit and push"),
        code(PUSH.replace("{COMMIT_MESSAGE}", "Lesson 8: training loop v2")),
    ]
    return notebook(cells)


def lesson09() -> dict:
    transfer_src = (ROOT / "starters/lesson09/forkcast/transfer.py").read_text(encoding="utf-8")
    cells = [
        md("""
# Lesson 9: Transfer learning and evaluation

Goes with the Lesson 9 page of the Forkcast course. A pretrained ResNet-18, frozen; a new head trained on cached features in seconds; the top block fine-tuned with its own learning rate; a confusion matrix to read.

Runtime: pick a **GPU**. The pretrained weights (45 MB) download once.
"""),
        md("## 1. Clone your repository and install the package"),
        code(SETUP),
        md("## 2. Twenty dishes at 224×224\n\nStreamed from the Hugging Face Hub: 150 training and 50 validation photos per class, preprocessed exactly as the pretrained network expects."),
        code('''
!pip install -q datasets
import importlib, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from torchvision import models
from torchvision.utils import make_grid
from datasets import load_dataset
from forkcast import train, vision
from forkcast.model import count_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = models.ResNet18_Weights.DEFAULT
prep = weights.transforms()                              # resize 256, crop 224, ImageNet normalization
print(prep)

KEEP = list(range(0, 101, 5))[:20]                       # twenty of the 101 classes
stream = load_dataset("ethz/food101", split="train", streaming=True)
names = [stream.features["label"].names[k] for k in KEEP]
per_class = {k: 0 for k in KEEP}; xs, ys = [], []
for ex in stream:
    k = ex["label"]
    if k in per_class and per_class[k] < 200:
        xs.append(prep(ex["image"].convert("RGB"))); ys.append(KEEP.index(k)); per_class[k] += 1
    if all(v >= 200 for v in per_class.values()):
        break
X = torch.stack(xs); Y = torch.tensor(ys)
perm = torch.randperm(len(Y), generator=torch.Generator().manual_seed(0))
X, Y = X[perm], Y[perm]
n_val = 20 * 50
train_loader = DataLoader(TensorDataset(X[n_val:], Y[n_val:]), batch_size=64, shuffle=True, drop_last=True)
val_loader = DataLoader(TensorDataset(X[:n_val], Y[:n_val]), batch_size=64)
print(X.shape, names)
'''),
        md("## 3. `forkcast/transfer.py`\n\nWrite `freeze`, `build_transfer_model`, `param_groups`, `top_k_accuracy` and `confusion_matrix`; the feature cache and the matrix helpers are provided."),
        code("%%writefile forkcast/transfer.py\n" + transfer_src),
        md("### Freeze, replace the head"),
        code('''
from forkcast import transfer
importlib.reload(transfer)
model = transfer.build_transfer_model(20)                # pretrained=True: downloads the weights once
print("trainable:", count_parameters(model), "of", sum(p.numel() for p in model.parameters()))
x_batch, _ = next(iter(val_loader))
print(model.to(device)(x_batch.to(device)).shape)
'''),
        md("## 4. Cache the features, train the head in seconds"),
        code('''
feats, ys_tr = transfer.extract_features(model, train_loader, device)
vfeats, ys_val = transfer.extract_features(model, val_loader, device)
print(feats.shape, vfeats.shape)
head_train = DataLoader(TensorDataset(feats, ys_tr), batch_size=128, shuffle=True)
head_val = DataLoader(TensorDataset(vfeats, ys_val), batch_size=256)
torch.manual_seed(0)
head = nn.Linear(512, 20)
h = train.fit_v2(head, head_train, head_val, epochs=30, lr=1e-3, loss_fn=nn.CrossEntropyLoss(), device=device)
model.fc.load_state_dict(head.state_dict())             # the full model now classifies pixels
print("head on frozen features: validation accuracy", round(vision.accuracy_loader(model, val_loader, device), 3))
print("Lesson 7 from scratch on photos like these was far lower; chance is 0.05")
'''),
        md("## 5. Fine-tune layer4 with two learning rates"),
        code('''
importlib.reload(transfer)
ft = transfer.build_transfer_model(20, trainable_blocks=1).to(device)
ft.fc.load_state_dict(head.state_dict())
print("trainable:", count_parameters(ft))
optimizer = torch.optim.Adam(transfer.param_groups(ft, lr_head=1e-3, lr_backbone=1e-4))
scaler = train.make_scaler(device, amp=True)
for epoch in range(3):
    loss = train.train_one_epoch_amp(ft, train_loader, optimizer, nn.CrossEntropyLoss(), device, amp=True, scaler=scaler)
    print(f"epoch {epoch}: train loss {loss:.3f}, validation accuracy {vision.accuracy_loader(ft, val_loader, device):.3f}")
'''),
        md("## 6. Evaluate honestly"),
        code('''
@torch.no_grad()
def collect_logits(m, loader):
    m.eval(); out, ys_ = [], []
    for x, y in loader:
        out.append(m(x.to(device)).cpu()); ys_.append(y)
    return torch.cat(out), torch.cat(ys_)

logits, y = collect_logits(ft, val_loader)
print("top-1", round(transfer.top_k_accuracy(logits, y, 1), 3), "top-5", round(transfer.top_k_accuracy(logits, y, 5), 3))
cm = transfer.confusion_matrix(logits.argmax(1), y, 20)
acc = transfer.per_class_accuracy(cm)
for i in acc.argsort()[:5]:
    print(f"hardest: {names[i]:22s} {acc[i]:.2f}")
print("most confused:", [(names[i], names[j], n) for i, j, n in transfer.most_confused(cm, 5)])
plt.figure(figsize=(7, 6)); plt.imshow(cm, cmap="Blues"); plt.xticks(range(20), names, rotation=90, fontsize=7); plt.yticks(range(20), names, fontsize=7); plt.xlabel("predicted"); plt.ylabel("true");
'''),
        md("### Look at the worst pair"),
        code('''
i, j, n = transfer.most_confused(cm, 1)[0]
wrong = (y == i) & (logits.argmax(1) == j)
imgs = X[:n_val][wrong][:8]
mean = torch.tensor(prep.mean)[:, None, None]; std = torch.tensor(prep.std)[:, None, None]
plt.figure(figsize=(12, 2)); plt.axis("off"); plt.title(f"{names[i]} predicted as {names[j]}")
plt.imshow(make_grid((imgs * std + mean).clamp(0, 1), nrow=8).permute(1, 2, 0))
'''),
        md("### Break it on purpose: everything trainable, one big learning rate"),
        code('''
reckless = transfer.build_transfer_model(20, trainable_blocks=4).to(device)
h = train.fit_v2(reckless, train_loader, val_loader, epochs=2, lr=1e-3, loss_fn=nn.CrossEntropyLoss(), device=device, amp=True)
print("all 11M parameters at lr=1e-3: validation accuracy", round(vision.accuracy_loader(reckless, val_loader, device), 3), "(the frozen head alone did far better)")
'''),
        md("## 7. Definition of done\n\nExpected: `7 passed`, then `69 passed` with no failures for everything so far (skipped tests belong to lessons you have not started)."),
        code("!pytest tests/test_09.py -q"),
        code("!pytest -m fast -q"),
        md("## 8. Commit and push"),
        code(PUSH.replace("{COMMIT_MESSAGE}", "Lesson 9: transfer learning and evaluation")),
    ]
    return notebook(cells)


def lesson10() -> dict:
    debug_src = (ROOT / "starters/lesson10/forkcast/debug.py").read_text(encoding="utf-8")
    cells = [
        md("""
# Lesson 10: Debugging and interpreting models

Goes with the Lesson 10 page of the Forkcast course. Confident mistakes, class balance, forward hooks, Grad-CAM, and calibration, first on the synthetic shapes, then on dish photos.

Runtime: a **GPU** for the dish photos; the shapes run anywhere.
"""),
        md("## 1. Clone your repository and install the package"),
        code(SETUP),
        md("## 2. `forkcast/debug.py`\n\n`reliability_bins` and `gallery` are provided. Write `misclassified`, `class_counts`, `class_weights`, `record_activation`, `GradCAM` and `expected_calibration_error` as the page builds them."),
        code("%%writefile forkcast/debug.py\n" + debug_src),
        md("## 3. A trained Lesson 7 network, and its mistakes"),
        code('''
import importlib, torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from forkcast import vision, train, debug
from forkcast.vision import SHAPE_CLASSES

train_loader, val_loader = vision.shape_loaders(1000, 400)
torch.manual_seed(0)
model = vision.SmallCNN()
train.fit_v2(model, train_loader, val_loader, epochs=6, lr=1e-3, loss_fn=nn.CrossEntropyLoss())
images, y = vision.make_shapes(400, seed=1)
model.eval()
with torch.no_grad():
    logits = model(images)
print("accuracy", round((logits.argmax(1) == y).float().mean().item(), 3))

importlib.reload(debug)
picks = debug.misclassified(logits, y, k=16)
print([(i, SHAPE_CLASSES[t], SHAPE_CLASSES[p], round(c, 3)) for i, t, p, c in picks[:5]])
plt.figure(figsize=(10, 3)); plt.axis("off"); plt.imshow(debug.gallery(images, picks).permute(1, 2, 0))
'''),
        md("## 4. Class balance\n\nThe shapes are balanced; build a lopsided set on purpose and see what weights do to per-class accuracy."),
        code('''
importlib.reload(debug)
print("counts", debug.class_counts(y, 4), "weights", debug.class_weights(debug.class_counts(y, 4)))

x_all, y_all = vision.make_shapes(3000, seed=5)
keep = torch.cat([(y_all == 0).nonzero().flatten()[:400], (y_all == 1).nonzero().flatten()[:400], (y_all == 2).nonzero().flatten()[:400], (y_all == 3).nonzero().flatten()[:40]])
from torch.utils.data import DataLoader, TensorDataset
skew = DataLoader(TensorDataset(x_all[keep], y_all[keep]), batch_size=64, shuffle=True, drop_last=True)
counts = debug.class_counts(y_all[keep], 4); print("lopsided counts:", counts.tolist(), "weights:", debug.class_weights(counts))
for name, loss_fn in [("plain", nn.CrossEntropyLoss()), ("weighted", nn.CrossEntropyLoss(weight=debug.class_weights(counts)))]:
    torch.manual_seed(0); m = vision.SmallCNN()
    train.fit_v2(m, skew, val_loader, epochs=8, lr=1e-3, loss_fn=loss_fn)
    with torch.no_grad():
        p = m.eval()(images).argmax(1)
    print(f"{name:9s} per-class accuracy:", [round(((p == y) & (y == k)).sum().item() / (y == k).sum().item(), 2) for k in range(4)])
'''),
        md("## 5. Hooks and Grad-CAM"),
        code('''
importlib.reload(debug)
for i in [0, 3, 6, 7]:
    print(f"features[{i}]:", tuple(debug.record_activation(model, model.features[i], images[:8]).shape))
print("hooks left on features[6]:", len(model.features[6]._forward_hooks))

cam = debug.GradCAM(model, model.features[6])
heat = cam(images[:8])
heat_wrong = cam(images[:8], class_idx=(y[:8] + 1) % 4)
cam.remove()
big = F.interpolate(heat[:, None], size=32, mode="bilinear", align_corners=False)[:, 0]
fig, axes = plt.subplots(2, 8, figsize=(14, 3.6))
for k in range(8):
    axes[0, k].imshow(images[k].permute(1, 2, 0)); axes[0, k].set_title(SHAPE_CLASSES[y[k]], fontsize=8); axes[0, k].axis("off")
    axes[1, k].imshow(images[k].permute(1, 2, 0)); axes[1, k].imshow(big[k], cmap="jet", alpha=0.5); axes[1, k].axis("off")
print("heatmap shape", tuple(heat.shape), "| same map for a different class?", torch.allclose(heat, heat_wrong))
'''),
        md("## 6. Calibration"),
        code('''
importlib.reload(debug)
probs = torch.softmax(logits, dim=1)
print("ECE", round(debug.expected_calibration_error(probs, y), 3), "| mean confidence", round(probs.max(1).values.mean().item(), 3))
conf, acc, count = debug.reliability_bins(probs, y)
plt.figure(figsize=(4, 4)); plt.bar(conf[count > 0], acc[count > 0], width=0.08, alpha=0.7); plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("confidence"); plt.ylabel("accuracy");
'''),
        md("## 7. The same four tools on dish photos\n\nLesson 9's recipe, compressed: ten classes, a frozen pretrained ResNet-18, a head trained on cached features."),
        code('''
!pip install -q datasets
from datasets import load_dataset
from torchvision import models
from forkcast import transfer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = models.ResNet18_Weights.DEFAULT; prep = weights.transforms()
KEEP = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
stream = load_dataset("ethz/food101", split="train", streaming=True)
names = [stream.features["label"].names[k] for k in KEEP]
per_class = {k: 0 for k in KEEP}; xs, ys = [], []
for ex in stream:
    k = ex["label"]
    if k in per_class and per_class[k] < 120:
        xs.append(prep(ex["image"].convert("RGB"))); ys.append(KEEP.index(k)); per_class[k] += 1
    if all(v >= 120 for v in per_class.values()):
        break
X = torch.stack(xs); Y = torch.tensor(ys)
perm = torch.randperm(len(Y), generator=torch.Generator().manual_seed(0)); X, Y = X[perm], Y[perm]
food_train = DataLoader(TensorDataset(X[300:], Y[300:]), batch_size=64, shuffle=True); food_val = DataLoader(TensorDataset(X[:300], Y[:300]), batch_size=64)
dish = transfer.build_transfer_model(10).to(device)
feats, ys_tr = transfer.extract_features(dish, food_train, device); vfeats, ys_val = transfer.extract_features(dish, food_val, device)
head = nn.Linear(512, 10)
train.fit_v2(head, DataLoader(TensorDataset(feats, ys_tr), batch_size=128, shuffle=True), DataLoader(TensorDataset(vfeats, ys_val), batch_size=256), epochs=30, lr=1e-3, loss_fn=nn.CrossEntropyLoss(), device=device)
dish.fc.load_state_dict(head.state_dict()); dish.eval()
with torch.no_grad():
    dlogits = torch.cat([dish(x.to(device)).cpu() for x, _ in food_val])
dy = Y[:300]
print("accuracy", round((dlogits.argmax(1) == dy).float().mean().item(), 3), "| ECE", round(debug.expected_calibration_error(torch.softmax(dlogits, 1), dy), 3))
mean = torch.tensor(prep.mean)[:, None, None]; std = torch.tensor(prep.std)[:, None, None]
shown = (X[:300] * std + mean).clamp(0, 1)
picks = debug.misclassified(dlogits, dy, k=8)
print([(names[t], "->", names[p], round(c, 2)) for _, t, p, c in picks])
plt.figure(figsize=(12, 2)); plt.axis("off"); plt.title("most confident mistakes"); plt.imshow(debug.gallery(shown, picks).permute(1, 2, 0))

cam = debug.GradCAM(dish, dish.layer4)
sel = [p[0] for p in picks[:6]]
heat = cam(X[:300][sel].to(device)).cpu(); cam.remove()
big = F.interpolate(heat[:, None], size=224, mode="bilinear", align_corners=False)[:, 0]
fig, axes = plt.subplots(1, 6, figsize=(14, 2.6))
for k, i in enumerate(sel):
    axes[k].imshow(shown[i].permute(1, 2, 0)); axes[k].imshow(big[k], cmap="jet", alpha=0.45); axes[k].set_title(f"{names[dy[i]]} as {names[picks[k][2]]}", fontsize=8); axes[k].axis("off")
'''),
        md("### Break it on purpose: hooks that stay"),
        code('''
stash = []
handles = [model.features[6].register_forward_hook(lambda m, i, o: stash.append(o)) for _ in range(50)]
with torch.no_grad():
    model(images[:8])
print("tensors kept alive by forgotten hooks:", len(stash), "| hooks on the layer:", len(model.features[6]._forward_hooks))
for h in handles:
    h.remove()
print("after removing:", len(model.features[6]._forward_hooks))
'''),
        md("## 8. Definition of done\n\nExpected: `7 passed`, then `76 passed` with no failures for everything so far (skipped tests belong to lessons you have not started)."),
        code("!pytest tests/test_10.py -q"),
        code("!pytest -m fast -q"),
        md("## 9. Commit and push"),
        code(PUSH.replace("{COMMIT_MESSAGE}", "Lesson 10: debugging and interpreting models")),
    ]
    return notebook(cells)


LESSONS = {"lesson00.ipynb": lesson00, "lesson01.ipynb": lesson01, "lesson02.ipynb": lesson02, "lesson03.ipynb": lesson03, "lesson04.ipynb": lesson04, "lesson05.ipynb": lesson05, "lesson06.ipynb": lesson06, "lesson07.ipynb": lesson07, "lesson08.ipynb": lesson08, "lesson09.ipynb": lesson09, "lesson10.ipynb": lesson10}


def main() -> None:
    out_dir = ROOT / "notebooks"
    out_dir.mkdir(exist_ok=True)
    for name, build in LESSONS.items():
        path = out_dir / name
        path.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
