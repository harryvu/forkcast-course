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


def lesson11() -> dict:
    text_src = (ROOT / "starters/lesson11/forkcast/text.py").read_text(encoding="utf-8")
    cells = [
        md("""
# Lesson 11: Text as tensors

Goes with the Lesson 11 page of the Forkcast course. Tokens, a vocabulary, padding and masks, embeddings for words, and a bag-of-embeddings sentiment model, first on the bundled templated reviews, then on real Yelp reviews.

Runtime: CPU is fine for the templated corpus; a GPU makes the Yelp run quick.
"""),
        md("## 1. Clone your repository and install the package"),
        code(SETUP),
        md("## 2. `forkcast/text.py`\n\nWrite `tokenize`, `Vocab`, `pad_batch`, `collate_reviews` and `BagOfEmbeddings` as the page builds them; the dataset and loader helpers are provided."),
        code("%%writefile forkcast/text.py\n" + text_src),
        md("## 3. Tokens and ids"),
        code('''
import importlib, torch
from torch import nn
import matplotlib.pyplot as plt
from forkcast import text, train
from forkcast.model import count_parameters
importlib.reload(text)

frame = text.load_reviews()
print(frame.head(3).to_string())
print(text.tokenize(frame["text"][0]))
cut = int(0.8 * len(frame)); tr, va = frame.iloc[:cut], frame.iloc[cut:]
vocab = text.Vocab([text.tokenize(t) for t in tr["text"]], min_freq=1)        # training texts only
print(len(vocab), vocab.itos[:8])
ids = vocab.encode(text.tokenize("The pad thai was delicious and the delivery was late. Mixed feelings!"))
print(ids); print(vocab.decode(ids))
'''),
        md("## 4. Padding, masks, loaders"),
        code('''
importlib.reload(text)
train_loader, val_loader = text.make_text_loaders(tr["text"], tr["label"], va["text"], va["label"], vocab, batch_size=32)
ids, mask, labels = next(iter(train_loader))
print(ids.shape, mask.shape, labels.shape, ids.dtype, mask.dtype)
print("real lengths:", mask.sum(dim=1)[:8].tolist())
print(ids[0]); print(mask[0].int())
'''),
        md("## 5. The model, and the mask at work"),
        code('''
importlib.reload(text)
model = text.BagOfEmbeddings(len(vocab), dim=16, hidden=16)
short = torch.tensor([[5, 6, 7]]); padded = torch.tensor([[5, 6, 7, 0, 0, 0]])
print("padding changes nothing:", torch.allclose(model(short, short != 0), model(padded, padded != 0)))
print("pad row:", model.embedding.weight[0])

torch.manual_seed(0)
model = text.BagOfEmbeddings(len(vocab), dim=32, hidden=32)
print(count_parameters(model), "parameters")
h = train.fit_v2(model, train_loader, val_loader, epochs=8, lr=3e-3, loss_fn=nn.CrossEntropyLoss())
print([round(v, 3) for v in h["val"]])

def accuracy_text(m, loader, device="cpu"):
    m.eval(); correct = total = 0
    with torch.no_grad():
        for i_, mk, lb in loader:
            correct += (m(i_.to(device), mk.to(device)).argmax(1).cpu() == lb).sum().item(); total += len(lb)
    return correct / total
print("validation accuracy on templated reviews:", accuracy_text(model, val_loader))
'''),
        md("## 6. Real reviews: Yelp polarity\n\n40,000 reviews streamed from the Hugging Face Hub; vocabulary of the 20,000 most frequent tokens from the training part; reviews truncated to 200 tokens."),
        code('''
!pip install -q datasets
from datasets import load_dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stream = load_dataset("fancyzhx/yelp_polarity", split="train", streaming=True).shuffle(seed=0, buffer_size=10_000)
texts, labels = [], []
for ex in stream:
    texts.append(ex["text"]); labels.append(ex["label"])
    if len(texts) >= 40_000:
        break
n_val = 4_000
yelp_vocab = text.Vocab([text.tokenize(t) for t in texts[n_val:]], min_freq=2, max_size=20_000)
print("vocabulary:", len(yelp_vocab))
y_train, y_val = text.make_text_loaders(texts[n_val:], labels[n_val:], texts[:n_val], labels[:n_val], yelp_vocab, batch_size=128, max_len=200)
torch.manual_seed(0)
yelp_model = text.BagOfEmbeddings(len(yelp_vocab), dim=64, hidden=64)
h = train.fit_v2(yelp_model, y_train, y_val, epochs=5, lr=3e-3, loss_fn=nn.CrossEntropyLoss(), device=device)
print("Yelp validation accuracy:", round(accuracy_text(yelp_model, y_val, device), 3))

emb = yelp_model.embedding.weight.detach().cpu(); normed = emb / emb.norm(dim=1, keepdim=True).clamp_min(1e-8)
for w in ["delicious", "rude", "overpriced", "friendly"]:
    if w in yelp_vocab.stoi:
        sims = normed @ normed[yelp_vocab.stoi[w]]
        print(f"nearest to {w!r}:", [yelp_vocab.itos[j] for j in sims.argsort(descending=True)[1:7].tolist()])
'''),
        md("### Break it on purpose: drop the mask\n\nA plain mean over T divides by the padded length. Short reviews suffer most."),
        code('''
class Unmasked(text.BagOfEmbeddings):
    def forward(self, ids, mask):
        return self.head(self.embedding(ids).mean(dim=1))       # zeros included

torch.manual_seed(0)
bad = Unmasked(len(yelp_vocab), dim=64, hidden=64)
train.fit_v2(bad, y_train, y_val, epochs=5, lr=3e-3, loss_fn=nn.CrossEntropyLoss(), device=device)
print("without the mask:", round(accuracy_text(bad, y_val, device), 3), "| with:", round(accuracy_text(yelp_model, y_val, device), 3))
'''),
        md("## 7. Definition of done\n\nExpected: `7 passed`, then `83 passed` with no failures for everything so far (skipped tests belong to lessons you have not started)."),
        code("!pytest tests/test_11.py -q"),
        code("!pytest -m fast -q"),
        md("## 8. Commit and push"),
        code(PUSH.replace("{COMMIT_MESSAGE}", "Lesson 11: text as tensors")),
    ]
    return notebook(cells)


def lesson12() -> dict:
    att_src = (ROOT / "starters/lesson12/forkcast/attention.py").read_text(encoding="utf-8")
    cells = [
        md("""
# Lesson 12: Attention from scratch to Transformer

Goes with the Lesson 12 page of the Forkcast course. Twin reviews that defeat a bag of words, attention by hand, positional encoding, and a tiny Transformer encoder.

Runtime: CPU is fine; a GPU helps for the Yelp run at the end.
"""),
        md("## 1. Clone your repository and install the package"),
        code(SETUP),
        md("## 2. `forkcast/attention.py`\n\nThe twin-review generator and loaders are provided. Write `scaled_dot_product`, `PositionalEncoding` and `TinyTransformer` as the page builds them."),
        code("%%writefile forkcast/attention.py\n" + att_src),
        md("## 3. The bag's ceiling"),
        code('''
import importlib, math, torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from forkcast import attention, text, train
from forkcast.model import count_parameters
importlib.reload(attention)

texts, labels = attention.make_negation_reviews(6, seed=0)
for t, l in zip(texts, labels):
    print(l, t)
train_loader, val_loader, vocab = attention.negation_loaders()
print(len(vocab), "tokens")

def accuracy(model, loader, device="cpu"):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for ids, mask, y in loader:
            correct += (model(ids.to(device), mask.to(device)).argmax(1).cpu() == y).sum().item(); total += len(y)
    return correct / total

torch.manual_seed(0)
bag = text.BagOfEmbeddings(len(vocab), dim=32, hidden=32)
h = train.fit_v2(bag, train_loader, val_loader, epochs=15, lr=3e-3, loss_fn=nn.CrossEntropyLoss())
print("bag: accuracy", accuracy(bag, val_loader), "| validation loss", round(h["val"][-1], 3), "| ln 2 =", round(math.log(2), 3))
'''),
        md("## 4. Attention by hand"),
        code('''
importlib.reload(attention)
torch.manual_seed(0)
q, k, v = torch.randn(2, 5, 8), torch.randn(2, 5, 8), torch.randn(2, 5, 8)
out, w = attention.scaled_dot_product(q, k, v)
print(out.shape, w.shape); print(w.sum(dim=-1))
print("matches PyTorch:", torch.allclose(out, F.scaled_dot_product_attention(q, k, v), atol=1e-5))
key_mask = torch.tensor([[True, True, True, False, False]])
_, w = attention.scaled_dot_product(q[:1], k[:1], v[:1], key_mask)
print("weights on the padded keys:", w[0, :, 3:].abs().max().item())
'''),
        md("## 5. Positions"),
        code('''
importlib.reload(attention)
pe = attention.PositionalEncoding(32, max_len=50)
print(pe.pe.shape, "| trainable parameters:", sum(p.numel() for p in pe.parameters()), "| in state_dict:", "pe" in pe.state_dict())
plt.figure(figsize=(8, 3)); plt.imshow(pe.pe[0].T, aspect="auto", cmap="RdBu"); plt.xlabel("position"); plt.ylabel("dimension"); plt.colorbar();
'''),
        md("## 6. The tiny Transformer"),
        code('''
importlib.reload(attention)
probe = attention.TinyTransformer(vocab_size=50, d_model=16, nhead=2, dropout=0.0).eval()
ids = torch.tensor([[5, 6, 7, 8]]); swapped = torch.tensor([[5, 8, 7, 6]]); padded = torch.tensor([[5, 6, 7, 8, 0, 0]])
print("order changes the output:", not torch.allclose(probe(ids, ids != 0), probe(swapped, swapped != 0)))
print("padding does not:", torch.allclose(probe(ids, ids != 0), probe(padded, padded != 0), atol=1e-5))

torch.manual_seed(0)
model = attention.TinyTransformer(len(vocab), d_model=32, nhead=2, num_layers=1, dim_feedforward=64)
print(count_parameters(model), "parameters")
h = train.fit_v2(model, train_loader, val_loader, epochs=25, lr=3e-3, loss_fn=nn.CrossEntropyLoss())
print("transformer: accuracy", accuracy(model, val_loader), "| validation loss by epoch:", [round(v, 2) for v in h["val"][::5]])
'''),
        md("### Look at the attention of one review"),
        code('''
model.eval()
ids, mask, y = next(iter(val_loader))
i = 0; toks = vocab.decode(ids[i][mask[i]]); n = len(toks)
with torch.no_grad():
    x = model.positions(model.embedding(ids[i:i+1, :n]))
    _, w = attention.scaled_dot_product(x, x, x)
plt.figure(figsize=(5, 4.5)); plt.imshow(w[0], cmap="Blues"); plt.xticks(range(n), toks, rotation=90); plt.yticks(range(n), toks); plt.title(f"label {y[i].item()}: {' '.join(toks)}", fontsize=9);
'''),
        md("### Break it on purpose: no positions"),
        code('''
class NoPositions(attention.TinyTransformer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.positions = nn.Identity()

torch.manual_seed(0)
blind = NoPositions(len(vocab), d_model=32, nhead=2, num_layers=1, dim_feedforward=64)
train.fit_v2(blind, train_loader, val_loader, epochs=25, lr=3e-3, loss_fn=nn.CrossEntropyLoss())
print("without positional encoding: accuracy", accuracy(blind, val_loader))
'''),
        md("## 7. Real reviews: bag against Transformer on Yelp\n\nReuses Lesson 11's download. Expect a modest gain, not a leap."),
        code('''
!pip install -q datasets
from datasets import load_dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stream = load_dataset("fancyzhx/yelp_polarity", split="train", streaming=True).shuffle(seed=0, buffer_size=10_000)
texts, labels = [], []
for ex in stream:
    texts.append(ex["text"]); labels.append(ex["label"])
    if len(texts) >= 40_000:
        break
n_val = 4_000
yelp_vocab = text.Vocab([text.tokenize(t) for t in texts[n_val:]], min_freq=2, max_size=20_000)
y_train, y_val = text.make_text_loaders(texts[n_val:], labels[n_val:], texts[:n_val], labels[:n_val], yelp_vocab, batch_size=128, max_len=200)
torch.manual_seed(0)
bag = text.BagOfEmbeddings(len(yelp_vocab), dim=64, hidden=64)
train.fit_v2(bag, y_train, y_val, epochs=5, lr=3e-3, loss_fn=nn.CrossEntropyLoss(), device=device)
torch.manual_seed(0)
tf = attention.TinyTransformer(len(yelp_vocab), d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
train.fit_v2(tf, y_train, y_val, epochs=5, lr=1e-3, loss_fn=nn.CrossEntropyLoss(), device=device)
print("Yelp validation accuracy: bag", round(accuracy(bag, y_val, device), 3), "| transformer", round(accuracy(tf, y_val, device), 3))
'''),
        md("## 8. Definition of done\n\nExpected: `7 passed`, then `90 passed` with no failures for everything so far (skipped tests belong to lessons you have not started)."),
        code("!pytest tests/test_12.py -q"),
        code("!pytest -m fast -q"),
        md("## 9. Commit and push"),
        code(PUSH.replace("{COMMIT_MESSAGE}", "Lesson 12: attention and a tiny Transformer")),
    ]
    return notebook(cells)


def lesson13() -> dict:
    lm_src = (ROOT / "starters/lesson13/forkcast/lm.py").read_text(encoding="utf-8")
    cells = [
        md("""
# Lesson 13: Pretrained language models

Goes with the Lesson 13 page of the Forkcast course. Subword tokenizers, a Hugging Face model as an `nn.Module`, gentle fine-tuning with clipping and accumulation, and sentence embeddings for menu search.

Runtime: pick a **GPU**. The toy model runs anywhere; the real DistilBERT fine-tune wants the GPU (about ten minutes on a T4).
"""),
        md("## 1. Clone your repository and install the package"),
        code(SETUP),
        md("## 2. `forkcast/lm.py`\n\nThe toy tokenizer and toy models are provided. Write `encode_batch`, `ClassificationCollator`, `mean_pool`, `embed_texts`, `search`, `clip_and_step` and `fit_lm` as the page builds them."),
        code("%%writefile forkcast/lm.py\n" + lm_src),
        md("## 3. Subword pieces, toy and real"),
        code('''
import importlib, math, torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from forkcast import lm, text
from forkcast.model import count_parameters
importlib.reload(lm)

frame = text.load_reviews()
toy_tok = lm.train_wordpiece_tokenizer(frame["text"], vocab_size=600)
real_tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
for name, tok in [("toy", toy_tok), ("real", real_tok)]:
    enc = lm.encode_batch(tok, ["Overpriced forkcast pizza, soggy fries!", "Cold."], max_length=32)
    print(name, enc["input_ids"].shape, tok.convert_ids_to_tokens(enc["input_ids"][0].tolist()))
    print("     mask of the short review:", enc["attention_mask"][1].tolist())
'''),
        md("## 4. A model is a module"),
        code('''
importlib.reload(lm)
collate = lm.ClassificationCollator(toy_tok, max_length=32)
cut = 640
train_loader = DataLoader(list(zip(frame["text"][:cut], frame["label"][:cut])), batch_size=32, shuffle=True, collate_fn=collate)
val_loader = DataLoader(list(zip(frame["text"][cut:], frame["label"][cut:])), batch_size=64, collate_fn=collate)
batch = next(iter(train_loader)); print({k: tuple(v.shape) for k, v in batch.items()})
torch.manual_seed(0)
tiny = lm.tiny_distilbert(toy_tok.vocab_size, num_labels=2)
out = tiny(**batch)
print(type(tiny).__name__, count_parameters(tiny), "parameters |", out.logits.shape, out.loss)
'''),
        md("## 5. Embeddings and search, toy encoder first"),
        code('''
importlib.reload(lm)
menu = ["spicy chicken ramen with soft egg", "margherita pizza with fresh basil", "beef pho with lime and herbs", "green curry with jasmine rice", "crispy fish tacos"]
enc = lm.tiny_encoder(toy_tok.vocab_size)
emb = lm.embed_texts(enc, toy_tok, menu)
print(emb.shape, emb.norm(dim=1))
print(lm.search("spicy chicken ramen with egg", menu, emb, enc, toy_tok, k=3))
'''),
        md("## 6. Fine-tune the toy model with clipping and accumulation"),
        code('''
importlib.reload(lm)
torch.manual_seed(0)
tiny = lm.tiny_distilbert(toy_tok.vocab_size, num_labels=2)
h = lm.fit_lm(tiny, train_loader, val_loader, epochs=3, lr=1e-3, accumulation_steps=2)
print("optimizer steps:", h["steps"], "| batches per epoch:", len(train_loader))
print("validation accuracy by epoch:", [round(a, 3) for a in h["val_acc"]])
'''),
        md("## 7. The real thing: DistilBERT on star ratings\n\n10,000 Yelp reviews with 1 to 5 stars, one epoch at 2e-5. About ten minutes on a T4."),
        code('''
!pip install -q datasets
from datasets import load_dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stream = load_dataset("fancyzhx/yelp_review_full", split="train", streaming=True).shuffle(seed=0, buffer_size=10_000)
texts, stars = [], []
for ex in stream:
    texts.append(ex["text"]); stars.append(ex["label"])          # 0..4 = one to five stars
    if len(texts) >= 10_000:
        break
real_collate = lm.ClassificationCollator(real_tok, max_length=128)
r_train = DataLoader(list(zip(texts[1000:], stars[1000:])), batch_size=16, shuffle=True, collate_fn=real_collate)
r_val = DataLoader(list(zip(texts[:1000], stars[:1000])), batch_size=64, collate_fn=real_collate)
torch.manual_seed(0)
bert = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)
print(count_parameters(bert), "parameters")
h = lm.fit_lm(bert, r_train, r_val, epochs=1, lr=2e-5, accumulation_steps=2, device=device)
print("star accuracy:", round(h["val_acc"][-1], 3), "(chance 0.20) | optimizer steps:", h["steps"])

bert.eval(); preds = []
with torch.no_grad():
    for b in r_val:
        preds.append(bert(**{k: v.to(device) for k, v in b.items()}).logits.argmax(-1).cpu())
preds = torch.cat(preds); truth = torch.tensor(stars[:1000])
print("mean absolute error in stars:", round((preds - truth).abs().float().mean().item(), 2))
'''),
        md("## 8. Search the menu by meaning\n\nA sentence-transformers encoder built for embeddings, against plain DistilBERT."),
        code('''
menu40 = ["spicy chicken ramen with soft egg", "tonkotsu ramen", "beef pho with lime and herbs", "green curry with jasmine rice", "pad thai with prawns", "margherita pizza with fresh basil", "four cheese lasagna", "spaghetti carbonara", "crispy fish tacos", "classic cheeseburger and fries", "falafel wrap with tahini", "greek salad with feta", "lentil soup", "grilled salmon with asparagus", "mushroom risotto", "chicken tikka masala", "lamb biryani", "vegetable dumplings", "kimchi fried rice", "bibimbap with tofu", "caesar salad", "tomato basil soup", "shrimp pad see ew", "quinoa bowl with roasted vegetables", "fish and chips", "pulled pork sandwich", "eggplant parmigiana", "miso soup with tofu", "chicken shawarma plate", "sushi platter", "poke bowl with tuna", "stuffed peppers", "pepperoni pizza", "beef tacos al pastor", "butter chicken", "udon noodle soup", "caprese sandwich", "mac and cheese", "coconut lentil dal", "chocolate lava cake"]
mini_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
mini = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
plain = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
emb_mini = lm.embed_texts(mini, mini_tok, menu40, device)
emb_plain = lm.embed_texts(plain, real_tok, menu40, device)
print(emb_mini.shape, emb_plain.shape)
for q in ["something spicy with noodles", "a light vegetarian lunch", "comfort food for a cold evening"]:
    print(f"\n{q!r}")
    print("  MiniLM:    ", [t for t, s in lm.search(q, menu40, emb_mini, mini, mini_tok, k=3, device=device)])
    print("  DistilBERT:", [t for t, s in lm.search(q, menu40, emb_plain, plain, real_tok, k=3, device=device)])
'''),
        md("### Break it on purpose: fine-tune at 1e-3"),
        code('''
torch.manual_seed(0)
reckless = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)
h = lm.fit_lm(reckless, r_train, r_val, epochs=1, lr=1e-3, accumulation_steps=2, device=device)
print("at lr=1e-3: star accuracy", round(h["val_acc"][-1], 3), "(2e-5 gave far more)")
'''),
        md("## 9. Definition of done\n\nExpected: `7 passed`, then `97 passed` with no failures for everything so far (skipped tests belong to lessons you have not started)."),
        code("!pytest tests/test_13.py -q"),
        code("!pytest -m fast -q"),
        md("## 10. Commit and push"),
        code(PUSH.replace("{COMMIT_MESSAGE}", "Lesson 13: pretrained language models")),
    ]
    return notebook(cells)


def lesson14() -> dict:
    rec_src = (ROOT / "starters/lesson14/forkcast/recsys.py").read_text(encoding="utf-8")
    cells = [
        md("""
# Lesson 14: Recommendations

Goes with the Lesson 14 page of the Forkcast course. User and restaurant embeddings, negative sampling on implicit feedback, hit rate at k against the popularity baseline, and a check against the hidden tastes that generated the orders.

Runtime: CPU is enough.
"""),
        md("## 1. Clone your repository and install the package"),
        code(SETUP),
        md("## 2. `forkcast/recsys.py`\n\nThe order generator, split, seen-items index and popularity baseline are provided. Write `MatrixFactorization`, `sample_negatives`, `train_one_epoch_rec`, `hit_rate_at_k` and `nearest_restaurants`."),
        code("%%writefile forkcast/recsys.py\n" + rec_src),
        md("## 3. The order history"),
        code('''
import importlib, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from forkcast import recsys
from forkcast.model import count_parameters
importlib.reload(recsys)

frame, user_factors, item_factors = recsys.make_order_history()
train, test = recsys.leave_one_out(frame)
seen = recsys.seen_items(train)
print(frame.head()); print(len(train), "training orders,", len(test), "held-out orders")
print("most ordered restaurants:", train["item"].value_counts().head(3).to_dict())
print("popularity hit@10:", round(recsys.popularity_hit_rate(train, test, seen, k=10), 3))
'''),
        md("## 4. The model"),
        code('''
importlib.reload(recsys)
torch.manual_seed(0)
model = recsys.MatrixFactorization(recsys.N_USERS, recsys.N_RESTAURANTS, dim=16)
print(count_parameters(model), "parameters")
users = torch.tensor([0, 1, 2]); items = torch.tensor([5, 6, 7])
print("forward agrees with score_all:", torch.allclose(model.score_all(users)[torch.arange(3), items], model(users, items), atol=1e-6))
print("untrained hit@10:", round(recsys.hit_rate_at_k(model, test, seen, 10), 3))
'''),
        md("## 5. Train with sampled negatives"),
        code('''
importlib.reload(recsys)
torch.manual_seed(0)
model = recsys.MatrixFactorization(recsys.N_USERS, recsys.N_RESTAURANTS, dim=16)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
loader = recsys.make_rec_loader(train)
losses, hits = [], []
for epoch in range(12):
    losses.append(recsys.train_one_epoch_rec(model, loader, optimizer, recsys.N_RESTAURANTS, seen))
    hits.append(recsys.hit_rate_at_k(model, test, seen, k=10))
print("loss:", [round(v, 3) for v in losses[::2]]); print("hit@10:", [round(v, 3) for v in hits[::2]])
fig, ax = plt.subplots(1, 2, figsize=(9, 3)); ax[0].plot(losses); ax[0].set_title("training loss"); ax[1].plot(hits); ax[1].axhline(recsys.popularity_hit_rate(train, test, seen, 10), ls="--", c="k"); ax[1].set_title("hit@10 vs popularity (dashed)");
'''),
        md("## 6. Hit rate against k"),
        code('''
ks = [1, 3, 5, 10, 20, 30]
learned = [recsys.hit_rate_at_k(model, test, seen, k) for k in ks]
popular = [recsys.popularity_hit_rate(train, test, seen, k) for k in ks]
print({k: (round(l, 3), round(p, 3)) for k, l, p in zip(ks, learned, popular)})
plt.plot(ks, learned, marker="o", label="learned"); plt.plot(ks, popular, marker="o", label="popularity"); plt.plot(ks, [k / 100 for k in ks], "k:", label="random"); plt.xlabel("k"); plt.ylabel("hit rate"); plt.legend();
'''),
        md("## 7. What it learned, against the hidden tastes"),
        code('''
importlib.reload(recsys)
learned_vec = F.normalize(model.item_vec.weight.detach(), dim=1); truth = F.normalize(item_factors, dim=1)
corr = torch.corrcoef(torch.stack([(learned_vec @ learned_vec.T).flatten(), (truth @ truth.T).flatten()]))[0, 1]
print("correlation of learned and hidden restaurant similarities:", round(corr.item(), 3))
print("nearest to restaurant 7 (learned):", recsys.nearest_restaurants(model, 7, k=5))
print("nearest to restaurant 7 (hidden): ", [int(i) for i in (truth @ truth[7]).topk(6).indices.tolist() if i != 7][:5])

for u in [0, 1, 2]:
    scores = model.score_all(torch.tensor([u]))
    scores[0, list(seen[u])] = float("-inf")
    held = int(test.loc[test["user"] == u, "item"].iloc[0])
    top = scores.topk(10).indices[0].tolist()
    print(f"user {u}: top-5 {top[:5]} | held-out {held} | rank {top.index(held) + 1 if held in top else '>10'}")
'''),
        md("### Break it on purpose: forget to hide the seen restaurants"),
        code('''
@torch.no_grad()
def hit_rate_no_mask(model, test, k=10):
    users = torch.tensor(test["user"].values); targets = torch.tensor(test["item"].values)
    top = model.score_all(users).topk(k, dim=1).indices
    return (top == targets[:, None]).any(dim=1).float().mean().item()
print("masked:", round(recsys.hit_rate_at_k(model, test, seen, 10), 3), "| unmasked:", round(hit_rate_no_mask(model, test), 3))
'''),
        md("## 8. Definition of done\n\nExpected: `7 passed`, then `104 passed` with no failures for everything so far (skipped tests belong to lessons you have not started)."),
        code("!pytest tests/test_14.py -q"),
        code("!pytest -m fast -q"),
        md("## 9. Commit and push"),
        code(PUSH.replace("{COMMIT_MESSAGE}", "Lesson 14: recommendations")),
    ]
    return notebook(cells)


def lesson15() -> dict:
    serve_src = (ROOT / "starters/lesson15/forkcast/serve.py").read_text(encoding="utf-8")
    cells = [
        md("""
# Lesson 15: Serving

Goes with the Lesson 15 page of the Forkcast course. An artifact that holds everything inference needs, an inference path that matches training, `torch.export`, FastAPI endpoints tested in-process, latency measurement, and the real server.

Runtime: CPU is enough.
"""),
        md("## 1. Clone your repository and install the package"),
        code(SETUP.rstrip("\n") + "\n!pip install -q fastapi httpx uvicorn\n"),
        md("## 2. `forkcast/serve.py`\n\nThe pydantic request models are provided. Write `EtaPredictor.features/predict/save/load`, `export_program`, `load_exported`, `latency_stats` and the three endpoint bodies as the page builds them."),
        code("%%writefile forkcast/serve.py\n" + serve_src),
        md("## 3. A trained ETA model, and the inference path"),
        code('''
import importlib, os, torch
from torch import nn
from fastapi.testclient import TestClient
from forkcast import serve, trips, train, text, vision
from forkcast.model import TabularNet
from forkcast.preprocess import Normalizer
importlib.reload(serve)

frame = trips.load_trips(verbose=False)
tr, va, _ = trips.split_frame(frame, seed=0)
norm = Normalizer().fit(torch.tensor(tr[trips.CONTINUOUS].values, dtype=torch.float32))
tl, vl = trips.make_loaders(tr, va, norm, batch_size=256)
torch.manual_seed(0)
model = TabularNet(4, hidden=(32, 32))
train.fit_loader(model, tl, vl, epochs=5, lr=3e-3)
eta = serve.EtaPredictor(model, norm, hidden=(32, 32))
orders = tr.head(8)[trips.CONTINUOUS + trips.CATEGORICAL].astype({"pickup_hour": int, "weekday": int, "passenger_count": int}).to_dict(orient="records")
print(orders[0])
print("batched:", [round(v, 2) for v in eta.predict(orders[:3])], "| single:", [round(eta.predict([o])[0], 2) for o in orders[:3]])
'''),
        md("## 4. The artifact"),
        code('''
importlib.reload(serve)
os.makedirs("artifacts", exist_ok=True)
eta = serve.EtaPredictor(model, norm, hidden=(32, 32))
eta.save("artifacts/eta.pt")
print("artifact:", round(os.path.getsize("artifacts/eta.pt") / 1024, 1), "kB; keys", sorted(torch.load("artifacts/eta.pt", weights_only=True)))
loaded = serve.EtaPredictor.load("artifacts/eta.pt")
print("same answers after reload:", torch.allclose(torch.tensor(loaded.predict(orders)), torch.tensor(eta.predict(orders)), atol=1e-5))
'''),
        md("## 5. Export"),
        code('''
importlib.reload(serve)
cont, cat = eta.features(orders)
serve.export_program(eta.model, (cont[:4], cat[:4]), "artifacts/eta.pt2")
exported = serve.load_exported("artifacts/eta.pt2")
with torch.inference_mode():
    for n in (1, 3, 8):
        print(f"batch {n}: max difference", (exported(cont[:n], cat[:n]) - eta.model(cont[:n], cat[:n])).abs().max().item())
'''),
        md("## 6. The API, tested in-process\n\nA sentiment model from Lesson 11 and a shape classifier from Lesson 7 stand in for the real dish model; the capstone swaps in Lesson 9's ResNet."),
        code('''
importlib.reload(serve)
reviews = text.load_reviews()
vocab = text.Vocab([text.tokenize(t) for t in reviews["text"]])
r_tl, r_vl = text.make_text_loaders(reviews["text"][:640], reviews["label"][:640], reviews["text"][640:], reviews["label"][640:], vocab, batch_size=32)
torch.manual_seed(0); bag = text.BagOfEmbeddings(len(vocab), dim=32, hidden=32)
train.fit_v2(bag, r_tl, r_vl, epochs=6, lr=3e-3, loss_fn=nn.CrossEntropyLoss())
s_tl, s_vl = vision.shape_loaders(640, 128)
torch.manual_seed(0); cnn = vision.SmallCNN()
train.fit_v2(cnn, s_tl, s_vl, epochs=3, lr=1e-3, loss_fn=nn.CrossEntropyLoss())
torch.save({"state_dict": bag.state_dict(), "itos": vocab.itos, "dim": 32, "hidden": 32}, "artifacts/sentiment.pt")
torch.save({"state_dict": cnn.state_dict(), "classes": vision.SHAPE_CLASSES, "size": 32}, "artifacts/dish.pt")

client = TestClient(serve.create_app(eta, sentiment=(bag, vocab), dish=(cnn, vision.SHAPE_CLASSES)))
print(client.get("/health").json())
print(client.post("/eta", json={"orders": orders[:2]}).json())
bad = dict(orders[0]); del bad["trip_distance"]
r = client.post("/eta", json={"orders": [bad]}); print(r.status_code, r.json()["detail"][0])
print(client.post("/sentiment", json={"texts": ["Delicious, will order again.", "Cold, soggy and an hour late."]}).json())
images, labels = vision.make_shapes(2, seed=3)
print(client.post("/dish", json={"pixels": images[0].tolist()}).json(), "| true:", vision.SHAPE_CLASSES[labels[0]])
r = client.post("/dish", json={"pixels": images[0].permute(1, 2, 0).tolist()}); print(r.status_code, r.json())
'''),
        md("## 7. Latency"),
        code('''
importlib.reload(serve)
single = serve.latency_stats(lambda o: eta.predict([o]), orders, repeats=5)
batched = serve.latency_stats(lambda os: eta.predict(os), [orders * 8], repeats=5)
http = serve.latency_stats(lambda o: client.post("/eta", json={"orders": [o]}), orders, repeats=3)
print(f"one order per call: p50 {single['p50_ms']:.2f} ms, p95 {single['p95_ms']:.2f} ms")
print(f"64 orders per call: {batched['mean_ms']:.2f} ms total, {batched['mean_ms'] / 64:.3f} ms per order")
print(f"HTTP round trip, one order: p50 {http['p50_ms']:.2f} ms")
'''),
        md("### Break it on purpose: a fresh normalizer"),
        code('''
fresh = Normalizer().fit(torch.tensor(__import__("pandas").DataFrame(orders)[trips.CONTINUOUS].values, dtype=torch.float32))
wrong = serve.EtaPredictor(model, fresh, hidden=(32, 32))
print("trained statistics:", eta.normalizer.mean.tolist()); print("fresh statistics:  ", fresh.mean.tolist())
print("right:", [round(v, 1) for v in eta.predict(orders[:4])]); print("wrong:", [round(v, 1) for v in wrong.predict(orders[:4])], "<- status 200, no error, minutes off")
'''),
        md("## 8. The real server\n\n`deploy/app.py` loads the artifacts you just wrote. Visit the printed URL + `/docs` if your environment exposes the port."),
        code('''
import subprocess, time, requests
server = subprocess.Popen(["uvicorn", "deploy.app:app", "--host", "127.0.0.1", "--port", "8000"], stdout=open("server.log", "w"), stderr=subprocess.STDOUT)
time.sleep(4)
print(requests.get("http://127.0.0.1:8000/health").json())
print(requests.post("http://127.0.0.1:8000/eta", json={"orders": orders[:2]}).json())
print(requests.post("http://127.0.0.1:8000/sentiment", json={"texts": ["Never again."]}).json())
server.terminate()
'''),
        md("## 9. Definition of done\n\nExpected: `7 passed`, then `111 passed` with no failures for everything so far (skipped tests belong to lessons you have not started)."),
        code("!pytest tests/test_15.py -q"),
        code("!pytest -m fast -q"),
        md("## 10. Commit and push"),
        code(PUSH.replace("{COMMIT_MESSAGE}", "Lesson 15: serving")),
    ]
    return notebook(cells)


def lesson16() -> dict:
    cap_src = (ROOT / "starters/lesson16/forkcast/capstone.py").read_text(encoding="utf-8")
    cells = [
        md("""
# Lesson 16: Capstone, multimodal Forkcast

Goes with the Lesson 16 page of the Forkcast course. One model for a listing's photo, description and numbers; an ablation over modalities; a model card; the artifact; and the whole test suite green.

Runtime: CPU is enough (about half a minute for the full ablation).
"""),
        md("## 1. Clone your repository and install the package"),
        code(SETUP),
        md("## 2. `forkcast/capstone.py`\n\nThe data generator, dataset and loaders are provided. Write `collate_listings`, `FusionModel`, `run_ablation` and `write_model_card` as the page builds them."),
        code("%%writefile forkcast/capstone.py\n" + cap_src),
        md("## 3. The listings"),
        code('''
import importlib, itertools, torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from forkcast import capstone as cap, train, debug
from forkcast.model import count_parameters
importlib.reload(cap)

images, texts, tabular, labels = cap.make_listings(1600)
print(images.shape, tabular.shape, "label balance", labels.float().mean().item())
plt.figure(figsize=(12, 2.4)); plt.axis("off"); plt.imshow(make_grid(images[:8], nrow=8).permute(1, 2, 0))
for t, tab, l in zip(texts[:8], tabular[:8], labels[:8]):
    print(f"{l.item()}  {t:45s} price {tab[0]:5.1f}  prep {tab[1]:4.0f}  rating {tab[2]:.1f}")
train_loader, val_loader, vocab, (tab_mean, tab_std) = cap.listing_loaders()
images_b, ids, mask, tabular_b, labels_b = next(iter(train_loader))
print(images_b.shape, ids.shape, mask.shape, tabular_b.shape, labels_b.shape)
'''),
        md("## 4. The fusion model"),
        code('''
importlib.reload(cap)
model = cap.FusionModel(len(vocab))
print(model(images_b, ids, mask, tabular_b).shape, "|", count_parameters(model), "parameters")
tab_only = cap.FusionModel(len(vocab), modalities=("tabular",))
print("tabular-only ignores the photo:", torch.allclose(tab_only(images_b, ids, mask, tabular_b), tab_only(torch.rand_like(images_b), ids, mask, tabular_b)))
'''),
        md("## 5. The ablation: all seven combinations"),
        code('''
importlib.reload(cap)
combos = [c for r in range(1, 4) for c in itertools.combinations(cap.MODALITIES, r)]
results = cap.run_ablation(lambda m: cap.FusionModel(len(vocab), modalities=m), train_loader, val_loader, combos, epochs=12, lr=2e-3)
for k, v in results.items():
    print(f"{' + '.join(k):26s} {v:.3f}")
plt.figure(figsize=(8, 3)); plt.bar([" + ".join(k) for k in results], list(results.values())); plt.xticks(rotation=30, ha="right"); plt.axhline(0.5, ls="--", c="k"); plt.ylabel("validation accuracy");
'''),
        md("## 6. Train the fused model and evaluate it the Lesson 10 way"),
        code('''
torch.manual_seed(0)
fused = cap.FusionModel(len(vocab))
h = train.fit_v2(fused, train_loader, val_loader, epochs=12, lr=2e-3, loss_fn=nn.CrossEntropyLoss())
print("fused validation accuracy:", round(cap.accuracy_fusion(fused, val_loader), 3))
fused.eval(); logits_all, y_all, imgs_all = [], [], []
with torch.no_grad():
    for im, i_, mk, tb, y in val_loader:
        logits_all.append(fused(im, i_, mk, tb)); y_all.append(y); imgs_all.append(im)
logits_all, y_all, imgs_all = torch.cat(logits_all), torch.cat(y_all), torch.cat(imgs_all)
probs = torch.softmax(logits_all, 1)
print("ECE:", round(debug.expected_calibration_error(probs, y_all), 3))
picks = debug.misclassified(logits_all, y_all, k=8)
print("most confident mistakes:", [(i, t, p, round(c, 2)) for i, t, p, c in picks[:4]])
plt.figure(figsize=(12, 2.4)); plt.axis("off"); plt.imshow(debug.gallery(imgs_all, picks).permute(1, 2, 0))
'''),
        md("## 7. The model card"),
        code('''
importlib.reload(cap)
card = cap.write_model_card("MODEL_CARD.md", "Forkcast listing demand v1",
    intended_use="Rank new listings for review priority. Not for pricing decisions.",
    data="1,600 synthetic listings: Lesson 7 shapes as photos, templated descriptions, simulated prices, prep times and ratings.",
    metrics={"accuracy (fused)": results[("image", "text", "tabular")], "accuracy (best single modality)": max(v for k, v in results.items() if len(k) == 1), "epochs": 12},
    limitations=["Synthetic data only; real photos and reviews will shift the distribution.", "32x32 images; a real dish model would use Lesson 9's pretrained backbone.", "The label is 'above median demand', not observed orders."],
    lesson_trail=["Lesson 7: image encoder (two conv blocks)", "Lesson 11: text encoder (embedding, masked mean)", "Lesson 4: tabular encoder (MLP)", "Lesson 8: training loop v2", "Lesson 15: artifact format"])
print(card)
'''),
        md("## 8. Ship it as an artifact"),
        code('''
import os; os.makedirs("artifacts", exist_ok=True)
torch.save({"state_dict": fused.state_dict(), "itos": vocab.itos, "tab_mean": tab_mean, "tab_std": tab_std, "d": 32, "hidden": 64, "size": 32, "modalities": list(fused.modalities)}, "artifacts/listing.pt")
bundle = torch.load("artifacts/listing.pt", weights_only=True)
fresh = cap.FusionModel(len(bundle["itos"]), d=bundle["d"], hidden=bundle["hidden"], size=bundle["size"], modalities=tuple(bundle["modalities"]))
fresh.load_state_dict(bundle["state_dict"]); fresh.eval()
with torch.inference_mode():
    print("identical predictions after reload:", torch.equal(fresh(images_b, ids, mask, tabular_b), fused(images_b, ids, mask, tabular_b)))
'''),
        md("### Break it on purpose: raw tabular inputs"),
        code('''
raw_mean, raw_std = torch.zeros(3), torch.ones(3)        # no standardization
imgs_, texts_, tab_, labels_ = cap.make_listings(1600)
raw_train = torch.utils.data.DataLoader(cap.ListingsDataset(imgs_[:1200], texts_[:1200], tab_[:1200], labels_[:1200], vocab, raw_mean, raw_std), batch_size=64, shuffle=True, collate_fn=cap.collate_listings)
raw_val = torch.utils.data.DataLoader(cap.ListingsDataset(imgs_[1200:], texts_[1200:], tab_[1200:], labels_[1200:], vocab, raw_mean, raw_std), batch_size=64, collate_fn=cap.collate_listings)
torch.manual_seed(0); raw_model = cap.FusionModel(len(vocab))
train.fit_v2(raw_model, raw_train, raw_val, epochs=12, lr=2e-3, loss_fn=nn.CrossEntropyLoss())
print("fused, raw tabular inputs:", round(cap.accuracy_fusion(raw_model, raw_val), 3), "| standardized:", round(results[("image", "text", "tabular")], 3))
'''),
        md("## 9. Definition of done\n\nExpected: `7 passed`, then `118 passed` for the whole course (the only skips are the GPU test and the optional Lesson 17)."),
        code("!pytest tests/test_16.py -q"),
        code("!pytest -m fast -q"),
        md("## 10. Commit and push"),
        code(PUSH.replace("{COMMIT_MESSAGE}", "Lesson 16: capstone, multimodal Forkcast")),
    ]
    return notebook(cells)


def lesson17() -> dict:
    exp_src = (ROOT / "starters/lesson17/forkcast/experiments.py").read_text(encoding="utf-8")
    cells = [
        md("""
# Lesson 17: Experiment hygiene (optional)

Goes with the Lesson 17 page of the Forkcast course. Seeds, a frozen config with a fingerprint, reproducible logged runs, grouping the log by configuration, and seed noise against a real difference.

Runtime: CPU is enough; every run takes well under a second.
"""),
        md("## 1. Clone your repository and install the package"),
        code(SETUP),
        md("## 2. `forkcast/experiments.py`\n\n`git_commit` and `load_log` are provided. Write `seed_everything`, the `RunConfig` methods, `run_experiment`, `compare` and `repeat_with_seeds`."),
        code("%%writefile forkcast/experiments.py\n" + exp_src),
        md("## 3. Seeds"),
        code('''
import importlib, random, torch
from dataclasses import replace
from torch.utils.data import DataLoader, TensorDataset
from forkcast import experiments as exp
from forkcast.model import MLP
importlib.reload(exp)

g1 = exp.seed_everything(123); a = MLP(4, (8,)); order1 = [b[0].tolist() for b in DataLoader(TensorDataset(torch.arange(16)), batch_size=4, shuffle=True, generator=g1)]
g2 = exp.seed_everything(123); b = MLP(4, (8,)); order2 = [b_[0].tolist() for b_ in DataLoader(TensorDataset(torch.arange(16)), batch_size=4, shuffle=True, generator=g2)]
print("same weights:", all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters())), "| same shuffle:", order1 == order2)
print("deterministic algorithms:", torch.are_deterministic_algorithms_enabled(), "| cudnn.benchmark:", torch.backends.cudnn.benchmark)
'''),
        md("## 4. Configs and fingerprints"),
        code('''
importlib.reload(exp)
cfg = exp.RunConfig(hidden=(16, 16), lr=0.005, seed=3)
cfg.to_json("run.json")
print("round trip equal:", exp.RunConfig.from_json("run.json") == cfg)
print("fingerprints:", cfg.fingerprint(), replace(cfg, lr=0.01).fingerprint(), replace(cfg, seed=4).fingerprint())
'''),
        md("## 5. Reproducible, logged runs"),
        code('''
importlib.reload(exp)
import os
if os.path.exists("experiments.jsonl"):
    os.remove("experiments.jsonl")
a = exp.run_experiment(exp.RunConfig(), "experiments.jsonl")
b = exp.run_experiment(exp.RunConfig(), "experiments.jsonl")
print("same digits:", a["best_val"] == b["best_val"], "|", round(a["best_val"], 4), "in", round(a["seconds"], 2), "s | commit", a["commit"])
print(open("experiments.jsonl").readline()[:240])
'''),
        md("## 6. Three experiments, five seeds each"),
        code('''
importlib.reload(exp)
import math
seeds = [0, 1, 2, 3, 4]
small = exp.repeat_with_seeds(exp.RunConfig(), seeds, "experiments.jsonl")
big = exp.repeat_with_seeds(replace(exp.RunConfig(), hidden=(64, 64)), seeds, "experiments.jsonl")
wd = exp.repeat_with_seeds(replace(exp.RunConfig(), weight_decay=1e-3), seeds, "experiments.jsonl")
for name, r in [("(32, 32)", small), ("(64, 64)", big), ("(32, 32) + weight decay", wd)]:
    print(f"{name:26s} {[round(v, 3) for v in r['values']]}  mean {r['mean']:.3f}  std {r['std']:.3f}")
print(exp.compare(exp.load_log("experiments.jsonl")).to_string())
print("spread of a 5-seed mean, small net:", round(small["std"] * math.sqrt(2 / 5), 3), "| (64,64) better by", round(small["mean"] - big["mean"], 3), "| weight decay better by", round(small["mean"] - wd["mean"], 3))
'''),
        md("### Break it on purpose: a stray random call"),
        code('''
def leaky_run(config):
    idx = list(range(10)); random.shuffle(idx)          # unseeded randomness BEFORE seed_everything
    return exp.run_experiment(config)["best_val"], idx[:3]
r1 = leaky_run(exp.RunConfig()); r2 = leaky_run(exp.RunConfig())
print("the run itself still matches:", r1[0] == r2[0], "| but the pre-seed shuffle does not:", r1[1], r2[1])
print("anything downstream of that shuffle would drift between runs; move it after seed_everything and it is pinned")
'''),
        md("## 7. Definition of done\n\nExpected: `7 passed`, then `125 passed` for the whole course (the only skip is the GPU test)."),
        code("!pytest tests/test_17.py -q"),
        code("!pytest -m fast -q"),
        md("## 8. Commit and push"),
        code(PUSH.replace("{COMMIT_MESSAGE}", "Lesson 17: experiment hygiene")),
    ]
    return notebook(cells)


LESSONS = {"lesson00.ipynb": lesson00, "lesson01.ipynb": lesson01, "lesson02.ipynb": lesson02, "lesson03.ipynb": lesson03, "lesson04.ipynb": lesson04, "lesson05.ipynb": lesson05, "lesson06.ipynb": lesson06, "lesson07.ipynb": lesson07, "lesson08.ipynb": lesson08, "lesson09.ipynb": lesson09, "lesson10.ipynb": lesson10, "lesson11.ipynb": lesson11, "lesson12.ipynb": lesson12, "lesson13.ipynb": lesson13, "lesson14.ipynb": lesson14, "lesson15.ipynb": lesson15, "lesson16.ipynb": lesson16, "lesson17.ipynb": lesson17}


def main() -> None:
    out_dir = ROOT / "notebooks"
    out_dir.mkdir(exist_ok=True)
    for name, build in LESSONS.items():
        path = out_dir / name
        path.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
