# Roadmap: Lessons 6 to 10

All 18 lessons (0 to 17) are built; Lessons 6 and 7 followed the specs below (Lesson 7 uses synthetic shapes for the
offline tests and streams ten Food-101 classes from the Hugging Face Hub in the notebook instead of the full
torchvision download). This file fixes the design of the next five so each can be
built to the same pattern: one `site/lessons/NN-*.html` fragment with 2–3 widgets and
6 questions, starter + reference solution, `tests/test_NN.py` (fast, CPU, deterministic),
a generated notebook, a "break it on purpose" step, and exact DoD output quoted on the page.
DoD counts below assume 7 tests per lesson; adjust when the tests are written.

## Lesson 6: Dataset, DataLoader and categorical embeddings  (end of Part 1)

- **Data.** First real data: NYC TLC yellow-taxi trip records (public parquet, one month,
  sampled to ~200k rows) as a stand-in for delivery ETAs. `tools/get_taxi.py` downloads and
  cleans in Colab; the repo bundles `data/taxi_sample.parquet` (~5k rows) so the DoD tests
  and CI run offline. Features: trip distance, pickup hour, weekday, passenger count,
  pickup zone id (categorical, ~260 values), dropoff zone id. Target: trip minutes.
  Cleaning rules taught explicitly: drop zero-distance trips, cap at 120 minutes, log the
  rows removed.
- **Objectives.** Wrap tensors in a `Dataset`; iterate mini-batches with a `DataLoader`
  (`batch_size`, `shuffle`, `drop_last`, `num_workers`, `collate_fn`); explain why
  mini-batches change the loss curve; replace one-hot zones with `nn.Embedding`; adapt
  the toolkit to loaders (`train_one_epoch(model, loader, ...)`, training loop v1.5).
- **PyTorch introduced.** `torch.utils.data.Dataset`, `DataLoader`, `TensorDataset`,
  `random_split`, `nn.Embedding`, `torch.cat` of continuous + embedded features, `.long()`
  for index tensors, `pandas` → tensors.
- **Exercises.** `forkcast/data.py`: `TripsDataset(frame)` returning
  `(continuous (d,), categorical (2,) long, target ())`; `forkcast/model.py`:
  `TabularNet(n_continuous, zone_vocab, emb_dim=8, hidden=(64, 64))`;
  `forkcast/train.py`: loader-based `train_one_epoch` / `evaluate` (keep the tensor
  versions working: dispatch on the argument type or add `*_loader` variants).
- **Tests (7).** Dataset length and item shapes/dtypes; loader batch shapes with
  `drop_last`; shuffle changes order but not the multiset; embedding table shape
  `(vocab, emb_dim)` and output `(N, emb_dim)`; `TabularNet` forward shape on a batch;
  one loader epoch lowers validation loss on the bundled sample; embeddings beat a
  no-zone baseline on the bundled sample by a margin (train 30 epochs, fast on CPU).
- **Widgets.** DataLoader animation (indices → shuffle → batches → collate, with
  `drop_last` toggle); mini-batch vs full-batch loss curves on the Lesson 5 problem
  (noisier but faster per epoch); embedding lookup: click a zone id, watch the row of the
  table come out, and a 2-D projection of learned zone vectors (precomputed).
- **Break it.** Forget `.long()` on zone ids → `IndexError`/dtype error; set
  `shuffle=False` on sorted-by-time data → validation loss stuck.
- **Risk.** Parquet download in Colab needs `pyarrow` (present) and the TLC URL; the
  download step is notebook-only and verified by shape, never by CI.

## Lesson 7: Images and the first CNN  (Part 2 starts; GPU runtime from here)

- **Data.** Food-101 via `torchvision.datasets.Food101`, restricted to 10 classes and
  resized to 64×64 for this lesson (`forkcast/vision/data.py: food10_subset`). The repo
  bundles 200 tiny 32×32 images (`data/food_mini/`) for offline tests.
- **Objectives.** Images as `(N, C, H, W)` tensors; normalization per channel;
  convolution as a sliding window; padding/stride/kernel shape arithmetic; pooling;
  a small CNN; `CrossEntropyLoss` and accuracy (first classification problem).
- **PyTorch introduced.** `torchvision.datasets`, `transforms.v2` (`ToImage`,
  `ToDtype`, `Resize`, `Normalize`), `nn.Conv2d`, `nn.MaxPool2d`, `nn.BatchNorm2d`,
  `nn.Flatten`, `nn.CrossEntropyLoss`, `argmax`, `torchvision.utils.make_grid`.
- **Exercises.** `conv_output_size(h, k, s, p)`; `SmallCNN(n_classes)`; `accuracy(logits, y)`;
  toolkit gains `loss_fn` plumbing for classification and an `accuracy` metric in `evaluate`.
- **Tests (7).** Shape formula against real `nn.Conv2d` outputs for 6 configurations;
  CNN forward shape `(N, 10)`; parameter count; `accuracy` on a hand-made batch;
  one epoch on the bundled mini set lowers loss; a 32×32 checkpoint reaches >60% on the
  bundled mini validation split within 20 epochs (CPU, ~20 s); `CrossEntropyLoss` on
  logits vs manual log-softmax formula.
- **Widgets.** Kernel sliding over a pixel grid with live output and shape formula
  (kernel/stride/padding sliders); pooling demo; channel stacking (RGB planes → a conv
  filter sees all three); softmax/cross-entropy visualiser (logits sliders → probabilities →
  loss for the true class).
- **Break it.** Feed `(N, H, W, C)` into a conv → channel error; forget `Flatten` before the
  linear head → mat shape error; use `MSELoss` for classes → it trains but accuracy stalls.

## Lesson 8: GPU training for real  (training loop v2)

- **Objectives.** Device-agnostic code (`device` plumbing, `.to(device)` for model and
  batches, `non_blocking`); mixed precision with `torch.autocast` + `GradScaler`; data
  augmentation (`RandomResizedCrop`, `RandomHorizontalFlip`, `ColorJitter`) and why it is
  regularization; checkpoint dicts (model + optimizer + epoch + best metric) with
  resume; logging to TensorBoard; `torch.compile` as a one-line speedup; timing an epoch.
- **PyTorch introduced.** `torch.cuda.is_available/get_device_name/memory_allocated`,
  `torch.autocast`, `torch.amp.GradScaler`, `pin_memory`, `torch.save` of dicts,
  `torch.utils.tensorboard.SummaryWriter`, `torch.compile`, `torch.backends.cudnn.benchmark`.
- **Exercises.** `forkcast/train.py` v2: `Trainer` class or `fit_loader(model, loaders,
  device, epochs, amp=True, ckpt_path=...)` that serves both tabular (Lesson 6) and image
  (Lesson 7) models; `save_checkpoint` / `load_checkpoint` with optimizer state and the
  epoch counter.
- **Tests (7).** Resumed training continues from the identical loss (save after 3 epochs,
  resume, compare with an uninterrupted run); the same `fit_loader` passes on `TabularNet`
  and `SmallCNN`; `autocast` path runs on CPU with bfloat16 without error; augmentation
  changes a batch while leaving shapes intact; checkpoint dict has the five required keys;
  `GradScaler` is skipped on CPU; TensorBoard log directory gets an events file.
- **Widgets.** GPU vs CPU timing bar for one epoch (recorded numbers, interactive batch
  size); augmentation previewer (one dish photo, re-roll the random transforms);
  checkpoint/resume timeline; the training loop v2 as a flow diagram with the device moves
  highlighted.
- **Break it.** Model on GPU, batch on CPU → device error; resume without optimizer state →
  loss jumps.

## Lesson 9: Transfer learning and evaluation

- **Data.** Food-101, all 101 classes, 224×224, GPU runtime, ~25 minutes of training
  with a frozen backbone; bundled mini set for tests.
- **Objectives.** Load a pretrained backbone from `torchvision.models` (ResNet-18 /
  EfficientNet-B0 with weight enums); freeze it; replace the head; fine-tune the last block
  with a smaller learning rate via optimizer parameter groups; evaluate with a confusion
  matrix, top-k accuracy and per-class error lists.
- **PyTorch introduced.** `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)`,
  `weights.transforms()`, `requires_grad_(False)`, `model.fc` replacement, optimizer param
  groups, `torch.topk`, `torch.inference_mode`, `torchmetrics` or a hand-rolled confusion
  matrix.
- **Exercises.** `build_transfer_model(n_classes, trainable_blocks=0)`; `freeze(module)`;
  `top_k_accuracy(logits, y, k)`; `confusion_matrix(preds, y, n_classes)`.
- **Tests (7).** No backbone parameter requires grad after `freeze`; the head has
  `n_classes` outputs; param groups carry two learning rates; `top_k_accuracy` on a
  hand-made batch; confusion matrix rows sum to class counts; a frozen-backbone model
  trains on the bundled mini set (CPU, random-weights fallback when the pretrained download
  is unavailable, so CI stays offline); fine-tuning changes only the unfrozen parameters.
- **Widgets.** Feature hierarchy (early layers edges → textures → parts → dishes) with a
  frozen/trainable slider; confusion-matrix explorer (click a cell to see the confused pair);
  top-k accuracy animation.
- **Break it.** Forget `model.eval()` before evaluation → BatchNorm statistics drift;
  fine-tune everything at `lr=0.01` → the pretrained features get wiped and accuracy drops.

## Lesson 10 (optional): Debugging and interpreting models

- **Objectives.** Error analysis workflow: find the worst-confused classes, build a gallery
  of mistakes, inspect class balance; Grad-CAM with forward hooks; calibration (reliability
  diagram, expected calibration error) and why confident-and-wrong matters for Forkcast.
- **PyTorch introduced.** `register_forward_hook`, `register_full_backward_hook`,
  `torch.bincount` for class counts, `torch.softmax` confidence, `torch.quantile`.
- **Exercises.** `misclassified(model, loader, k)`; `GradCAM(model, layer)`;
  `expected_calibration_error(probs, y, bins)`; `class_weights(counts)` for `CrossEntropyLoss`.
- **Tests (7).** Hook fires once per forward and captures the right shape; Grad-CAM map has
  the spatial shape of the target layer; ECE is 0 for a perfectly calibrated toy and large
  for an overconfident one; `class_weights` inversely proportional to counts; `misclassified`
  returns only wrong predictions sorted by confidence; a weighted loss changes per-class
  gradients; the gallery helper returns a `make_grid` tensor of the expected shape.
- **Widgets.** Reliability diagram with adjustable overconfidence; Grad-CAM overlay toggles
  on a few bundled images (precomputed maps); class-balance bars with the weight formula.
- **Break it.** Register a hook and forget to remove it → every later forward is slowed and
  memory grows; compute ECE on logits instead of probabilities.

## Cross-cutting decisions

- Tests never download. Everything CI needs is bundled under `data/` and kept small
  (< 5 MB total); real datasets are fetched in notebooks with a visible, checked download step.
- GPU-only behaviours are tested through CPU fallbacks (`bfloat16` autocast, `GradScaler`
  disabled) and marked so that a GPU runtime exercises the real path.
- Each lesson's page quotes DoD output measured in the sandbox; the Colab notebook has an
  identical final cell so the two never disagree.
