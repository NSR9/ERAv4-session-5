

# Deep Dive: Network Architecture & Training Loop (with Optimizers)

This section explains how the CNN is put together, how the training loop works under the hood, and why we use specific optimizers/schedulers—tailored to an MNIST-style setup.

---

## 1. Network Architecture (Conceptual → Practical)

### 1.1 Input & Preprocessing
- **Input shape**: `N × 1 × 28 × 28` (batch, channels, height, width)
- **Transforms**: random crop/rotate (train), tensor conversion, and normalization with MNIST mean/std. Normalization helps gradients and BatchNorm behave more predictably.

### 1.2 Convolutional Blocks (feature extractors)
Typical block: Conv2d → ReLU → BatchNorm2d → Dropout → (optional MaxPool2d)

- **Conv2d**: learns spatial filters (edges, strokes, corners)
- **ReLU**: introduces non-linearity
- **BatchNorm2d**: stabilizes activations across the batch → faster, more stable training
- **Dropout** (≈0.05–0.2 for MNIST): combats overfitting by randomly zeroing features
- **MaxPool2d**: halves H/W (e.g., 28→14→7), reduces compute and adds translational invariance

> Tip: With BatchNorm, you can often set `bias=False` in the preceding conv (BN provides affine shift/scale).

### 1.3 Spatial Downsampling Strategy
- For MNIST, 1–2 pooling operations are enough (e.g., 28×28 → 14×14 → 7×7)
- Alternatively, use stride-2 convolutions for learnable downsampling

### 1.4 Global Average Pooling (GAP)
- `AdaptiveAvgPool2d(1)` turns each channel’s H×W map into a single scalar (mean over spatial dims)
- Output becomes `N × C × 1 × 1`, which you can squeeze to `N × C`
- Replaces large fully connected layers → drastically fewer parameters and less overfitting

Why GAP works well here:
- With a final `1×1 conv` producing 10 channels, GAP averages per-class evidence across spatial positions—simple and robust

### 1.5 Classifier Head
Two common endings:

- **A) Conv → GAP (no FC)**  
  `... → Conv2d(in_ch, 10, kernel_size=1) → AdaptiveAvgPool2d(1) → Flatten → logits (N×10)`
  - Minimal parameters; great for MNIST and tight parameter budgets

- **B) Conv → GAP → (Small FC)**  
  `... → GAP → Linear(C, 10)`
  - Slightly more parameters; sometimes useful to mix channels post-GAP. For MNIST, A is usually enough.

> FAQ: “Should I add an FC after GAP?”  
> Usually no for MNIST—GAP + 1×1 conv already acts as a linear classifier.

### 1.6 Output & Loss Interface
- If the model returns logits (preferred with CrossEntropyLoss), do not apply softmax/log_softmax in `forward`:
```python
criterion = nn.CrossEntropyLoss()  # expects logits
loss = criterion(logits, targets)
```
- If the model returns log-probabilities via `F.log_softmax`, use:
```python
criterion = nn.NLLLoss()           # expects log-probs
loss = criterion(log_probs, targets)
```
- Gotcha: Don’t combine `log_softmax` and `CrossEntropyLoss` (CE already applies log-softmax internally).

### 1.7 Shape Math & Param Counts (quick reference)
Conv output size (no dilation):
```
H_out = floor((H + 2P − K)/S) + 1
W_out = floor((W + 2P − K)/S) + 1
```
Conv parameters:
```
params = out_ch * (in_ch * K * K) + (bias ? out_ch : 0)
```
BatchNorm2d parameters: `2 * C` (γ, β) + running stats (non-trainable)

---

## 2. Training Loop: What Each Line Really Does

### 2.1 Mode Switching
- `model.train()`: enables Dropout and BN’s batch statistics
- `model.eval()`: disables Dropout and uses BN’s running statistics

### 2.2 Per-Epoch Flow
- Zero grads: `optimizer.zero_grad()` (prevents gradient accumulation)
- Forward: `pred = model(data)` (logits or log-probs)
- Loss: `loss = criterion(pred, target)` (CE expects logits; NLL expects log-probs)
- Backward: `loss.backward()` (populates `.grad` for leaf parameters)
- Step: `optimizer.step()` (updates weights)
- Metrics: accumulate loss/accuracy for logging
- Scheduler: typically step once per epoch after validation (for StepLR)

### 2.3 Accuracy Computation
```python
pred.argmax(dim=1).eq(target).sum().item()
```
Works for logits and log-probs since argmax is invariant to monotonic transforms.

### 2.4 Validation/Test
- Use `with torch.no_grad()` to save memory and time
- Ensure `model.eval()` for deterministic BN/Dropout behavior
- Average loss over all samples (sum batch losses weighted by batch size, then divide by dataset size)

### 2.5 Recommended “golden checks”
- Overfit a tiny subset (e.g., 100 samples) → should reach ~100% quickly
- Ensure `targets.dtype == torch.long`
- Verify no softmax when using CrossEntropyLoss
- Tune LR if loss diverges (too high) or stagnates (too low)

---

## 3. Optimizers & Schedulers: What/Why/When

### 3.1 Adam (current default)
- Adaptive per-parameter learning rates; handles noisy gradients well
- Good default for small/medium nets; fast convergence on MNIST
- Typical settings: `lr=1e-3, betas=(0.9, 0.999), eps=1e-8`
- Pros: little tuning needed; robust start
- Cons: can generalize slightly worse than tuned SGD on larger vision tasks

### 3.2 AdamW (worth trying)
- Adam with decoupled weight decay (better L2 regularization)
- Often cleaner convergence and better generalization than Adam with `weight_decay` inside Adam
- Example:
```python
torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

### 3.3 SGD + Momentum / Nesterov (the classic)
- Requires more LR tuning but can generalize strongly
- Starter config:
```python
torch.optim.SGD(
    model.parameters(),
    lr=0.05, momentum=0.9, nesterov=True, weight_decay=5e-4,
)
```
- Pair with a good scheduler (see below)

### 3.4 Learning-Rate Schedulers
- **StepLR**: drops LR by `gamma` every `step_size` epochs  
  Example: `step_size=15, gamma=0.1` → 10× drop at epoch 15
- **CosineAnnealingLR**: smooth cosine decay from initial LR to near-zero; great “set & forget” for fixed epochs
- **OneCycleLR**: increases LR then decreases (cosine), often with momentum cycling; very effective for short, aggressive training
- **ReduceLROnPlateau**: monitors a metric (e.g., val loss) and reduces LR when improvement stalls

> Tip: If switching from Adam to SGD, re-tune LR and pick an appropriate scheduler (Cosine/OneCycle often shine).

---

## 4. Regularization & Stability
- **Dropout**: ~0.05–0.2 in conv blocks is enough for MNIST; too high can underfit
- **BatchNorm**: keep enabled; stabilizes training and allows higher learning rates
- **Weight Decay**: use with SGD/AdamW (`1e-4` is a good start on MNIST)
- **Gradient Clipping**: optional (`clip_grad_norm_`) if you see gradient spikes
- **Label Smoothing**: optional (e.g., `CrossEntropyLoss(label_smoothing=0.05)` in newer PyTorch)

---

## 5. Practical Patterns (Pseudo-code)

### 5.1 Forward Head (logits → CrossEntropy)
```python
# forward()
x = feature_extractor(x)                       # convs, pools, BN, dropout
x = classifier_head(x)                         # 1×1 conv to 10
x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)  # N×10
return x  # logits
```

### 5.2 Train Step
```python
model.train()
for data, target in train_loader:
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    logits = model(data)                       # N×10
    loss = criterion(logits, target)           # CrossEntropyLoss
    loss.backward()
    optimizer.step()
```

### 5.3 Eval Step
```python
model.eval()
test_loss, correct = 0.0, 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        logits = model(data)
        test_loss += criterion(logits, target).item() * data.size(0)
        correct   += (logits.argmax(1) == target).sum().item()

test_loss /= len(test_loader.dataset)
test_acc   = 100.0 * correct / len(test_loader.dataset)
```

### 5.4 Scheduler
```python
for epoch in range(1, num_epochs + 1):
    train_one_epoch(...)
    validate(...)
    scheduler.step()  # StepLR: once per epoch (after val is common)
```

---

## 6. GAP vs. Fully Connected: Final Notes
- **GAP-only heads**: lower parameter count, less overfitting, strong for MNIST; ideal for <25K or <19K param budgets
- **GAP + FC**: slightly more expressive and higher params; on MNIST often no measurable gain if the backbone is decent

---

## ✅ Evaluation & Results

- Evaluated after each epoch on test set.  
- Metrics: **Loss** (generalization error) & **Accuracy** (classification rate).  
- Plotted curves confirm steady convergence without major overfitting.  

---

## 📊 Training & Test Logs

```text
Epoch 1
Train: Loss=0.5521 Batch_id=117 Accuracy=76.12
Test set: Average loss: 0.4052, Accuracy: 9528/10000 (95.28%)

Epoch 2
Train: Loss=0.3897 Batch_id=117 Accuracy=94.45
Test set: Average loss: 0.2133, Accuracy: 9776/10000 (97.76%)

Epoch 3
Train: Loss=0.2925 Batch_id=117 Accuracy=96.08
Test set: Average loss: 0.1476, Accuracy: 9837/10000 (98.37%)

Epoch 4
Train: Loss=0.2339 Batch_id=117 Accuracy=96.86
Test set: Average loss: 0.1277, Accuracy: 9855/10000 (98.55%)

Epoch 5
Train: Loss=0.2132 Batch_id=117 Accuracy=97.25
Test set: Average loss: 0.1145, Accuracy: 9862/10000 (98.62%)

Epoch 6
Train: Loss=0.1680 Batch_id=117 Accuracy=97.54
Test set: Average loss: 0.0939, Accuracy: 9880/10000 (98.80%)

Epoch 7
Train: Loss=0.1396 Batch_id=117 Accuracy=97.69
Test set: Average loss: 0.0760, Accuracy: 9898/10000 (98.98%)

Epoch 8
Train: Loss=0.1180 Batch_id=117 Accuracy=97.90
Test set: Average loss: 0.0677, Accuracy: 9917/10000 (99.17%)

Epoch 9
Train: Loss=0.0881 Batch_id=117 Accuracy=97.92
Test set: Average loss: 0.0621, Accuracy: 9907/10000 (99.07%)

Epoch 10
Train: Loss=0.0957 Batch_id=117 Accuracy=98.10
Test set: Average loss: 0.0592, Accuracy: 9911/10000 (99.11%)

Epoch 11
Train: Loss=0.1213 Batch_id=117 Accuracy=98.23
Test set: Average loss: 0.0523, Accuracy: 9922/10000 (99.22%)

Epoch 12
Train: Loss=0.2048 Batch_id=117 Accuracy=98.26
Test set: Average loss: 0.0481, Accuracy: 9915/10000 (99.15%)

Epoch 13
Train: Loss=0.1570 Batch_id=117 Accuracy=98.29
Test set: Average loss: 0.0422, Accuracy: 9942/10000 (99.42%)

Epoch 14
Train: Loss=0.1885 Batch_id=117 Accuracy=98.26
Test set: Average loss: 0.0394, Accuracy: 9930/10000 (99.30%)

Epoch 15
Train: Loss=0.0707 Batch_id=117 Accuracy=98.45
Test set: Average loss: 0.0337, Accuracy: 9934/10000 (99.34%)

Epoch 16
Train: Loss=0.0634 Batch_id=117 Accuracy=98.56
Test set: Average loss: 0.0337, Accuracy: 9945/10000 (99.45%)

Epoch 17
Train: Loss=0.0641 Batch_id=117 Accuracy=98.61
Test set: Average loss: 0.0336, Accuracy: 9940/10000 (99.40%)

Epoch 18
Train: Loss=0.1131 Batch_id=117 Accuracy=98.69
Test set: Average loss: 0.0337, Accuracy: 9943/10000 (99.43%)

Epoch 19
Train: Loss=0.0654 Batch_id=117 Accuracy=98.62
Test set: Average loss: 0.0326, Accuracy: 9943/10000 (99.43%)

Epoch 20
Train: Loss=0.0332 Batch_id=117 Accuracy=98.71
Test set: Average loss: 0.0326, Accuracy: 9946/10000 (99.46%)

---

## 7. Sanity & Reproducibility
- Set seeds for reproducibility:
```python
torch.manual_seed(42); torch.cuda.manual_seed_all(42)
np.random.seed(42); random.seed(42)
```
- Optional deterministic cuDNN (slower):
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## TL;DR
- Use logits + CrossEntropyLoss (no `log_softmax` in `forward`) or log-probs + NLLLoss — not both
- Adam or AdamW are excellent defaults; SGD + Momentum with Cosine/OneCycle can generalize best
- GAP + 1×1 Conv → 10 → GAP is a compact, high-performing classifier head for MNIST