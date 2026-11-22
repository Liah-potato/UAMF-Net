# Decision procedure and implementation map

## Inputs and experts

For each pathology image, a manifest supplies a reference class and explicit
paths to available CT and gastroscopy images from the same patient. Reference
labels are used for training and evaluation, never as inputs to arbitration.

Three separate `timm` classifiers produce three-class softmax vectors. The
experts are existing image-classification architectures with task-specific
classification heads. The fusion procedure does not concatenate features,
train a fusion head, or implement a custom 3D encoder.

## Pathology views and entropy

The five deterministic pathology views are an original center crop, horizontal
flip, vertical flip, 90-degree rotation, and enlarged center crop. The default
224-pixel crop uses a 256-pixel resize; the enlarged view uses 280 pixels.
Training uses a random crop and random horizontal flip. All image transforms
use the configured RGB normalization. CT uses the 512-pixel input specified for
its selected backbone; gastroscopy uses 224 pixels. Both auxiliary experts use
one center-cropped view.

For pathology view probabilities $p^{(v)}$, average probabilities before
calculating entropy:

$$\bar p = \frac{1}{V}\sum_{v=1}^{V}p^{(v)},\qquad
H(\bar p)=-\sum_{c=0}^{2}\bar p_c\ln\bar p_c.$$

The baseline label is $y_p=\arg\max_c\bar p_c$ and confidence is
$C_p=\max_c\bar p_c$. Natural-log entropy lies in $[0,\ln 3]$.
This is predictive entropy of the averaged distribution. It is not a separate
estimate of epistemic uncertainty or a measurement of cellular adhesion.

If $H(\bar p)<\tau_H$, retain $y_p$ without invoking either auxiliary expert.
At equality, the gate opens. The default $\tau_H$ is 0.45.

## Auxiliary selection

When the gate opens, infer each available auxiliary modality once. Among
available experts, select the one with the highest maximum class probability.
An exact confidence tie follows `policy.tie_priority`, defaulting to CT.
Within an expert, equal class probabilities use the lowest class index.

Let the selected auxiliary prediction be $y_a$, its confidence be $C_a$, and
$\Delta=C_a-C_p$. The comparison uses raw softmax confidence; no probability
calibration guarantee is assumed. If both auxiliary inputs are absent, retain
the pathology label. Missing inputs are represented by `None`, not artificial
uniform probability predictions.

## Complete arbitration rules

The default policy uses $\tau_{rec}=0.65$, $\tau_{benign}=0.80$, and $\delta=0.15$.
These are configuration defaults, not a claim of an independently validated
optimal threshold set.

| Original pathology class | Auxiliary class | Additional conditions | Final class |
| --- | --- | --- | --- |
| 0 or 2 | 1 | $C_a>\tau_{rec}$ and $\Delta>\delta$ | 1 |
| 1 | 2 | $C_a>\tau_{benign}$ | 2 |
| Any | Any other case | — | Original pathology class |

All auxiliary confidence and margin comparisons are strict. The benign rule
does not add a margin condition. Class 0 is protected by the entropy gate when
confident but may change to Class 1 when the gate opens and the recall conditions
are met. There is no direct Class 0-to-Class 2 or Class 1-to-Class 0 override.
No override can occur while the gate is closed.

```text
pathology = average five pathology softmax vectors
baseline = argmax(pathology)
if entropy(pathology) < entropy_threshold:
    return baseline
available = predictions from available CT and gastroscopy inputs
if available is empty:
    return baseline
auxiliary = highest-confidence available expert, using an explicit tie rule
if baseline != 1 and auxiliary.class == 1:
    if auxiliary.confidence > recall_threshold and
       auxiliary.confidence - pathology.confidence > margin:
        return 1
elif baseline == 1 and auxiliary.class == 2:
    if auxiliary.confidence > benign_threshold:
        return 2
return baseline
```

## Training objective and selection

Each expert is trained independently. For logits $z$ and hard reference class
$y$, the default per-image objective is:

$$\mathcal L=-\ln p_y+\epsilon_{poly}(1-p_y),\qquad \epsilon_{poly}=1.0.$$

`head_only` freezes the backbone parameters and running statistics while training
the model's classifier returned by `get_classifier()`. `full` updates all model
parameters. Standard cross-entropy with optional label smoothing is available
as an explicit alternative. The hard-label Poly-1 path rejects nonzero label
smoothing so that its definition remains unambiguous.

Checkpoints are selected using validation loss. Threshold search uses validation
traces only and exports a new fixed configuration. Test evaluation does not
search thresholds. This workflow is an implementation interface, not evidence
that any particular historical experiment followed these settings.

## Evaluation units and accounting

Every manifest row is one pathology-image evaluation record. Multiple images
from the same patient stay in a single split. Auxiliary images repeated across
that patient's pathology rows are deduplicated for single-modality training.
No patient-level aggregation or confidence interval is implied by image accuracy.

Let $R$ count incorrect-to-correct changes and $W$ count correct-to-incorrect
changes. Net additional correct predictions equal $R-W$, while total overrides
can also include incorrect-to-incorrect class changes. The implementation reports
these quantities separately and retains unchanged examples in its output.

Per-class metrics use a one-versus-rest confusion matrix. Macro-F1 averages the
three class F1 values, assigning zero where a denominator is zero. The confusion
matrix uses reference classes as rows and predicted classes as columns.

## Code map

| Method component | Source |
| --- | --- |
| Explicit sample pairing and patient separation | `uamf_net/data.py` |
| Model architecture, normalization, and views | `uamf_net/models.py` |
| Training objective and frozen-backbone behavior | `uamf_net/training.py` |
| Entropy, gate, auxiliary selection, and arbitration | `uamf_net/fusion.py` |
| Conditional expert calls and complete traces | `uamf_net/pipeline.py` |
| Validation-only policy selection | `uamf_net/tuning.py` |
| Per-class metrics and override counts | `uamf_net/metrics.py` |
| Hand-constructed examples without clinical labels | `uamf_net/demo.py` |

No clinical images, model checkpoints, reconstructed patient records, benchmark
results, or resource measurements are embedded in the source code.
