# Conscious Demo

### Interpretable Visual Memory through Hierarchical Hypergraphs

**English** · [简体中文](README.zh-CN.md)

Conscious Demo explores how a visual system can **remember parts, compose objects, and retrieve their spatial arrangement** using explicit, inspectable structures. A fixed convolutional encoder supplies visual features; a growing hierarchy stores their appearance, geometry, and observation history.

The central research question is: **can persistent compositional memory support recognition and incremental learning without continually retraining the visual encoder?**

[Architecture](#1-model-architecture-and-principles) · [Experimental results](#2-model-performance) · [Algorithms](#3-algorithmic-details) · [Deployment](#34-deployment-and-reproduction)

## 1. Model Architecture and Principles

![Conceptual architecture: fixed CNN, hierarchical memory and spatial response pathways, with the exploratory saccade controller](docs/assets/readme/architecture.png)

*Figure 1. Project architecture supplied by the author. Gmem stores feature compositions; Gpos evaluates their spatial support. The lower panel describes the exploratory MICRO / MACRO / REVIEW saccade controller. Current supervised experiments use an observation-level graph builder and optimizer in place of this acquisition loop. “Conv” denotes the conceptual spatial-response computation; the current hierarchy also uses indexed candidates and explicit geometric verification.*

### A memory pathway and a spatial pathway

The model separates **what is stored** from **where it is supported in a new observation**:

- **Gmem — compositional memory.** Local feature nodes form regions; regions form entity hypotheses. Higher-level structures can share lower-level components while preserving their distinct roles and relative positions.
- **Gpos — spatial retrieval.** Feature responses generate candidate locations. Region and entity constraints verify whether the expected components occur in a compatible arrangement.
- **Observation-driven learning.** New observations either support an existing composition or create a protected observation template. Ambiguous associations remain explicit rather than forcing a merge.

This separation makes individual decisions inspectable: a retrieved entity can be traced to its supporting regions, feature responses, geometric constraints, and independent observations.

| Level | Memory representation | Spatial inference |
| --- | --- | --- |
| I | Local feature prototypes | Feature similarity and evaluability across the input |
| II | Regional compositions, represented with anchor–periphery relations and additional constraints | Region correspondence and relative geometry |
| III | Compositions of region occurrences, representing annotated objects or entity hypotheses | Joint entity verification and localization |

Gpos describes three levels of spatial computation; it is not implemented as three persistent graph copies. We use **hierarchical hypergraph memory** rather than “GNN” because the current model does not train a conventional message-passing graph neural network.

The broader goal is a perception–memory–action system. This repository currently develops its visual-memory component; multimodal understanding and an action decoder remain future work.

## 2. Model Performance

### Demonstrated strengths

**Perfect recall of all 34 stored objects in the evaluated memory pool**, together with correct training-view class predictions and reproducible checkpoint restoration, establishes a working memory–retrieval loop.

| Experiment | Observed result | Evaluation scope |
| --- | --- | --- |
| Frozen retrieval of stored objects | **34 / 34 target entities retrieved** | 20 VOC2007 training images, 34 annotated objects; known object boxes |
| Class prediction on those stored views | **34 / 34 correct** | Same training observations; not held-out accuracy |
| Original-view control across three categories | **3 / 3 retrieved** | One dog, one car, one cat |
| Small translation / brightness controls | **2 / 3 retrieved in each condition** | Translation: 4 pixels; brightness multiplier: 0.85 |
| Checkpoint and replay consistency | **5 / 5 checks passed** | Graph version, ledger, node counts, replay idempotence, query equivalence |

These are selected, reproducible results from the completed CUDA run `voc2007_20260922_222716_908742`, reviewed on September 23, 2026. The batch phase restored an existing 34-object memory; it did not add new independent learning evidence. See the [complete experimental review](docs/record/2026-09-23_full_experiment_review.md).

### From annotated views to explicit visual parts

![Two VOC2007 cat observations, normalized views, and writable regions](docs/assets/readme/object_views.png)

*Figure 2. Two actual Notebook observations: source image and annotation, normalized CNN input, and writable region. Bounding boxes guide observation selection; the writable mask is not a foreground segmentation label.*

![Five-modality region segmentation with selected anchors](docs/assets/readme/feature_regions.png)

*Figure 3. Recorded segmentation of a cat observation into gradient, color, curvature, anisotropy, and orientation regions. Stars identify selected anchors. Attribute labels indicate their parent gradient regions. Colors distinguish regions rather than semantic classes.*

### Inspectable retrieval evidence

![Successful translated dog retrieval and familiarity, difference, and unknown maps](docs/assets/readme/familiarity_translation.png)

*Figure 4. A successful 4-pixel translation control for dog image 003671. The panels show the input, familiarity, reliable difference, and unknown support. Heatmaps describe sparse template support, not dense object segmentation; bright unknown regions have not been explained by this template.*

### Current research frontier

Training-view recall is established, but **cross-image recognition is still an open problem**. In this run, none of the 84 readout-fit / validation / test objects produced an accepted entity. A weak-evidence classifier reached 51.61% test accuracy by predicting every object as `car`; it is not evidence of useful three-class discrimination. A 0.95 scale change and local occlusion each failed for all three control objects.

The next steps are therefore precise: resolve competing local correspondences, improve scale and partial-visibility inference, and validate class discrimination against initialization and constant-class baselines. Familiarity-guided updating remains a read-only diagnostic experiment. These boundaries distinguish the demonstrated memory mechanism from the capabilities still under investigation.

## 3. Algorithmic Details

### 3.1 Fixed CNN: structured visual features

The encoder combines a retinal preprocessing stage with a fixed multiscale feature bank. Gaussian smoothing and convolutional derivatives provide local measurements; first- and second-order derivatives are reused across the feature pipeline. The supervised encoder defaults to scales `(1, 2, 4)` and freezes all encoder parameters.

| Feature family | Role |
| --- | --- |
| `RGB` | Color and appearance continuity |
| `grad` | Gradient structure and edge continuity |
| `curv` | Local curvature attributes |
| `aps` | Local anisotropy; not the aspect ratio of an object bounding box |
| `ori` | Orientation, accompanied by confidence-aware processing |

The default `grad_refined` partitioning restricts edge attributes to retained gradient parent regions and then refines them according to their own responses. Related edge modalities share a **source family**, preventing multiple measurements of one contour from being counted as independent structural evidence.

Implementation: [features.py](src/nns/cnns/features.py), [FixedCNNEncoder](src/nns/memorygraphs/supervised_readout.py), and [partitioning design](docs/algorithm/Alg_RegionRefinementAndShape.md).

### 3.2 Hypergraph memory: features, regions, and entities

A higher-level node represents a composition of lower-level **occurrences and relations**, not just an unordered bag of feature vectors. An occurrence retains its role even when another occurrence references the same prototype.

- **Gmem I** stores feature prototypes associated with modalities.
- **Gmem II** organizes selected samples around regional anchors, retaining spatial relations and cross-region contacts.
- **Gmem III** binds regional occurrences into entities, recording member offsets, inter-member relations, source families, reliability, and evidence history. A region may participate in multiple entities.

Writing and retrieval are complementary. The writer turns continuous observations into explicit compositions; the retriever tests whether those compositions explain a new feature field. Binary signatures and inverted associations reduce the candidate set, followed by geometric verification. A coarse match is a candidate, not an identity decision; an incomplete search cannot prove novelty.

At a candidate pose, the verifier combines weighted feature scores, evaluable support, matched coverage, geometric residuals, and evidence-occupancy constraints. Current retrieval primarily models translation. A compact expression for its appearance aggregation is:

$$
S(H,b)=\exp\left(\frac{\sum_{i\in\mathcal E(H,b)}w_i\log\max(s_i(b),\epsilon)}{\sum_{i\in\mathcal E(H,b)}w_i}\right).
$$

Here $\mathcal E(H,b)$ contains evaluable slots of hypothesis $H$ at translation $b$; $s_i$ includes local response and displacement weighting. Acceptance additionally requires coverage and structural checks. A high appearance score alone is insufficient.

Implementation and design: [graph_memorypool_onceoptimizer.py](src/nns/memorygraphs/graph_memorypool_onceoptimizer.py), [hierarchical graph design](docs/algorithm/Alg_GraphandOptimizer.md).

### 3.3 Learning, consolidation, retrieval, and classification

The following stages describe the current observation-level pipeline and its relationship to the broader design.

| Stage | Algorithm and update behavior |
| --- | --- |
| Observation construction | Normalize an annotated object view, determine valid support, segment continuous regions, and sample anchors/peripheral points under configured budgets. |
| Regional reuse | Compare observations with existing Gmem II templates using coarse filtering and structural matching. Preserve local observations when reuse is ambiguous. |
| Entity association | Use region/role correspondence and geometry to select a compatible same-label entity in supervised learning. Otherwise create a separate annotated seed. Labels supervise writing; normal retrieval does not receive the target label. |
| Evidence accumulation | Track support and opportunities by source episode. Repeated views of one episode contribute at most one unit of independent support; geometry statistics update only when new supported evidence is added. |
| Structural consolidation | Update supported member offsets and relations; accumulate unexplained members as probationary candidates. Membership promotion remains conditional. Shared parts retain separate occurrence roles. |
| Ambiguity management | Maintain pending, confirmed, rejected, or stale associations. Template-version changes invalidate outdated association evidence. Confirmation does not automatically merge entity nodes. |
| Commit and recovery | Verify that a proposed entity explains its own write geometry. Retry with a local reconstruction when appropriate; commit a complete result, record its source, and save versioned checkpoints. |

For a newly supported displacement $\hat\delta$ with additional evidence weight $\Delta w$, the evidence accumulator updates its mean as:

$$
\mu' = \mu + \frac{\Delta w}{W+\Delta w}(\hat\delta-\mu).
$$

The same accumulator tracks geometric variability. Replaying an already recorded source does not increase independent support merely because it was presented again. This evidence-based learning is distinct from gradient training of the encoder.

**Classification on a growing graph.** Entity evidence is aggregated into a fixed $6C$-dimensional vector for $C$ classes: score, evaluability, matched coverage, normalized geometric error, missing/rejected status, and search completeness. This allows a fixed-size linear head while the graph grows. The standard readout uses accepted entities; a separate weak-evidence head includes rejected hypotheses for comparison. Neither weak class evidence nor a class-head prediction authorizes graph updates.

The head is fitted on `readout_fit` with a frozen graph; `validation` selects its checkpoint. Graph versions and feature-source contracts prevent mixing stale or incompatible caches. The detection interface generates candidate boxes, applies graph retrieval, and reports detections and proposal coverage; the current detector uses the standard graph readout.

**Familiarity-guided learning: diagnostic prototype.** A read-only probe groups local evidence by source family and separates familiarity $F$, reliable difference $D$, and uncertainty $U$:

$$
F_f=q_fv_f(1-u_f)s_f,\qquad D_f=q_fv_f(1-u_f)(1-s_f),\qquad U_f=1-q_fv_f(1-u_f).
$$

These engineering scores are not calibrated probabilities. The current probe tests whether correspondence is strong enough to motivate future selective updating; it does not yet implement that updater or reliable dense difference localization.

**Earlier active-observation controller.** The diagram's MICRO / MACRO / REVIEW loop combines interest, inhibition of return, semantic guidance, and movement cost to choose successive observations. Earlier controller implementations remain available for comparison. Current reported results come from the once-per-observation and supervised pipelines, not from an evaluation of the complete saccade loop.

Code: [supervised learning](src/nns/memorygraphs/supervised_graph_learning.py) · [pending associations](src/nns/memorygraphs/pending_associations.py) · [readout and classifier](src/nns/memorygraphs/supervised_readout.py) · [familiarity probe](src/nns/memorygraphs/familiarity_probe.py).

Design: [observation-level optimizer](docs/algorithm/Alg_onceoptimizer_beta.md) · [supervised learning](docs/algorithm/Alg_SupervisedGraphLearning.md) · [long-term training](docs/algorithm/Alg_LongTermSupervisedTraining.md) · [familiarity-guided learning](docs/algorithm/Alg_FamiliarityGuidedGraphLearning.md) · [experiment log](docs/record/index.md).

### 3.4 Deployment and Reproduction

This is a research Notebook workflow, not a packaged inference service. The development environment uses **Linux, Python 3.11, PyTorch 2.10.0+cu128, and torchvision 0.25.0+cu128**. CPU execution is supported, although graph search can be expensive. A complete dependency lockfile is not yet provided.

**Install from the repository root:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy pillow pandas matplotlib jupyterlab ipykernel
python -m ipykernel install --user --name conscious-demo --display-name "Conscious Demo"
```

For CPU-only execution, replace the PyTorch installation command with:

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
```

**Prepare VOC2007:**

```text
VOCdevkit/VOC2007/
├── JPEGImages/
├── Annotations/
└── ImageSets/Main/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

Provide the corresponding test images and annotations if evaluating the test split. In the first code cell of [the supervised Notebook](src/test_supervised_graph_learning.ipynb), set:

```python
VOC_ROOT = Path('/your/data/VOCdevkit/VOC2007')
RESUME_CHECKPOINT = None  # Fresh graph; or set an explicitly chosen compatible checkpoint
```

Default classes are `cat`, `dog`, and `car`. Official train images are partitioned into graph-memory construction and classifier fitting subsets; official val/test remain separate. The default protocol filters difficult objects and uses simplified detection metrics, not the official VOC evaluation implementation.

```bash
jupyter lab src/test_supervised_graph_learning.ipynb
```

Select **Conscious Demo**, then **Restart Kernel and Run All**. Experiment switches are enabled by default. `BATCH_IMAGE_LIMIT` and `QUERY_IMAGE_LIMIT` default to 20 source images; `PILOT_OBJECT_LIMIT` defaults to three objects. Full execution can take hours. `BATCH_START_IMAGE` selects a learning window; replaying an existing source is not new training.

Results are written to `results/supervised_graph/voc2007_<timestamp>/`: `memory.pt`, per-object learning/query logs, classifier caches, `head_results.json`, `training_recall.json`, `detection.json`, familiarity figures, and `all_results.json`. The [observation-level Notebook](src/test_memorypool_onceoptimizer.ipynb) provides a lower-level entry point; check its local input paths before running.

```bash
# Environment check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

# Static compilation and focused regression checks
python -m compileall -q src/nns
PYTHONPATH=src python -m unittest discover -s src/tests -p 'test_supervised*.py'
PYTHONPATH=src python -m unittest discover -s src/tests -p 'test_onceoptimizer_binary_modalities.py'
```

GPU feature computation coexists with CPU graph logic. If cuDNN versions conflict, check the selected kernel and external library paths; see the [environment record](docs/record/2026-09-15_onceoptimizer_performance_cuda_report.md). Figure provenance is documented in [the asset manifest](docs/assets/readme/README.md).
