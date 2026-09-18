"""Fixed-vocabulary graph classification, frozen-graph head fitting and ROI detection."""
import copy
import math
from dataclasses import asdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .supervised_data import ObjectAnnotation, ObjectViewTransform, SupervisedConfig
from .graph_memorypool_onceoptimizer import configure_orientation_contract, prepare_feature_subspaces


class FixedCNNEncoder(nn.Module):
    """Existing fixed CNN, including confidence-gated advanced modalities; no label input."""
    def __init__(self, memory_config, config=None, scales=(1., 2., 4.)):
        super().__init__()
        from ..cnns.features import RetinaModel, MultiScaleFeatureBank
        self.cfg, self.config = memory_config, config or SupervisedConfig()
        configure_orientation_contract(memory_config, len(scales))
        self.retina = RetinaModel(cropped_size=self.config.view_size, output_size=self.config.view_size)
        self.bank = MultiScaleFeatureBank(list(scales))
        self.requires_grad_(False)
        self.to(memory_config.device)
        self.eval()

    @torch.no_grad()
    def forward(self, observation):
        image = observation.view_tensor.to(self.cfg.device)
        rgb, grad, _, _, _, cropped = self.retina(image, -1, -1)
        curv, aps, ori, confidence = self.bank(cropped, precomputed_derivs={'grad': grad}, return_confidence=True)
        return prepare_feature_subspaces({'rgb': rgb, 'grad': grad, 'curvature': curv,
            'aspect': aps, 'orientation': ori, 'orientation_confidence': confidence}, self.cfg)


class ClassEvidenceReadout:
    def __init__(self, classes, config=None):
        self.classes = tuple(classes)
        self.config = config or SupervisedConfig()

    def encode(self, query, coordinator, graph_version):
        # h, evaluable, matched, normalized geometric error, missing, unresolved.
        rows = np.zeros((len(self.classes), 6), np.float32)
        complete = bool(query.diagnostics.get('search_complete', False))
        assignment_complete = bool(query.diagnostics.get('assignment_search_complete', not coordinator.gmem_i.nodes))
        rows[:, 4] = 1
        rows[:, 5] = not complete
        best = {}
        for match in query.entities:
            entity = coordinator.gmem_iii.entity_nodes.get(match.template_id)
            metadata = getattr(entity, 'supervision', None)
            if not metadata or metadata['label_id'] not in self.classes or not match.accepted:
                continue
            c = self.classes.index(metadata['label_id'])
            sigma = coordinator.cfg.graph_geometry_sigma
            q = match.score * match.matched_coverage * math.exp(-match.geometry_error**2/(2*sigma**2))
            if c in best and q <= rows[c, 0]:
                continue
            rows[c, :5] = q, match.evaluable_coverage, match.matched_coverage, match.geometry_error/sigma, 0
            best[c] = match
        return {'features': torch.from_numpy(rows.flatten()), 'scores': rows[:, 0].copy(),
                'best': best, 'search_complete': complete, 'graph_version': graph_version,
                'classes': self.classes, 'assignment_search_complete': assignment_complete}

    def predict(self, encoded, coordinator, observation=None):
        scores = encoded['scores']
        ranking = np.argsort(-scores, kind='stable')
        top = int(ranking[0])
        second = scores[ranking[1]] if len(ranking) > 1 else 0.
        found = top in encoded['best']
        accepted = found and scores[top] >= self.config.classification_threshold and scores[top]-second >= self.config.classification_margin
        # Sparse retrieval can support a positive hypothesis but cannot prove absence.
        status = 'PREDICTED' if accepted else ('UNKNOWN' if encoded['search_complete'] else 'UNRESOLVED')
        match = encoded['best'].get(top)
        box = None
        if match is not None and observation is not None:
            metadata = coordinator.gmem_iii.entity_nodes[match.template_id].supervision
            relative = metadata['canonical_box_relative_to_root']
            box = observation.box_to_original(tuple(v+match.point[i % 2] for i, v in enumerate(relative)))
        return {'status': status, 'class_id': self.classes[top] if accepted else None,
                'scores': {c: float(v) for c, v in zip(self.classes, scores)},
                'top_k': [(self.classes[int(i)], float(scores[i])) for i in ranking[:5]],
                'entity_id': match.template_id if match else None, 'point': match.point if match else None,
                'bbox': box, 'search_complete': encoded['search_complete'],
                'assignment_search_complete': encoded['assignment_search_complete'],
                'absence_proven': not found and encoded['search_complete'] and encoded['assignment_search_complete'],
                'graph_version': encoded['graph_version']}


class ClassifierTrainer:
    """Train only an evidence head on cached, frozen-graph features from disjoint source images."""
    def __init__(self, classes, graph_version, config=None, device='cpu'):
        self.classes, self.graph_version = tuple(classes), graph_version
        self.config, self.device = config or SupervisedConfig(), torch.device(device)
        # fork_rng avoids changing the application's random stream during construction.
        with torch.random.fork_rng():
            torch.manual_seed(self.config.seed)
            self.head = nn.Linear(6*len(classes), len(classes)).to(self.device)
        self.optimizer = torch.optim.Adam(self.head.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        self.history = []
        self.mode = 'single_label'

    def _validate_batch(self, batch, split):
        if batch['split'] != split or batch['graph_version'] != self.graph_version or tuple(batch['classes']) != self.classes:
            raise ValueError('Readout cache split, vocabulary or graph version mismatch')
        x, y = batch['features'], batch['targets']
        if x.ndim != 2 or x.shape[1] != self.head.in_features or not len(x) or len(x) != len(y) or not torch.isfinite(x).all():
            raise ValueError('Malformed or empty readout batch')
        if len(batch['source_ids']) != len(x):
            raise ValueError('Source image identity is required for every row')
        return x.to(self.device), y.to(self.device)

    def fit(self, training, validation, memory_source_ids, multilabel=False):
        x, y = self._validate_batch(training, 'readout_fit')
        vx, vy = self._validate_batch(validation, 'validation')
        train_sources, val_sources, memory_sources = set(training['source_ids']), set(validation['source_ids']), set(memory_source_ids)
        if not memory_sources or train_sources & val_sources or train_sources & memory_sources or val_sources & memory_sources:
            raise ValueError('memory_build/readout_fit/validation must have disjoint source images')
        self.mode = 'multilabel' if multilabel else 'single_label'
        if multilabel:
            if y.shape != (len(x), len(self.classes)) or vy.shape != (len(vx), len(self.classes)):
                raise ValueError('Multilabel targets must be [N,C], with unknown entries as NaN')
            if any(bool((torch.isfinite(t) & (t != 0) & (t != 1)).any()) for t in (y, vy)):
                raise ValueError('Known multilabel targets must be 0 or 1')
            known = torch.isfinite(y)
            if not known.any() or not torch.isfinite(vy).any():
                raise ValueError('At least one known train/validation label required')
            pos = ((y == 1) & known).sum(0)
            neg = ((y == 0) & known).sum(0)
            weights = (neg/pos.clamp_min(1)).clamp_min(1)
            def loss(logits, target):
                mask = torch.isfinite(target)
                raw = F.binary_cross_entropy_with_logits(logits, torch.nan_to_num(target).float(), pos_weight=weights, reduction='none')
                return raw[mask].mean() if mask.any() else logits.sum()*0
        else:
            if any(not torch.isfinite(t).all() or bool(((t < 0) | (t >= len(self.classes)) | (t != t.long())).any()) for t in (y, vy)):
                raise ValueError('Single-label targets must be finite vocabulary indices')
            y, vy = y.long(), vy.long()
            counts = torch.bincount(y, minlength=len(self.classes))
            if len(counts) != len(self.classes) or (counts == 0).any() or ((vy < 0) | (vy >= len(self.classes))).any():
                raise ValueError('Training must cover the fixed vocabulary; targets must be valid class indices')
            weights = counts.sum()/counts.float()/len(self.classes)
            def loss(logits, target):
                return F.cross_entropy(logits, target, weight=weights)
        generator = torch.Generator().manual_seed(self.config.seed)
        best, best_state, patience = math.inf, None, 0
        for epoch in range(self.config.epochs):
            self.head.train()
            order = torch.randperm(len(x), generator=generator).to(self.device)
            for start in range(0, len(x), self.config.batch_size):
                ids = order[start:start+self.config.batch_size]
                self.optimizer.zero_grad(set_to_none=True)
                value = loss(self.head(x[ids]), y[ids])
                value.backward()
                self.optimizer.step()
            self.head.eval()
            with torch.no_grad():
                value = float(loss(self.head(vx), vy))
            self.history.append({'epoch': epoch, 'validation_loss': value})
            if value < best:
                best, patience = value, 0
                best_state = (copy.deepcopy(self.head.state_dict()), copy.deepcopy(self.optimizer.state_dict()))
            else:
                patience += 1
                if patience >= self.config.patience:
                    break
        if best_state is None:
            raise ValueError('Nonfinite validation loss; no classifier checkpoint accepted')
        self.head.load_state_dict(best_state[0])
        self.optimizer.load_state_dict(best_state[1])
        return self.history

    @torch.no_grad()
    def predict(self, features, graph_version):
        if graph_version != self.graph_version:
            raise ValueError('Graph changed: rebuild readout cache and refit/calibrate the classifier')
        self.head.eval()
        logits = self.head(features.to(self.device))
        return logits.sigmoid() if self.mode == 'multilabel' else logits.softmax(-1)

    def state_dict(self):
        return {'classes': self.classes, 'graph_version': self.graph_version, 'config': asdict(self.config),
                'head': copy.deepcopy(self.head.state_dict()), 'optimizer': copy.deepcopy(self.optimizer.state_dict()),
                'mode': self.mode, 'history': copy.deepcopy(self.history)}

    def load_state_dict(self, state):
        if tuple(state['classes']) != self.classes or state['graph_version'] != self.graph_version or state['config'] != asdict(self.config):
            raise ValueError('Classifier checkpoint contract mismatch')
        self.head.load_state_dict(state['head'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.mode, self.history = state['mode'], copy.deepcopy(state['history'])


def classification_metrics(targets, predictions, classes):
    """Rejects/unresolved are a separate prediction column and count as errors."""
    if len(targets) != len(predictions) or not targets:
        raise ValueError('Nonempty equally-sized targets/predictions required')
    index = {c: i for i, c in enumerate(classes)}
    confusion = np.zeros((len(classes), len(classes)+1), dtype=int)
    unresolved = 0
    for target, prediction in zip(targets, predictions):
        if target not in index:
            raise ValueError('Unknown evaluation target')
        confusion[index[target], index.get(prediction['class_id'], len(classes))] += 1
        unresolved += prediction['status'] == 'UNRESOLVED'
    tp = np.diag(confusion[:, :-1])
    actual, predicted = confusion.sum(1), confusion[:, :-1].sum(0)
    f1 = 2*tp/np.maximum(1, actual+predicted)
    accepted = int(confusion[:, :-1].sum())
    return {'accuracy': float(tp.sum()/len(targets)), 'macro_f1': float(f1.mean()),
            'recall': dict(zip(classes, (tp/np.maximum(1, actual)).tolist())), 'confusion': confusion.tolist(),
            'prediction_columns': [*classes, 'REJECTED_OR_UNRESOLVED'], 'coverage': accepted/len(targets),
            'accepted_accuracy': float(tp.sum()/accepted) if accepted else None,
            'unresolved_rate': unresolved/len(targets), 'samples': len(targets)}


def box_iou(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    area_a, area_b = max(0., a[2]-a[0])*max(0., a[3]-a[1]), max(0., b[2]-b[0])*max(0., b[3]-b[1])
    return intersection/max(1e-12, area_a+area_b-intersection)


def classwise_nms(detections, threshold):
    kept = []
    for item in sorted(detections, key=lambda d: -d['score']):
        if not any(item['class_id'] == old['class_id'] and box_iou(item['bbox'], old['bbox']) > threshold for old in kept):
            kept.append(item)
    return kept


class ProposalDetector:
    def __init__(self, learner, encoder):
        self.learner, self.encoder = learner, encoder
        self.config = learner.config
        self.transform = ObjectViewTransform(self.config)
        self.readout = ClassEvidenceReadout(learner.classes, self.config)

    def proposals(self, height, width):
        boxes, seen = [], set()
        for fraction in self.config.proposal_scales:
            for aspect in self.config.proposal_aspects:
                if fraction <= 0 or aspect <= 0:
                    raise ValueError('Positive proposal scales/aspects required')
                bw = min(width, max(2, round(width*fraction*math.sqrt(aspect))))
                bh = min(height, max(2, round(height*fraction/math.sqrt(aspect))))
                sx, sy = max(1, round(bw*self.config.proposal_stride_fraction)), max(1, round(bh*self.config.proposal_stride_fraction))
                for y in sorted(set(range(0, height-bh+1, sy)) | {height-bh}):
                    for x in sorted(set(range(0, width-bw+1, sx)) | {width-bw}):
                        box = (x, y, x+bw, y+bh)
                        if box not in seen:
                            seen.add(box)
                            boxes.append(box)
        budget = self.config.proposal_budget
        if budget is not None and budget < 0:
            raise ValueError('proposal_budget must be nonnegative or None')
        return boxes if budget is None else boxes[:budget], budget is None or len(boxes) <= budget

    @torch.no_grad()
    def detect(self, image, image_id='inference', exact=False):
        # No ground-truth annotations or labels are accepted by this inference API.
        proposals, complete = self.proposals(*image.shape[-2:])
        detections, unresolved = [], 0
        for i, box in enumerate(proposals):
            annotation = ObjectAnnotation(image_id, str(i), None, box, f'proposal:{i}')
            observation = self.transform.make_view(image, annotation)
            query = self.learner.model.query_hierarchy(self.encoder(observation), valid_mask=observation.valid_mask, exact=exact)
            evidence = self.readout.encode(query, self.learner.model, self.learner.graph_version)
            unresolved += not evidence['search_complete']
            # Retain every accepted instance, not just one maximum per class.
            for match in query.entities:
                entity = self.learner.model.gmem_iii.entity_nodes[match.template_id]
                metadata = getattr(entity, 'supervision', None)
                if not match.accepted or not metadata or metadata['label_id'] not in self.learner.classes:
                    continue
                score = match.score*match.matched_coverage*math.exp(-match.geometry_error**2/(2*self.learner.cfg.graph_geometry_sigma**2))
                if score < self.config.classification_threshold:
                    continue
                relative = metadata['canonical_box_relative_to_root']
                recovered = observation.box_to_original(tuple(v+match.point[k % 2] for k, v in enumerate(relative)))
                if recovered[0] < recovered[2] and recovered[1] < recovered[3]:
                    detections.append({'image_id': image_id, 'class_id': metadata['label_id'], 'score': score,
                        'bbox': recovered, 'entity_id': match.template_id, 'proposal_id': i})
        return {'detections': classwise_nms(detections, self.config.nms_iou), 'proposals': proposals,
                'proposal_search_complete': complete, 'unresolved_proposals': unresolved,
                'search_complete': complete and unresolved == 0, 'graph_version': self.learner.graph_version}


def detection_metrics(predictions, annotations, classes, iou_threshold=.5, proposals=None):
    """All-point interpolated AP at one explicit IoU; no COCO multi-threshold claim."""
    ap, recall, errors = {}, {}, {'unmatched_or_localization': 0, 'duplicate': 0}
    for label in classes:
        truth = {}
        for image_id, objects in annotations.items():
            truth[image_id] = [a.bbox for a in objects if a.class_id == label]
        used, tp, fp = set(), [], []
        total = sum(map(len, truth.values()))
        for item in sorted((d for d in predictions if d['class_id'] == label), key=lambda d: -d['score']):
            candidates = truth.get(item['image_id'], [])
            overlaps = [box_iou(item['bbox'], box) for box in candidates]
            j = int(np.argmax(overlaps)) if overlaps else -1
            hit = j >= 0 and overlaps[j] >= iou_threshold
            key = (item['image_id'], j)
            correct = hit and key not in used
            tp.append(correct)
            fp.append(not correct)
            if correct:
                used.add(key)
            elif hit:
                errors['duplicate'] += 1
            else:
                errors['unmatched_or_localization'] += 1
        if not total:
            ap[label] = None
            continue
        cumulative = np.cumsum(tp)
        r = cumulative/total
        p = cumulative/np.maximum(1, cumulative+np.cumsum(fp))
        mr, mp = np.r_[0, r, 1], np.r_[0, p, 0]
        mp = np.maximum.accumulate(mp[::-1])[::-1]
        changed = np.flatnonzero(mr[1:] != mr[:-1])
        ap[label] = float(np.sum((mr[changed+1]-mr[changed])*mp[changed+1]))
        if proposals is not None:
            recall[label] = sum(any(box_iou(box, proposal) >= iou_threshold for proposal in proposals.get(image_id, []))
                for image_id, boxes in truth.items() for box in boxes)/total
    valid = [v for v in ap.values() if v is not None]
    return {'iou_threshold': iou_threshold, 'ap': ap, 'map': float(np.mean(valid)) if valid else None,
            'proposal_recall': recall, 'errors': errors}


def multilabel_metrics(targets, scores, classes, threshold=.5):
    """NaN targets are unknown, not negative; AP/F1 use only known entries."""
    y, s = np.asarray(targets), np.asarray(scores)
    if y.shape != s.shape or y.ndim != 2 or y.shape[1] != len(classes) or not np.isfinite(s).all():
        raise ValueError('Expected equally shaped [N,C] targets and finite scores')
    known = np.isfinite(y)
    if np.any(known & (y != 0) & (y != 1)):
        raise ValueError('Known multilabel targets must be binary')
    positive = s >= threshold
    tp = (known & (y == 1) & positive).sum(0)
    fp = (known & (y == 0) & positive).sum(0)
    fn = (known & (y == 1) & ~positive).sum(0)
    ap = {}
    for i, label in enumerate(classes):
        ids = np.flatnonzero(known[:, i])
        ids = ids[np.argsort(-s[ids, i], kind='stable')]
        truth = y[ids, i] == 1
        precision = np.cumsum(truth)/np.arange(1, len(ids)+1)
        ap[label] = float(precision[truth].mean()) if truth.any() else None
    return {'ap': ap, 'macro_f1': float(np.mean(2*tp/np.maximum(1, 2*tp+fp+fn))),
            'micro_f1': float(2*tp.sum()/max(1, (2*tp+fp+fn).sum())), 'threshold': threshold}
