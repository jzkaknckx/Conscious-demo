"""Dataset-level entry points. Expensive work runs only on an explicit method call."""
import time
import resource
from collections import Counter
from pathlib import Path
import torch
from .supervised_data import ObjectAnnotation, ObjectViewTransform
from .supervised_readout import ClassEvidenceReadout, classification_metrics


def resource_snapshot(device):
    # Linux ru_maxrss is KiB; VmRSS is current resident memory, not allocated tensor storage.
    result = {'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            result['rss_bytes'] = int(line.split()[1])*1024
    device = torch.device(device)
    if device.type == 'cuda' and torch.cuda.is_available():
        result.update(cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                      cuda_reserved_bytes=torch.cuda.memory_reserved(device),
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
    return result


class SupervisedExperiment:
    def __init__(self, learner, encoder, dataset, split_manifest):
        self.learner, self.encoder, self.dataset = learner, encoder, dataset
        self.splits = split_manifest
        if tuple(split_manifest['classes']) != learner.classes:
            raise ValueError('Split vocabulary differs from learner')
        seen = set()
        for split in ('memory_build', 'readout_fit', 'validation', 'test'):
            ids = set(split_manifest['splits'][split])
            if seen & ids:
                raise ValueError('Source image leakage across splits')
            seen |= ids
        if learner.split_manifest is not None and learner.split_manifest != split_manifest:
            raise ValueError('Existing learner belongs to a different experiment split')
        learner.split_manifest = split_manifest
        self.transform = ObjectViewTransform(learner.config)
        self.readout = ClassEvidenceReadout(learner.classes, learner.config)

    def _objects(self, split):
        for image_id in self.splits['splits'][split]:
            image, objects = self.dataset.load_image_objects(image_id)
            for annotation in objects:
                if annotation.class_id in self.learner.classes:
                    others = [a.bbox for a in objects if a.object_id != annotation.object_id]
                    yield annotation, self.transform.make_view(image, annotation, other_boxes=others)

    def build_memory(self, search_budget=None, on_result=None):
        results = []
        for annotation, observation in self._objects('memory_build'):
            start = time.perf_counter()
            features = self.encoder(observation)
            if torch.device(self.learner.cfg.device).type == 'cuda':
                torch.cuda.synchronize(self.learner.cfg.device)
            encoded_at = time.perf_counter()
            result = self.learner.learn_object(observation, features, search_budget)
            if torch.device(self.learner.cfg.device).type == 'cuda':
                torch.cuda.synchronize(self.learner.cfg.device)
            result.update(image_id=annotation.image_id, object_id=annotation.object_id,
                class_id=annotation.class_id, encoder_seconds=encoded_at-start,
                total_seconds=time.perf_counter()-start, resources=resource_snapshot(self.learner.cfg.device))
            results.append(result)
            if on_result is not None:
                on_result(result)
        return {'objects': results, 'status_counts': dict(Counter(r['status'] for r in results))}

    @torch.no_grad()
    def query_split(self, split, exact=False):
        if split not in ('readout_fit', 'validation', 'test'):
            raise ValueError('Classification caches must be independent of memory_build')
        rows, targets, source_ids, predictions, timings = [], [], [], [], []
        version = self.learner.graph_version
        for annotation, observation in self._objects(split):
            # The annotation supplies a box for task A only; its label is never passed to retrieval/readout.
            start = time.perf_counter()
            features = self.encoder(observation)
            query = self.learner.model.query_hierarchy(features, valid_mask=observation.valid_mask, exact=exact)
            encoded = self.readout.encode(query, self.learner.model, version)
            predictions.append(self.readout.predict(encoded, self.learner.model, observation))
            rows.append(encoded['features'])
            targets.append(self.learner.classes.index(annotation.class_id))
            source_ids.append(annotation.image_id)
            timings.append(time.perf_counter()-start)
        if version != self.learner.graph_version:
            raise RuntimeError('Graph changed while building a frozen readout cache')
        return {'features': torch.stack(rows) if rows else torch.empty(0, 6*len(self.learner.classes)),
                'targets': torch.tensor(targets, dtype=torch.long), 'source_ids': source_ids,
                'split': split, 'classes': self.learner.classes, 'graph_version': version,
                'predictions': predictions, 'seconds': timings, 'resources': resource_snapshot(self.learner.cfg.device)}

    def evaluate(self, cache, trainer=None):
        if cache['split'] not in ('validation', 'test') or cache['graph_version'] != self.learner.graph_version:
            raise ValueError('Evaluation requires a current validation/test cache')
        predictions = cache['predictions']
        if trainer is not None:
            if trainer.mode != 'single_label':
                raise ValueError('Use multilabel_metrics for multilabel evaluation')
            probabilities = trainer.predict(cache['features'], self.learner.graph_version).cpu()
            predictions = [{'class_id': self.learner.classes[int(row.argmax())], 'status': 'PREDICTED'} for row in probabilities]
        targets = [self.learner.classes[int(y)] for y in cache['targets']]
        return classification_metrics(targets, predictions, self.learner.classes)
