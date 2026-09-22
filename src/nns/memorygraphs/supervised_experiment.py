"""Dataset-level entry points. Expensive work runs only on an explicit method call."""
import json
import time
import resource
from collections import Counter
from pathlib import Path
import torch
from .experiment_diagnostics import query_audit, save_json, json_value, weak_class_features
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

    def _objects(self, split, image_ids=None):
        for image_id in (self.splits['splits'][split] if image_ids is None else image_ids):
            image, objects = self.dataset.load_image_objects(image_id)
            for annotation in objects:
                if annotation.class_id in self.learner.classes:
                    others = [a.bbox for a in objects if a.object_id != annotation.object_id]
                    yield annotation, self.transform.make_view(image, annotation, other_boxes=others)

    def image_window(self, split, max_images=None, start_image=0):
        if not isinstance(start_image, int) or start_image < 0:
            raise ValueError('start_image must be a nonnegative integer')
        if max_images is not None and (not isinstance(max_images, int) or max_images < 1):
            raise ValueError('max_images must be positive or None')
        ids = self.splits['splits'][split]
        return ids[start_image:None if max_images is None else start_image + max_images]

    def build_memory(self, search_budget=None, on_result=None, max_images=None,
                     start_image=0, profile_stages=False):
        results = []
        ids = self.image_window('memory_build', max_images, start_image)
        for annotation, observation in self._objects('memory_build', ids):
            start = time.perf_counter()
            features = self.encoder(observation)
            if torch.device(self.learner.cfg.device).type == 'cuda':
                torch.cuda.synchronize(self.learner.cfg.device)
            encoded_at = time.perf_counter()
            result = self.learner.learn_object(observation, features, search_budget, profile_stages=profile_stages)
            if torch.device(self.learner.cfg.device).type == 'cuda':
                torch.cuda.synchronize(self.learner.cfg.device)
            result.update(image_id=annotation.image_id, object_id=annotation.object_id,
                class_id=annotation.class_id, encoder_seconds=encoded_at-start,
                total_seconds=time.perf_counter()-start, resources=resource_snapshot(self.learner.cfg.device))
            results.append(result)
            if on_result is not None:
                on_result(result)
        return {'objects': results, 'status_counts': dict(Counter(r['status'] for r in results)),
                'image_ids': ids, 'start_image': start_image, 'max_images': max_images}

    @torch.no_grad()
    def audit_memory_recall(self, max_images=None, start_image=0, on_result=None):
        """Frozen training recall, distinct from ledger replay and held-out evaluation."""
        version = self.learner.graph_version
        rows = []
        for annotation, observation in self._objects('memory_build', self.image_window('memory_build', max_images, start_image)):
            entry = self.learner.ledger.get(observation.ledger_key)
            if entry is None:
                row = {'image_id': annotation.image_id, 'object_id': annotation.object_id,
                       'status': 'NOT_COMMITTED', 'class_id': annotation.class_id}
            else:
                features = self.encoder(observation)
                query = self.learner.model.query_hierarchy(features, valid_mask=observation.valid_mask)
                target = entry['entity_id']
                hits = [m.template_id for m in query.entities]
                labels = [self.learner.model.gmem_iii.entity_nodes[eid].supervision['label_id'] for eid in hits]
                prediction = self.readout.predict(self.readout.encode(query, self.learner.model, version), self.learner.model, observation)
                row = {'image_id': annotation.image_id, 'object_id': annotation.object_id,
                       'class_id': annotation.class_id, 'target_entity_id': target, 'hits': hits,
                       'prediction': prediction, 'classification_correct': prediction['class_id'] == annotation.class_id,
                       'status': 'TARGET_HIT' if target in hits else 'SAME_CLASS_ONLY' if annotation.class_id in labels
                                 else 'WRONG_CLASS_ONLY' if hits else 'NO_HIT',
                       'structural_fallback': query.diagnostics.get('structural_fallback', False),
                       'graph_version': version}
            rows.append(row)
            if on_result is not None:
                on_result(row)
            if self.learner.graph_version != version:
                raise RuntimeError('Graph changed during frozen recall audit')
        return {'graph_version': version, 'rows': rows,
                'status_counts': dict(Counter(row['status'] for row in rows)),
                'evaluation_kind': 'training_recall_not_generalization'}

    @torch.no_grad()
    def query_split(self, split, exact=False, max_images=None, start_image=0, output_dir=None, on_result=None):
        if split not in ('readout_fit', 'validation', 'test'):
            raise ValueError('Classification caches must be independent of memory_build')
        rows, targets, source_ids, predictions, timings, audits, weak_rows = [], [], [], [], [], [], []
        log_path = None
        if output_dir is not None:
            log_path = Path(output_dir)/f"{split}_queries.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("")
        version = self.learner.graph_version
        for annotation, observation in self._objects(split, self.image_window(split, max_images, start_image)):
            # The annotation supplies a box for task A only; its label is never passed to retrieval/readout.
            start = time.perf_counter()
            features = self.encoder(observation)
            query = self.learner.model.query_hierarchy(features, valid_mask=observation.valid_mask, exact=exact, trace=True)
            encoded = self.readout.encode(query, self.learner.model, version)
            predictions.append(self.readout.predict(encoded, self.learner.model, observation))
            rows.append(encoded['features'])
            targets.append(self.learner.classes.index(annotation.class_id))
            source_ids.append(annotation.image_id)
            timings.append(time.perf_counter()-start)
            weak_rows.append(weak_class_features(query, self.learner.model, self.learner.classes))
            audit = {'image_id': annotation.image_id, 'object_id': annotation.object_id,
                     'target_class': annotation.class_id, 'graph_version': version,
                     'seconds': timings[-1], **query_audit(query, predictions[-1])}
            audits.append(audit)
            if log_path is not None:
                with log_path.open('a') as stream: stream.write(json.dumps(json_value(audit), ensure_ascii=False)+'\n')
            if on_result is not None: on_result(audit)
        if version != self.learner.graph_version:
            raise RuntimeError('Graph changed while building a frozen readout cache')
        result = {'features': torch.stack(rows) if rows else torch.empty(0, 6*len(self.learner.classes)),
                'weak_features': torch.stack(weak_rows) if weak_rows else torch.empty(0, 6*len(self.learner.classes)),
                'feature_source': 'accepted', 'audits': audits,
                'targets': torch.tensor(targets, dtype=torch.long), 'source_ids': source_ids,
                'split': split, 'classes': self.learner.classes, 'graph_version': version,
                'predictions': predictions, 'seconds': timings, 'resources': resource_snapshot(self.learner.cfg.device)}
        if output_dir is not None:
            temporary = Path(output_dir)/f'{split}_cache.pt.tmp'
            torch.save(result, temporary); temporary.replace(Path(output_dir)/f'{split}_cache.pt')
            save_json(Path(output_dir)/f'{split}_summary.json', {
                'samples': len(rows), 'graph_version': version,
                'rejection_counts': dict(Counter(p.get('rejection_reason') or 'accepted' for p in predictions)),
                'accepted_feature_unique_rows': len(torch.unique(result['features'], dim=0)),
                'weak_feature_unique_rows': len(torch.unique(result['weak_features'], dim=0)),
                'metrics': classification_metrics([self.learner.classes[int(y)] for y in targets], predictions, self.learner.classes) if rows else None})
        return result

    def evaluate(self, cache, trainer=None):
        if cache['split'] not in ('validation', 'test') or cache['graph_version'] != self.learner.graph_version:
            raise ValueError('Evaluation requires a current validation/test cache')
        predictions = cache['predictions']
        if trainer is not None:
            if cache.get('feature_source', 'accepted') != trainer.feature_source:
                raise ValueError('Classifier evidence source differs from cache')
            if trainer.mode != 'single_label':
                raise ValueError('Use multilabel_metrics for multilabel evaluation')
            probabilities = trainer.predict(cache['features'], self.learner.graph_version).cpu()
            predictions = [{'class_id': self.learner.classes[int(row.argmax())], 'status': 'PREDICTED'} for row in probabilities]
        targets = [self.learner.classes[int(y)] for y in cache['targets']]
        return classification_metrics(targets, predictions, self.learner.classes)
