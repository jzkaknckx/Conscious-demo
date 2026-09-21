"""Explicit annotation contracts and geometry for supervised object observations."""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import random
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SupervisedConfig:
    learning_contract: str = 'supervised-family-quality-v2'
    view_size: int = 256
    context_fraction: float = .1
    preprocessing_version: str = 'object-view-v1'
    member_budget: int = 24
    family_budget: int = 8
    leaf_budget: int = 512
    min_members: int = 2
    skip_truncated_regions: bool = True  # choose alternative complete samples; never silently accept truncation
    incomplete_geometry_weight: float = .5  # soft penalty, not blanket rejection
    modality_diversity_bonus: float = .25  # reward reliable unused modalities
    grid_size: int = 4
    min_inside_fraction: float = .8
    classification_threshold: float = .45
    classification_margin: float = .04
    nms_iou: float = .5
    proposal_scales: tuple = (.25, .5, 1.)
    proposal_aspects: tuple = (.5, 1., 2.)
    proposal_stride_fraction: float = .5
    proposal_budget: int | None = None
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    patience: int = 5
    seed: int = 0

    def __post_init__(self):
        for name in ('view_size', 'member_budget', 'family_budget', 'leaf_budget',
                     'min_members', 'grid_size', 'batch_size', 'epochs', 'patience'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.member_budget < self.min_members or not np.isfinite(self.context_fraction) or self.context_fraction < 0:
            raise ValueError('Invalid member budget/context')
        for name in ('min_inside_fraction', 'classification_threshold', 'classification_margin', 'nms_iou',
                     'incomplete_geometry_weight', 'modality_diversity_bonus'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in [0,1]')
        if not 0 < self.proposal_stride_fraction <= 1:
            raise ValueError('proposal_stride_fraction must be in (0, 1]')


@dataclass(frozen=True)
class ObjectAnnotation:
    image_id: str
    object_id: str
    class_id: str | None
    bbox: tuple
    annotation_id: str
    dataset_version: str = 'ILSVRC-DET-2014'
    annotation_version: str = '1'
    coordinate_convention: str = '0based_halfopen'
    annotation_source: str = 'manual'
    label_confidence: float = 1.
    flags: dict = field(default_factory=dict)

    @property
    def episode_id(self):
        return json.dumps((self.dataset_version, self.image_id, self.object_id))

    def validate(self, height, width, require_label=True):
        x0, y0, x1, y1 = self.bbox
        if not all(np.isfinite(self.bbox)) or not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError('INVALID_ANNOTATION: nonfinite, degenerate or out-of-image bbox')
        if require_label and not self.class_id:
            raise ValueError('INVALID_ANNOTATION: missing class_id')
        if not np.isfinite(self.label_confidence) or not 0 < self.label_confidence <= 1:
            raise ValueError('INVALID_ANNOTATION: invalid label confidence')


class AnnotationDataset:
    def __init__(self, image_dir, annotation_dir, coordinate_convention, dataset_version='ILSVRC-DET-2014'):
        if coordinate_convention not in ('1based_inclusive', '0based_halfopen'):
            raise ValueError('Explicit coordinate convention required')
        self.image_dir, self.annotation_dir = Path(image_dir), Path(annotation_dir)
        self.coordinate_convention, self.dataset_version = coordinate_convention, dataset_version

    def annotations(self, image_id):
        path = self.annotation_dir / f'{image_id}.xml'
        root = ET.parse(path).getroot()
        width, height = int(root.findtext('size/width')), int(root.findtext('size/height'))
        result = []
        for i, obj in enumerate(root.findall('object')):
            box = [float(obj.findtext(f'bndbox/{key}')) for key in ('xmin', 'ymin', 'xmax', 'ymax')]
            if self.coordinate_convention == '1based_inclusive':
                box[0] -= 1
                box[1] -= 1
            annotation = ObjectAnnotation(image_id, str(i), obj.findtext('name'), tuple(box),
                f'{image_id}:{i}', self.dataset_version, coordinate_convention=self.coordinate_convention,
                annotation_source=str(path), flags={k: obj.findtext(k) for k in ('difficult', 'truncated', 'occluded')})
            annotation.validate(height, width)
            result.append(annotation)
        return result

    def manifest(self):
        return {p.stem: self.annotations(p.stem) for p in sorted(self.annotation_dir.glob('*.xml'))}

    def load_image_objects(self, image_id):
        from PIL import Image
        paths = [self.image_dir / f'{image_id}{ext}' for ext in ('.JPEG', '.jpg', '.jpeg', '.png')]
        path = next((p for p in paths if p.exists()), None)
        if path is None:
            raise FileNotFoundError(image_id)
        with Image.open(path) as image:
            values = np.asarray(image.convert('RGB')).copy()
        objects = self.annotations(image_id)
        for annotation in objects:
            annotation.validate(*values.shape[:2])
        return torch.from_numpy(values).permute(2, 0, 1).float().div(255).unsqueeze(0), objects


def grouped_split(manifest, classes, seed=0, fractions=(.4, .2, .2, .2)):
    """One source image in exactly one split; fail explicitly if coverage is insufficient."""
    if len(fractions) != 4 or any(x <= 0 for x in fractions) or not np.isclose(sum(fractions), 1):
        raise ValueError('Four positive fractions summing to one required')
    if not classes or len(set(classes)) != len(classes):
        raise ValueError('A nonempty unique class vocabulary is required')
    names = ('memory_build', 'readout_fit', 'validation', 'test')
    ids = sorted(k for k, annotations in manifest.items() if any(a.class_id in classes for a in annotations))
    random.Random(seed).shuffle(ids)
    cuts = [0] + [round(len(ids) * sum(fractions[:i])) for i in range(1, 4)] + [len(ids)]
    splits = {name: ids[cuts[i]:cuts[i+1]] for i, name in enumerate(names)}
    missing = {name: sorted(set(classes) - {a.class_id for k in keys for a in manifest[k]})
               for name, keys in splits.items()}
    if any(missing.values()):
        raise ValueError(f'Insufficient grouped class coverage; reduce classes or supply a curated split: {missing}')
    return {'seed': seed, 'classes': list(classes), 'splits': splits}


@dataclass
class SupervisedObservation:
    annotation: ObjectAnnotation
    view_tensor: torch.Tensor
    image_valid_mask: torch.Tensor
    object_roi_mask: torch.Tensor
    transform: torch.Tensor
    inverse_transform: torch.Tensor
    canonical_box: tuple
    original_shape: tuple
    preprocessing_version: str
    foreground_probability: torch.Tensor | None = None
    ambiguous_ownership_mask: torch.Tensor | None = None
    augmentation_id: str = 'identity'
    source_digest: str = ''

    @property
    def ledger_key(self):
        return (self.annotation.episode_id, self.annotation.annotation_version, self.preprocessing_version)

    @property
    def valid_mask(self):
        return self.image_valid_mask & self.object_roi_mask

    def box_to_original(self, box):
        points = torch.tensor([[box[0], box[1], 1.], [box[2], box[3], 1.]], dtype=torch.float64)
        points = points @ self.inverse_transform.T.cpu()
        h, w = self.original_shape
        return (float(points[0, 0].clamp(0, w)), float(points[0, 1].clamp(0, h)),
                float(points[1, 0].clamp(0, w)), float(points[1, 1].clamp(0, h)))


class ObjectViewTransform:
    def __init__(self, cfg=None):
        self.cfg = cfg or SupervisedConfig()

    def make_view(self, image, annotation, foreground_probability=None, other_boxes=()):
        """Pixel-edge affine convention; CNN sees context before the ROI mask is applied."""
        if (image.ndim != 4 or image.shape[:2] != (1, 3) or not image.is_floating_point()
                or not torch.isfinite(image).all() or bool(((image < 0) | (image > 1)).any())):
            raise ValueError('Expected floating RGB [1,3,H,W] in [0,1]')
        h, w = image.shape[-2:]
        annotation.validate(h, w, require_label=False)
        x0, y0, x1, y1 = annotation.bbox
        margin = self.cfg.context_fraction
        left, top = x0 - (x1-x0)*margin, y0 - (y1-y0)*margin
        cw, ch = (x1-x0)*(1+2*margin), (y1-y0)*(1+2*margin)
        size = self.cfg.view_size
        scale = size / max(cw, ch)
        px, py = (size-cw*scale)/2, (size-ch*scale)/2
        transform = torch.tensor([[scale, 0, px-left*scale], [0, scale, py-top*scale], [0, 0, 1]], dtype=torch.float64)
        yy, xx = torch.meshgrid(torch.arange(size, device=image.device, dtype=image.dtype)+.5,
                                torch.arange(size, device=image.device, dtype=image.dtype)+.5, indexing='ij')
        ox, oy = (xx-px)/scale+left, (yy-py)/scale+top
        grid = torch.stack((2*ox/w-1, 2*oy/h-1), -1).unsqueeze(0)
        view = F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=False)
        valid = ((ox >= max(0., left)) & (ox < min(w, left+cw)) &
                 (oy >= max(0., top)) & (oy < min(h, top+ch)))
        roi = (ox >= x0) & (ox < x1) & (oy >= y0) & (oy < y1)
        ownership = torch.zeros_like(valid)
        for bx0, by0, bx1, by1 in other_boxes:
            ownership |= (ox >= bx0) & (ox < bx1) & (oy >= by0) & (oy < by1)
        fg = None
        if foreground_probability is not None:
            fg = torch.as_tensor(foreground_probability, device=image.device, dtype=image.dtype).reshape(1, 1, h, w)
            if not torch.isfinite(fg).all() or bool(((fg < 0) | (fg > 1)).any()):
                raise ValueError('Foreground probabilities must be finite and in [0,1]')
            fg = F.grid_sample(fg, grid, align_corners=False)[0, 0]
        canonical = ((x0-left)*scale+px, (y0-top)*scale+py, (x1-left)*scale+px, (y1-top)*scale+py)
        digest = hashlib.sha256(image.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        return SupervisedObservation(annotation, view, valid, roi, transform, torch.linalg.inv(transform),
            canonical, (h, w), self.cfg.preprocessing_version, fg, ownership, source_digest=digest)
