"""Small CUDA compatibility smoke check, not a throughput/quality benchmark."""
import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'src'))
from nns.cnns.features import RetinaModel
from nns.memorygraphs.graph_memorypool_onceoptimizer import MemoryConfig, MultilevelCoordinator

device = torch.device('cuda:0')
assert torch.cuda.is_available()
result = {'python': sys.executable, 'torch': torch.__version__, 'torchvision': torchvision.__version__,
          'cuda_build': torch.version.cuda, 'gpu': torch.cuda.get_device_name(device),
          'capability': torch.cuda.get_device_capability(device), 'arches': torch.cuda.get_arch_list(),
          'total_memory_mib': torch.cuda.get_device_properties(device).total_memory / 1024**2}
with torch.no_grad():
    a = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
    assert (a @ a.T).sum().item() == 3680.
    pixels = torch.full((1, 3, 32, 32), .5, device=device)
    cnn = RetinaModel(cropped_size=32, output_size=32).eval().to(device)
    _, grad, _, _, _, cropped = cnn(pixels, center_x=-1, center_y=-1)
    assert grad.is_cuda and cropped.is_cuda
    grid = torch.zeros((1, 2, 2, 2), device=device)
    sampled = F.grid_sample(cropped, grid, align_corners=True)
    assert torch.isfinite(sampled).all().item()
    cfg = MemoryConfig()
    cfg.device = device; cfg.H = cfg.W = 32
    learner = MultilevelCoordinator(cfg)
    steps = []
    for step in range(2):
        learned = learner.learn_view({'grad': grad, 'rgb': cropped}, source_id='cuda_smoke',
                                     episode_id='same_smoke_episode',
                                     valid_mask=torch.ones((32, 32), dtype=torch.bool, device=device))
        assert learned.diagnostics['stage_seconds']['region_matching'] >= 0
        assert learner.gmem_i.nodes
        assert all(n.prototype.is_cuda and n.mask.is_cuda for n in learner.gmem_i.nodes.values())
        steps.append({k: learned.diagnostics[k] for k in ('memory_counts', 'stage_seconds', 'work_counts')})
    query = learner.query_hierarchy({'grad': grad, 'rgb': cropped})
    torch.cuda.synchronize(device)
    result.update(status='passed', checks=['matmul', 'Retina CNN', 'grid_sample',
                  '32x32 first/repeated learning', 'hierarchy query', 'persistent CUDA tensors'],
                  steps=steps, query_entities=len(query.entities),
                  peak_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2)
(root / 'results/cuda_environment_20260915/after_cuda_check.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
