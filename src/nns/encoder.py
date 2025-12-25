

# ----------------------------
# Main unified class implementation
# ----------------------------

class SparseGraphMemory:
    """Sparse graph memory implementing Route A: modal-separated graphs + inverted indexes.

    High-level usage:
        mem = SparseGraphMemory(modalities=["edge","hue","flow","curv"],
                                 pos_window=5.0, spatial_threshold=5.0)
        # input_features: list of dicts, each with keys: modality, signature (hashable) or vector, position=(x,y)
        candidates = mem.retrieve(input_features)
        mem.update(input_features, allow_update=True)

    The implementation keeps everything in-memory using Python dicts and lists for clarity.
    It is designed for debug and unit testing; for large-scale deployment replace data
    structures with more scalable implementations (C-extensions, DB-backed inverted indexes, FAISS for vectors).
    """

    class Node:
        def __init__(self, node_id: int, modality: str, signature: Any, position: Tuple[float,float], vector: Optional[List[float]] = None):
            self.id = node_id
            self.modality = modality
            self.signature = signature  # discrete key or quantized hash
            self.position = position
            self.vector = vector  # optional continuous vector
            self.count = 1
            self.last_seen = time.time()

        def touch(self, pos: Tuple[float,float], alpha: float = 0.2):
            """Update node position with EMA and increment count."""
            x_old, y_old = self.position
            x_new = alpha * pos[0] + (1 - alpha) * x_old
            y_new = alpha * pos[1] + (1 - alpha) * y_old
            self.position = (x_new, y_new)
            self.count += 1
            self.last_seen = time.time()

    class Edge:
        def __init__(self, src: int, dst: int):
            self.src = src
            self.dst = dst
            self.weight = 1
            self.last_seen = time.time()

        def touch(self):
            self.weight += 1
            self.last_seen = time.time()

    class ModalGraph:
        def __init__(self, modality: str):
            self.modality = modality
            self.nodes: Dict[int, SparseGraphMemory.Node] = {}
            # adjacency: node_id -> dict(neighbor_id -> Edge)
            self.adj: Dict[int, Dict[int, SparseGraphMemory.Edge]] = {}

        def add_node(self, node: 'SparseGraphMemory.Node') -> None:
            self.nodes[node.id] = node
            self.adj[node.id] = {}

        def add_edge(self, src_id: int, dst_id: int) -> None:
            if dst_id in self.adj[src_id]:
                self.adj[src_id][dst_id].touch()
            else:
                e = SparseGraphMemory.Edge(src_id, dst_id)
                self.adj[src_id][dst_id] = e

        def remove_node(self, node_id: int) -> None:
            # remove edges to/from node and delete node entry
            if node_id in self.adj:
                # remove outgoing
                del self.adj[node_id]
            # remove incoming
            for nid, neigh in list(self.adj.items()):
                if node_id in neigh:
                    del neigh[node_id]
            if node_id in self.nodes:
                del self.nodes[node_id]

    def __init__(self,
                 modalities: List[str],
                 spatial_threshold: float = 8.0,
                 pos_window: float = 5.0,
                 prune_interval: float = 60.0,
                 min_node_count: int = 1):
        """Create a SparseGraphMemory.

        Args:
            modalities: list of modality names (each gets its own modal graph and inverted index)
            spatial_threshold: max distance (pixels or units) to consider a node match
            pos_window: time window (seconds) used for timestamp-based pruning decisions
            prune_interval: how often (s) to run pruning automatically (if enable_auto_prune True)
            min_node_count: minimal count to keep node during pruning
        """
        self.modalities = modalities
        self.spatial_threshold = spatial_threshold
        self.pos_window = pos_window
        self.prune_interval = prune_interval
        self.min_node_count = min_node_count

        self.graphs: Dict[str, SparseGraphMemory.ModalGraph] = {m: SparseGraphMemory.ModalGraph(m) for m in modalities}
        # inverted_index: modality -> key -> set(node_ids)
        self.inverted_index: Dict[str, Dict[Any, Set[int]]] = {m: {} for m in modalities}
        # node id allocator
        self._next_node_id = 1
        # locks per modality
        self.locks: Dict[str, bool] = {m: False for m in modalities}
        # last prune time
        self._last_prune = time.time()

    # ---------------------------
    # Helper utilities
    # ---------------------------
    def _dist(self, p1: Tuple[float,float], p2: Tuple[float,float]) -> float:
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return math.hypot(dx, dy)

    def _quantize_signature(self, modality: str, signature: Any) -> Any:
        """Default identity quantization. Override or pass hashed signatures externally.
        For continuous vectors, caller should quantize (e.g., binning or LSH) before calling.
        """
        return signature

    # ---------------------------
    # Core ops: retrieve and update
    # ---------------------------
    def retrieve(self, input_features: List[Dict], top_k: int = 10) -> Dict[str, List[Tuple[int, float]]]:
        """Retrieve candidate nodes per modality for given input features.

        input_features: list of dicts, each with fields:
           - modality: str
           - signature: hashable key (discrete) OR 'vector' provided
           - position: (x,y)
           - optionally 'vector' for ANN matching (not implemented by default)

        Returns dict: modality -> list of (node_id, score) sorted by score desc.
        Score combines signature match (binary) and spatial proximity.
        """
        candidates: Dict[str, Dict[int, float]] = {m: {} for m in self.modalities}

        for feat in input_features:
            modality = feat['modality']
            if modality not in self.modalities:
                continue
            sig = self._quantize_signature(modality, feat.get('signature'))
            pos = feat.get('position', (0.0, 0.0))

            postings = self.inverted_index.get(modality, {}).get(sig, set())
            # if postings empty, we still may want to return nearby nodes (optional)
            for nid in postings:
                node = self.graphs[modality].nodes.get(nid)
                if node is None:
                    continue
                d = self._dist(pos, node.position)
                spatial_score = max(0.0, 1.0 - d / (self.spatial_threshold + 1e-6))
                # signature match gives base score 1, spatial reduces by distance
                score = 1.0 * spatial_score
                # keep best score per node
                prev = candidates[modality].get(nid, 0.0)
                if score > prev:
                    candidates[modality][nid] = score

        # convert to sorted lists
        out: Dict[str, List[Tuple[int,float]]] = {}
        for m, cmap in candidates.items():
            lst = sorted(cmap.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
            out[m] = lst
        return out

    def update(self, input_features: List[Dict], allow_update: bool = True, pos_alpha: float = 0.2, create_if_missing: bool = True) -> None:
        """Update memory with given features. This runs recognition+update synchronously.

        For all active features in the same input, we also update edges between matched/created nodes.
        """
        # early exit if all modalities locked or updates disabled
        if not allow_update:
            return

        # per-modality matched node ids
        matched_nodes_per_mod: Dict[str, List[int]] = {m: [] for m in self.modalities}

        # first pass: match or create nodes per modality
        for feat in input_features:
            modality = feat['modality']
            if modality not in self.modalities:
                continue
            if self.locks.get(modality, False):
                # skip updates for this modality
                continue
            sig = self._quantize_signature(modality, feat.get('signature'))
            pos = feat.get('position', (0.0, 0.0))

            postings = self.inverted_index.get(modality, {}).get(sig, set())
            best_nid = None
            best_score = 0.0
            for nid in postings:
                node = self.graphs[modality].nodes.get(nid)
                if node is None:
                    continue
                d = self._dist(pos, node.position)
                if d <= self.spatial_threshold and node.count > 0:
                    # score by proximity and count
                    score = (1.0 - d / (self.spatial_threshold + 1e-6)) * (1.0 + math.log(1 + node.count))
                    if score > best_score:
                        best_score = score
                        best_nid = nid

            if best_nid is not None:
                # update existing node
                node = self.graphs[modality].nodes[best_nid]
                node.touch(pos, alpha=pos_alpha)
                matched_nodes_per_mod[modality].append(best_nid)
            else:
                # create new node
                if create_if_missing:
                    nid = self._next_node_id
                    self._next_node_id += 1
                    node = SparseGraphMemory.Node(nid, modality, sig, pos, vector=feat.get('vector'))
                    self.graphs[modality].add_node(node)
                    # update inverted index
                    inv = self.inverted_index[modality]
                    inv.setdefault(sig, set()).add(nid)
                    matched_nodes_per_mod[modality].append(nid)

        # second pass: update edges between simultaneously active nodes (across modalities)
        # build flat list of active node ids
        active_nodes: List[Tuple[str,int]] = []
        for mod, lst in matched_nodes_per_mod.items():
            for nid in lst:
                active_nodes.append((mod, nid))

        # for all pairs within active nodes, update edges inside the same modal graph or cross-modal?
        # Route A: we keep edges inside each modality's graph between nodes of same modality
        # and optionally track cross-modal co-occurrence as a separate structure.
        # Here implement both: intra-modal edges, and a cross_modal_cooccur dict.

        # intra-modal edges
        for i in range(len(active_nodes)):
            for j in range(i+1, len(active_nodes)):
                mod_i, nid_i = active_nodes[i]
                mod_j, nid_j = active_nodes[j]
                if mod_i == mod_j:
                    g = self.graphs[mod_i]
                    g.add_edge(nid_i, nid_j)
                    g.add_edge(nid_j, nid_i)
                else:
                    # cross-modal: record co-occurrence edge in both modal graphs as light-weight
                    # create pseudo-edge entries (we store them in both graphs' adj with negative dst ids offset?)
                    # simpler: maintain cross-modal map
                    pass

        # third: update cross-modal co-occurrence map
        # cross_modal_cooccur: (modA, nidA) -> dict( (modB,nidB) -> weight )
        if not hasattr(self, 'cross_modal_cooccur'):
            self.cross_modal_cooccur = {}

        for i in range(len(active_nodes)):
            for j in range(i+1, len(active_nodes)):
                a = active_nodes[i]
                b = active_nodes[j]
                key_a = (a[0], a[1])
                key_b = (b[0], b[1])
                self.cross_modal_cooccur.setdefault(key_a, {})
                self.cross_modal_cooccur.setdefault(key_b, {})
                self.cross_modal_cooccur[key_a].setdefault(key_b, 0)
                self.cross_modal_cooccur[key_b].setdefault(key_a, 0)
                self.cross_modal_cooccur[key_a][key_b] += 1
                self.cross_modal_cooccur[key_b][key_a] += 1

    # ---------------------------
    # Maintenance: pruning, lock control, serialization
    # ---------------------------
    def prune(self, now: Optional[float] = None) -> None:
        """Remove low-count or old nodes and weak edges."""
        if now is None:
            now = time.time()
        for mod, g in self.graphs.items():
            # remove nodes with count < min_node_count
            for nid, node in list(g.nodes.items()):
                age = now - node.last_seen
                if node.count < self.min_node_count or age > self.pos_window * 60:
                    g.remove_node(nid)
                    # remove from inverted index
                    inv = self.inverted_index[mod]
                    sig = node.signature
                    if sig in inv and nid in inv[sig]:
                        inv[sig].remove(nid)
                        if len(inv[sig]) == 0:
                            del inv[sig]
            # prune weak edges
            for nid, neigh in list(g.adj.items()):
                for nb, e in list(neigh.items()):
                    if e.weight <= 0:
                        del neigh[nb]

        # prune cross_modal_cooccur
        if hasattr(self, 'cross_modal_cooccur'):
            for k, v in list(self.cross_modal_cooccur.items()):
                for kk in list(v.keys()):
                    if v[kk] <= 0:
                        del v[kk]
                if len(v) == 0:
                    del self.cross_modal_cooccur[k]

    def lock_modality(self, modality: str) -> None:
        if modality in self.locks:
            self.locks[modality] = True

    def unlock_modality(self, modality: str) -> None:
        if modality in self.locks:
            self.locks[modality] = False

    # ---------------------------
    # Debug & inspection helpers
    # ---------------------------
    def inspect_modal(self, modality: str, limit: int = 10) -> Dict:
        """Return summary of a modality graph for debugging."""
        if modality not in self.modalities:
            return {}
        g = self.graphs[modality]
        nodes = list(g.nodes.values())[:limit]
        node_info = [{
            'id': n.id, 'sig': n.signature, 'pos': n.position, 'count': n.count
        } for n in nodes]
        edges = []
        for nid, neigh in list(g.adj.items())[:limit]:
            for nb, e in list(neigh.items())[:limit]:
                edges.append({'src': nid, 'dst': nb, 'w': e.weight})
        return {'nodes': node_info, 'edges': edges}


# ----------------------------
# Quick unit-test style demo function (not run automatically)
# ----------------------------

def demo_usage():
    mem = SparseGraphMemory(modalities=['edge','hue','flow','curv'], spatial_threshold=10.0)
    # synthetic features: each entry must have modality, signature, position
    feats = [
        {'modality': 'edge', 'signature': 'edge_v1', 'position': (10,10)},
        {'modality': 'hue',  'signature': 'hue_red',  'position': (12,11)},
        {'modality': 'curv', 'signature': 'curv_round','position': (11,9)}
    ]
    print('Retrieve before update:', mem.retrieve(feats))
    mem.update(feats, allow_update=True)
    print('Retrieve after update:', mem.retrieve(feats))
    print('Inspect edge modality:', mem.inspect_modal('edge'))

 
 