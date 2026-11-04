"""Owned cluster: builds nodes from config, finds nodes that fit a job, allocates and frees resources."""
from sim.models import Node


def build_nodes(cfg: dict) -> list[Node]:
    nodes = []
    for c in cfg["cluster"]["node_classes"]:
        for i in range(c["count"]):
            nodes.append(Node(id=f"{c['name']}-{i}", cls=c["name"], cores=c["cores"],
                              mem_gb=c["mem_gb"], gpus=c["gpus"]))
    return nodes


def fits(node: Node, job) -> bool:
    return node.free_cores >= job.cores and node.free_mem >= job.mem_gb and node.free_gpus >= job.gpus


def best_fit(nodes: list[Node], job) -> Node | None:
    """Node left with the least spare cores after placing the job (ties: first node)."""
    candidates = [n for n in nodes if fits(n, job)]
    return min(candidates, key=lambda n: n.free_cores - job.cores, default=None)


def allocate(node: Node, job) -> None:
    node.free_cores -= job.cores
    node.free_mem -= job.mem_gb
    node.free_gpus -= job.gpus
    node.running.add(job.id)


def release(node: Node, job) -> None:
    node.free_cores += job.cores
    node.free_mem += job.mem_gb
    node.free_gpus += job.gpus
    node.running.discard(job.id)
