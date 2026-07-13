"""
# Microservice Cascade Failure Simulator

## What it is

A graph-theory dependency analyzer tracking error propagation across directed
networks:

- **Nodes** - Microservices in the architecture.
- **Directed edges** - Dependency / call relationships (`A → B` means A depends
  on B). When B fails, A is exposed to timeouts, exhausted connection pools, or
  missing fallbacks.
- **Discrete-time cascade** - Starting from a parameterized patient-zero node,
  each step gives every healthy service that depends on a failed upstream a
  chance to fail with probability `failure_probability`, until no new failures
  occur (stable state).

## What it is used for

High-availability engineering, chaos engineering (like Netflix's Chaos Monkey),
and determining architectural single points of failure:

- **Blast-radius analysis** - See how far an outage spreads before mitigations
  exist.
- **Chaos experiments** - Rehearse killing a core dependency and inspect the
  resulting graph visually.
- **Architecture review** - Highlight hubs whose failure endangers many
  dependents.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

EdgeSpec = Union[
    Tuple[Hashable, Hashable],
    Tuple[Hashable, Hashable, Mapping[str, Any]],
]


def build_graph(
    nodes: Iterable[Hashable],
    edges: Sequence[EdgeSpec],
) -> nx.DiGraph:
    """
    Build a directed dependency graph.

    Edge form: ``(source, target)`` or ``(source, target, {"traffic": n, ...})``.
    Convention: ``source → target`` means *source depends on target*.
    """
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    for edge in edges:
        if len(edge) == 2:
            u, v = edge
            attrs: Dict[str, Any] = {"traffic": 1.0}
        elif len(edge) == 3:
            u, v, attrs = edge  # type: ignore[misc]
            attrs = dict(attrs)
            attrs.setdefault("traffic", 1.0)
        else:
            raise ValueError(f"Invalid edge spec: {edge!r}")
        g.add_edge(u, v, **attrs)
    return g


def simulate_cascade(
    graph: nx.DiGraph,
    patient_zero: Hashable,
    failure_probability: float,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Propagate failures from ``patient_zero`` until the system stabilizes.

    A healthy service that depends on (has an edge toward) any failed service
    fails in the current step with probability ``failure_probability``.
    """
    if patient_zero not in graph:
        raise ValueError(f"patient_zero {patient_zero!r} is not in the graph")
    if not 0.0 <= failure_probability <= 1.0:
        raise ValueError("failure_probability must be in [0, 1]")

    rng = random.Random(seed)
    failed = {patient_zero}
    status: Dict[Hashable, str] = {n: "healthy" for n in graph.nodes}
    status[patient_zero] = "patient_zero"
    history: List[List[Hashable]] = [[patient_zero]]
    steps = 0

    while True:
        steps += 1
        newly_failed: List[Hashable] = []
        for node in graph.nodes:
            if node in failed:
                continue
            # node → dep means node depends on dep; cascade from failed deps.
            depends_on_failed = any(dep in failed for dep in graph.successors(node))
            if depends_on_failed and rng.random() < failure_probability:
                newly_failed.append(node)

        if not newly_failed:
            break

        for node in newly_failed:
            failed.add(node)
            status[node] = "cascaded"
        history.append(newly_failed)

    return {
        "failed_nodes": failed,
        "status": status,
        "history": history,
        "steps": steps,
        "patient_zero": patient_zero,
        "failure_probability": failure_probability,
    }


def plot_blast_radius(
    graph: nx.DiGraph,
    status: Mapping[Hashable, str],
    output_path: str = "blast_radius.png",
    title: Optional[str] = None,
    layout_seed: int = 42,
) -> None:
    """Render the architecture with healthy / patient-zero / cascaded colors."""
    color_map = {
        "healthy": "#2f9e44",
        "patient_zero": "#e03131",
        "cascaded": "#f76707",
    }
    node_colors = [color_map.get(status.get(n, "healthy"), "#868e96") for n in graph.nodes]

    traffics = [float(graph.edges[e].get("traffic", 1.0)) for e in graph.edges]
    if traffics:
        t_min, t_max = min(traffics), max(traffics)
        if t_max == t_min:
            widths = [3.0] * len(traffics)
        else:
            widths = [1.0 + 5.0 * (t - t_min) / (t_max - t_min) for t in traffics]
    else:
        widths = []

    pos = nx.spring_layout(graph, seed=layout_seed, k=1.8)
    fig, ax = plt.subplots(figsize=(11, 8))

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        width=widths,
        edge_color="#adb5bd",
        arrows=True,
        arrowsize=18,
        connectionstyle="arc3,rad=0.06",
        alpha=0.85,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=1600,
        edgecolors="black",
        linewidths=1.0,
    )
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8, font_weight="bold")

    legend_handles = [
        mpatches.Patch(color=color_map["healthy"], label="Healthy"),
        mpatches.Patch(color=color_map["patient_zero"], label="Patient zero"),
        mpatches.Patch(color=color_map["cascaded"], label="Cascaded failure"),
    ]
    ax.legend(handles=legend_handles, loc="best")
    ax.set_title(title or "Microservice cascade blast radius")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved blast radius map to {output_path}")
    plt.show()


def print_cascade_log(result: Mapping[str, Any]) -> None:
    """Print discrete-time failure history to the console."""
    print("=" * 60)
    print("Cascade simulation log")
    print("=" * 60)
    print(f"Patient zero: {result['patient_zero']}")
    print(f"Failure probability: {result['failure_probability']}")
    for step, nodes in enumerate(result["history"]):
        label = "t=0 (initial)" if step == 0 else f"t={step}"
        print(f"{label}: failed → {', '.join(str(n) for n in nodes)}")
    healthy = [n for n, s in result["status"].items() if s == "healthy"]
    print("-" * 60)
    print(f"Steps until stable: {result['steps']}")
    print(f"Total failed: {len(result['failed_nodes'])}")
    print(f"Still healthy: {', '.join(str(n) for n in healthy) or '(none)'}")
    print("=" * 60)


def build_sample_architecture() -> nx.DiGraph:
    """Programmatic sample: API Gateway fronting Auth, Payment, Inventory, Notification."""
    nodes = [
        "API Gateway",
        "Auth",
        "Payment",
        "Inventory",
        "Notification",
        "Database",
        "Cache",
    ]
    edges: List[EdgeSpec] = [
        ("API Gateway", "Auth", {"traffic": 90}),
        ("API Gateway", "Payment", {"traffic": 70}),
        ("API Gateway", "Inventory", {"traffic": 65}),
        ("API Gateway", "Notification", {"traffic": 40}),
        ("Auth", "Cache", {"traffic": 80}),
        ("Auth", "Database", {"traffic": 35}),
        ("Payment", "Database", {"traffic": 100}),
        ("Inventory", "Database", {"traffic": 95}),
        ("Notification", "Cache", {"traffic": 25}),
        ("Payment", "Notification", {"traffic": 30}),
    ]
    return build_graph(nodes, edges)


if __name__ == "__main__":
    architecture = build_sample_architecture()
    patient_zero = "Database"
    failure_probability = 0.55

    result = simulate_cascade(
        graph=architecture,
        patient_zero=patient_zero,
        failure_probability=failure_probability,
        seed=42,
    )
    print_cascade_log(result)

    cascaded = sum(1 for s in result["status"].values() if s == "cascaded")
    plot_blast_radius(
        graph=architecture,
        status=result["status"],
        output_path="blast_radius.png",
        title=(
            f"Blast radius after {patient_zero} failure "
            f"({cascaded} cascaded, {result['steps']} steps)"
        ),
    )
