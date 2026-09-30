# Cascade Failure Simulator

A discrete-time simulator of outage spread across a microservice dependency graph. A failed service can take down callers that depend on it, until the graph stops changing. The run writes a blast-radius figure to `blast_radius.png`.

An edge `A → B` means A depends on B. Starting from a patient-zero node, each step gives every healthy dependent a chance to fail.

## Requirements

- Python 3.10 or newer
- `networkx`
- `matplotlib`

```bash
pip install networkx matplotlib
```

## Run

```bash
python cascade_failure.py
```
