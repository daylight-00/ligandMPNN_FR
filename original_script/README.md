# Original GRL LigandMPNN-FR Script

This directory preserves Prof. Gyu Rie Lee’s original 2022 LigandMPNN-FR script as the historical baseline for this project.

The original script is important because it already introduced the core LigandMPNN–Rosetta recycling idea:

```text
LigandMPNN sequence design
→ Rosetta threading / repacking / minimization
→ recycle Rosetta-processed PDBs into later LigandMPNN cycles
```

The main repository builds on this earlier workflow and reframes it as **metric-guided directional search over backbone-conditioned sequence sampling spaces**.

## What the original script did

The original script used a development-stage LigandMPNN implementation and PyRosetta to run iterative design-refinement cycles for protein-ligand complexes.

At a high level:

```text
Input protein-ligand PDB
        ↓
LigandMPNN sequence sampling
        ↓
PyRosetta SimpleThreadingMover
        ↓
RosettaScripts bsite repack/min
        ↓
Recycle output PDBs into next LigandMPNN cycle
        ↓
Final cycle FastRelax
```

Important implementation details:

- LigandMPNN generated sequences from the current input PDB.
- PyRosetta `SimpleThreadingMover` applied each sampled sequence to the input pose.
- Intermediate cycles used a binding-site repack/min protocol.
- The final cycle switched to a FastRelax protocol.
- Rosetta output PDBs were used as the next cycle's LigandMPNN inputs.
- The ligand-distance constraint file was generated from the initial input structure and reused through the workflow.
- Multiple output trajectories could propagate across cycles.

This means the original script was not simply a "final relax only" pipeline. It already performed Rosetta cleanup during intermediate cycles; the key distinction is that intermediate cycles used bsite repack/min, while the final cycle used FastRelax.

## How the current implementation differs

The current implementation is not just an API update. The main conceptual change is that each cycle now performs candidate-pool evaluation and metric-guided state selection.

```text
Original:
  recycle Rosetta-processed LigandMPNN outputs

Current:
  generate a candidate pool
  → FastRelax every candidate
  → evaluate post-relaxation metrics
  → recycle only the best relaxed state
```

## Comparison

| Aspect | Original GRL script | Current implementation |
|---|---|---|
| Core idea | LigandMPNN–Rosetta recycling | Metric-guided LigandMPNN–FastRelax recycling |
| LigandMPNN version | Development-stage LigandMPNN API | Stock/current LigandMPNN-style API |
| Structure generation | PyRosetta `SimpleThreadingMover` | LigandMPNN structure writing with optional side-chain packing |
| Intermediate Rosetta step | Bsite repack/min | FastRelax for every candidate in every cycle |
| Final Rosetta step | FastRelax in the last cycle | Same FastRelax-style evaluation throughout cycles |
| Candidate handling | Multiple trajectories may propagate | Best relaxed state selected per cycle |
| Selection signal | MPNN score sorting; Rosetta metrics mainly recorded | Configurable metric selects the recycled state |
| Default steering metric | Not an explicit per-cycle selector | `ddg` by default |
| Constraint reference | Initial input structure anchored | Candidate/current-state constraint generation inside worker |
| Search view | Recycling/refinement of trajectories | Directional optimization of backbone-conditioned sequence space |
| Parallelization | Serial-style loop | Parallel FastRelax over candidate pool |

## Conceptual reframing

The original script established that LigandMPNN outputs can be structurally processed by Rosetta and recycled into future design cycles.

The current implementation asks a different question:

> Can post-relaxation metrics identify a backbone-ligand state that will make the next LigandMPNN sampling round more productive?

This reframes candidate generation from fixed-backbone oversampling into an optimization loop over the input state that defines LigandMPNN's conditional sequence distribution.

## Why this matters

LigandMPNN sequence sampling is conditioned on the input backbone-ligand geometry. Sampling more sequences from the same fixed structure mainly explores the same local sequence distribution.

The current workflow uses diversity within each cycle to choose a direction:

```text
stochastic sequence proposals
        +
post-FastRelax metric selection
        ↓
directional update of the next input state
```

In this framing, Rosetta FastRelax is not only a final cleanup step. It provides the relaxed structures and metrics used to decide which state should define the next LigandMPNN sampling space.

## Technical updates

The current implementation also includes several engineering updates:

- compatibility with the current LigandMPNN-style `data_utils`, `model_utils`, and `feature_dict` workflow;
- optional LigandMPNN side-chain packing before Rosetta refinement;
- multiprocessing for FastRelax over multiple candidates;
- configurable selection metrics;
- JSON statistics for per-cycle analysis;
- more explicit output organization.

## Notes

The two workflows should be compared by their actual state-transition logic, not only by the presence or absence of Rosetta relaxation.

A concise distinction is:

```text
Original script:
  iterative recycling with bsite repack/min in intermediate cycles
  and FastRelax in the final cycle

Current script:
  per-cycle candidate-pool FastRelax
  followed by metric-guided best-state recycling
```

## Reference

Original concept and implementation by Gyu Rie Lee, 2022.
