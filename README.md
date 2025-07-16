# LigandMPNN + FastRelax Integration

**Author:** David Hyunyoo Jang  
**Date:** 07/16/25  
**Updated:** Latest LigandMPNN implementation

## Overview

Iterative protein design pipeline combining LigandMPNN sequence generation with PyRosetta FastRelax structural optimization. This implementation updates the original development-stage LigandMPNN-FR code to work with the latest LigandMPNN API.

## Acknowledgments

- **LigandMPNN components**: Adapted from Justas's ligand_proteinmpnn implementation
- **MPNN-FR concept**: Inspired by Nate B and bcov's PPI MPNN-FR pipeline  
- **Original development**: Based on Gyu Rie Lee's ligMPNN_FR framework

## Dependencies

### Required Software
```bash
mamba create -n rfdaa -y \
    -c nvidia -c pytorch -c dglteam/label/th24_cu124 -c conda-forge -c defaults \
    python=3.12.11 \
    pytorch=2.4.1 \
    pytorch-cuda=12.4 \
    dgl \
    torchdata \
    hydra-core \
    omegaconf \
    scipy=1.14.1 \
    icecream \
    openbabel \
    assertpy \
    opt-einsum \
    pandas \
    pydantic \
    deepdiff \
    e3nn \
    fire \
    numpy \
    ml-collections \
    dm-tree
pip install -U git+https://github.com/jamesmkrieger/ProDy@unpin_numpy

# Core dependencies
pip install numpy prody

# PyRosetta (separate installation required)
# Download from: https://www.pyrosetta.org/downloads
```

### LigandMPNN Installation
```bash
# Ensure LigandMPNN is installed at:
# /home/hwjang/aipd/LigandMPNN/
# Or modify paths in scripts accordingly
```

## Key Differences

### Differences from Original LigandMPNN

This implementation differs from the standard LigandMPNN `run.py` in several key aspects:

#### 1. **Iterative Design Workflow**
- **Original LigandMPNN**: Single-shot sequence generation
- **This Implementation**: Multi-cycle iterative design with structural optimization

#### 2. **PyRosetta Integration**
- **Original LigandMPNN**: No structural refinement
- **This Implementation**: Fast relax after each sequence generation cycle

#### 3. **API Updates for Latest Version**
- **Key API Changes**: Updated `feature_dict` structure for LigandMPNN v_32_010_25
  ```python
  # Required additions for latest API
  feature_dict["batch_size"] = 1
  feature_dict["temperature"] = temperature
  ```
- **Model Initialization**: Updated ProteinMPNN parameters and checkpoint loading

#### 4. **Structure Threading**
- **Original LigandMPNN**: Outputs sequences only
- **This Implementation**: Threads sequences onto structures and performs structural optimization

### Differences from Gyu Rie Lee's Original Implementation

This implementation modernizes and extends Gyu Rie Lee's original LigandMPNN+FastRelax concept:

#### 1. **LigandMPNN Version Compatibility**
- **Original (Gyu Rie)**: Based on earlier LigandMPNN version with different API
- **This Implementation**: Updated for LigandMPNN v_32_010_25 with modern API

#### 2. **Code Architecture**
- **Original**: Single monolithic script with global variables
- **This Implementation**: 
  - Modular class-based design
  - Clean separation of concerns
  - Multiple implementation versions (simple, complete, experimental)

#### 3. **API Integration**
- **Original**: Used `protein_mpnn_utils` and older featurization methods
- **This Implementation**: 
  - Uses latest `data_utils` and `model_utils`
  - Updated `featurize()`, `parse_PDB()`, and `model.sample()` calls
  - Proper handling of `feature_dict` structure

#### 4. **Error Handling and Robustness**
- **Original**: Minimal error handling
- **This Implementation**: 
  - Comprehensive exception handling
  - Input validation
  - Graceful failure modes

#### 5. **Configuration and Flexibility**
- **Original**: Hardcoded parameters and limited options
- **This Implementation**: 
  - Extensive command-line interface
  - Flexible constraint handling
  - Multiple temperature and sampling options

#### 6. **Output Organization**
- **Original**: Basic file output
- **This Implementation**: 
  - Structured output directories
  - Comprehensive logging
  - Statistical analysis and reporting

#### 7. **PyRosetta Initialization**
- **Original**: Complex initialization with manual flag construction
- **This Implementation**: 
  - Streamlined initialization process
  - Better flag management
  - Improved ligand parameter handling

## Scripts

### 1. `simple_ligmpnn_fr.py`
**Purpose:** Clean, minimal implementation for testing and learning  
**Features:** Core LigandMPNN + FastRelax workflow  
**Recommended for:** Quick prototyping, understanding the algorithm

### 2. `ligandmpnn_fastrelax_complete.py` 
**Purpose:** Full-featured production version  
**Features:** All advanced options, comprehensive error handling  
**Recommended for:** Research applications, complex design tasks

### 3. `ligandmpnn_fastrelax_v2.py`
**Purpose:** Extended version with additional experimental features  
**Features:** Enhanced constraint handling, residue-specific controls  
**Status:** Development version

## Basic Usage

### Simple Version
```bash
python simple_ligmpnn_fr.py \
    --pdb_path input_complex.pdb \
    --ligand_params_path ligand.params \
    --out_folder output_dir \
    --n_cycles 3 \
    --temperature 0.1
```

### Complete Version
```bash
python ligandmpnn_fastrelax_complete.py \
    --pdb_path input_complex.pdb \
    --ligand_params_path ligand.params \
    --out_folder output_dir \
    --path_to_model_weights /path/to/ligandmpnn/models \
    --n_cycles 5 \
    --temperature 0.1 \
    --use_side_chain_context \
    --save_stats \
    --verbose
```

## Command Line Arguments

### Essential Arguments
| Argument | Type | Description |
|----------|------|-------------|
| `--pdb_path` | str | Input PDB file with protein-ligand complex |
| `--ligand_params_path` | str | Ligand parameters file (.params) |
| `--out_folder` | str | Output directory path |

### Design Parameters
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--n_cycles` | int | 3 | Number of design-relax iterations |
| `--temperature` | float | 0.1 | LigandMPNN sampling temperature |
| `--num_seq_per_target` | int | 1 | Sequences per target |

### Advanced Options
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use_side_chain_context` | flag | False | Include side chain atoms in context |
| `--use_genpot` | flag | False | Use general potential during relax |
| `--fixed_residues` | list | [] | Residues to keep fixed |
| `--redesigned_residues` | list | [] | Residues to redesign |
| `--omit_AAs` | str | "X" | Amino acids to exclude |
| `--seed` | int | 42 | Random seed |
| `--verbose` | flag | True | Detailed output |
| `--save_stats` | flag | False | Save detailed statistics |

### Model Configuration
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--path_to_model_weights` | str | auto | Path to LigandMPNN model directory |
| `--model_name` | str | ligandmpnn_v_32_010_25 | Model checkpoint name |
| `--checkpoint_path` | str | None | Direct path to model file |

## Output Structure

```
output_directory/
├── seqs/                           # Generated sequences
│   ├── input_cycle_1.fa           #   FASTA format with scores
│   ├── input_cycle_2.fa
│   └── input_cycle_3.fa
├── backbones/                      # Threaded structures
│   ├── input_cycle_1_threaded.pdb #   After sequence threading
│   ├── input_cycle_2_threaded.pdb
│   └── input_cycle_3_threaded.pdb
├── relaxed/                        # Final structures
│   ├── input_cycle_1_relaxed.pdb  #   After FastRelax
│   ├── input_cycle_2_relaxed.pdb
│   └── input_cycle_3_relaxed.pdb
└── stats/ (optional)              # Statistics files
    ├── input_cycle_1.json         #   Detailed metrics
    ├── input_cycle_2.json
    └── input_cycle_3.json
```

## Algorithm Workflow

```
Input: Protein-ligand complex PDB + ligand params
    ↓
┌─→ [Cycle N] ←─┐
│   │            │
│   ├── LigandMPNN sequence generation
│   │   ├── Parse PDB structure
│   │   ├── Featurize with ligand context  
│   │   ├── Sample sequences with temperature
│   │   └── Score and select best sequence
│   │
│   ├── Sequence threading
│   │   ├── Load backbone structure
│   │   ├── Mutate residues to new sequence
│   │   └── Save threaded structure
│   │
│   ├── PyRosetta FastRelax
│   │   ├── Setup scoring function
│   │   ├── Apply constraints (if any)
│   │   ├── Run FastRelax protocol
│   │   └── Save relaxed structure
│   │
│   └── Update input for next cycle
│       └──────────────────┘
```

## Example Commands

### Basic Design
```bash
# Simple 3-cycle design
python simple_ligmpnn_fr.py \
    --pdb_path complex.pdb \
    --ligand_params_path ligand.params \
    --n_cycles 3
```

### Focused Redesign
```bash
# Design specific binding site residues
python ligandmpnn_fastrelax_complete.py \
    --pdb_path complex.pdb \
    --ligand_params_path ligand.params \
    --redesigned_residues "A45 A46 A48 A92 A95" \
    --temperature 0.15 \
    --n_cycles 5 \
    --use_side_chain_context \
    --save_stats
```

### High-throughput Design
```bash
# Generate multiple sequences with statistical analysis
python ligandmpnn_fastrelax_complete.py \
    --pdb_path complex.pdb \
    --ligand_params_path ligand.params \
    --num_seq_per_target 3 \
    --temperature 0.2 \
    --n_cycles 4 \
    --omit_AAs "CP" \
    --verbose \
    --save_stats
```

## Key Features

### Latest LigandMPNN Integration
- **Updated API**: Uses current LigandMPNN featurize() and model.sample() methods
- **Proper feature_dict**: Includes all required keys (batch_size, temperature, bias, symmetry)
- **Error handling**: Robust integration with comprehensive debugging

### Enhanced Design Control
- **Residue masking**: Selective design of binding site vs. scaffold regions
- **AA composition**: Bias or omit specific amino acids
- **Temperature control**: Fine-tune sequence diversity
- **Iterative refinement**: Multiple design-relax cycles

### Comprehensive Output
- **Sequence tracking**: FASTA files with scores for each cycle
- **Structure evolution**: PDB files for threading and relaxation stages  
- **Statistics**: JSON metadata with detailed metrics (optional)
- **Organized directories**: Clean separation of different output types

## Troubleshooting

### Common Issues

**1. LigandMPNN Import Errors**
```bash
# Check LigandMPNN installation path
ls /home/hwjang/aipd/LigandMPNN/
# Verify data_utils.py and model_utils.py exist
```

**2. PyRosetta Initialization Failed**
```bash
# Check PyRosetta installation
python -c "import pyrosetta; pyrosetta.init()"
# Verify ligand params file format
```

**3. CUDA Out of Memory**
```bash
# Use CPU mode
CUDA_VISIBLE_DEVICES='' python simple_ligmpnn_fr.py ...
# Or reduce batch size in advanced scripts
```

**4. Model Checkpoint Not Found**
```bash
# Check model path
ls /home/hwjang/aipd/LigandMPNN/model_params/
# Or specify direct path
--checkpoint_path /path/to/ligandmpnn_v_32_010_25.pt
```

## Performance Notes

### Timing Expectations
- **LigandMPNN**: ~10-30 seconds per sequence (GPU)
- **FastRelax**: ~30-120 seconds per structure (CPU)
- **Total per cycle**: ~1-3 minutes for typical complexes

### Resource Requirements
- **GPU memory**: 2-8 GB (depending on complex size)
- **RAM**: 4-16 GB (for PyRosetta operations)
- **Storage**: ~50-200 MB per design cycle

## Development Notes

### Code Architecture
- **Modular design**: Separate setup functions for LigandMPNN and PyRosetta
- **Error resilience**: Comprehensive exception handling and validation
- **Extensibility**: Easy to add new features or modify protocols

### Recent Updates
- **API compatibility**: Updated for LigandMPNN v_32_010_25
- **Bug fixes**: Resolved KeyError issues with feature_dict
- **Performance**: Optimized memory usage and file I/O
- **Documentation**: Enhanced user guide and troubleshooting

### Future Enhancements
- **Constraint support**: Distance and angle constraints (TODO)
- **Multi-chain design**: Complex multi-protein systems
- **Custom scoring**: User-defined scoring functions
- **Batch processing**: High-throughput pipeline automation
