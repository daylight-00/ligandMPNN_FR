# Original LigandMPNN-FR Script

This directory contains the original LigandMPNN-FR script by Gyu Rie Lee (2022).

## Differences from Current Implementation

The current implementation includes several key improvements and updates compared to the original script:

### 1. **Per-Cycle Relaxation**
- **Original**: Single final relaxation step after all cycles
- **Current**: Structural relaxation within each cycle, enabling gradual backbone optimization

### 2. **Updated LigandMPNN API**
- Updated from development-stage to current stable API
- Uses `data_utils` and `model_utils` with proper `feature_dict`
- Compatible with LigandMPNN v_32_010_25
- Original used early development version

### 3. **Code Improvements**

#### Distance Constraint Parsing
- **Original**: Manual string parsing (89 lines)
- **Current**: BioPython + NumPy vectorized calculations (51 lines)
- More robust and maintainable code

#### Performance Enhancements
- **Parallel processing**: Multiprocessing support for FastRelax optimization
- **Vectorized calculations**: NumPy-based operations for better performance
- **Better error handling**: More informative error messages and validation

#### Code Quality
- Cleaner structure with better separation of concerns
- Improved documentation and type hints
- More configurable command-line options

### Algorithm Differences

| Aspect | Original | Current Implementation |
|--------|----------|----------------------|
| **Relaxation** | Final step only | Per-cycle relaxation |
| **Backbone** | Static until final step | Gradual optimization |
| **API** | Development version | Current stable API |
| **Parsing** | Manual string manipulation | BioPython-based |
| **Performance** | Single-threaded | Multiprocessing support |

### Why These Changes Matter

1. **Per-cycle relaxation**: Allows the backbone to gradually improve throughout the optimization process, leading to better final structures
2. **Updated API**: Ensures compatibility with current LigandMPNN releases and benefits from recent improvements
3. **Code quality**: Makes the codebase easier to maintain, extend, and debug

## Reference

Original concept and implementation by Gyu Rie Lee (2022).
