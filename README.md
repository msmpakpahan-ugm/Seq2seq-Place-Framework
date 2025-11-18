# Seq2seq-Place-Framework

This directory contains the core library modules for the MOFPSS-Seq2seq project, which implements sequence-to-sequence models and genetic algorithms for Fog Application Placement Problem (FAPP) and Multi-Objective Fog Service Placement Problem (MOFSPP).

## Table of Contents

- [Sequence-to-Sequence Models](#sequence-to-sequence-models)
- [Genetic Algorithms](#genetic-algorithms)
- [Placement Utilities](#placement-utilities)
- [Preprocessing Utilities](#preprocessing-utilities)
- [YAFS Integration](#yafs-integration)
- [Other Utilities](#other-utilities)

---

## Sequence-to-Sequence Models

Seq2seq models for learning optimal module placement strategies using neural networks.

### Core Models

- **[FAPP_seq2seqV4.md](FAPP_seq2seqV4.md)** - Basic seq2seq implementation with support for custom, BPE, and Unigram tokenizers. Foundation for all other versions.

### Enhanced Versions with External Context

- **[FAPP_seq2seqV7_RPEC.md](FAPP_seq2seqV7_RPEC.md)** - Version 7 with Resource Placement with External Context (RPEC). Adds external context validation during decoding.

- **[FAPP_seq2seqV8_RPEC.md](FAPP_seq2seqV8_RPEC.md)** - Version 8 with improved RPEC implementation. Enhanced context management and error handling.

- **[FAPP_seq2seqV8_RPECv2.md](FAPP_seq2seqV8_RPECv2.md)** - Version 2 of V8 RPEC with additional refinements and bug fixes.

### Version 9 Models (Latest)

- **[FAPP_seq2seqV9_E3RP.md](FAPP_seq2seqV9_E3RP.md)** - Enhanced 3-Resource Placement strategy. Optimizes placement across three key resources.

- **[FAPP_seq2seqV9_ECRP.md](FAPP_seq2seqV9_ECRP.md)** - Energy-Conscious Resource Placement. Optimizes placement considering energy consumption.

- **[FAPP_seq2seqV9_EDRP.md](FAPP_seq2seqV9_EDRP.md)** - Energy-Delay Resource Placement. Balances energy consumption and response time with multi-objective optimization.

**Recommended:** For most use cases, start with `FAPP_seq2seqV9_ECRP.py` or `FAPP_seq2seqV9_EDRP.py` as they include the latest features and optimizations.

---

## Genetic Algorithms

Genetic algorithm implementations for placement optimization using evolutionary computation.

- **[GAv3_FAPP.md](GAv3_FAPP.md)** - Basic genetic algorithm for Fog Application Placement Problem.

- **[GAv5_FAPP.md](GAv5_FAPP.md)** - Version 5 with additional fitness functions and improved optimization strategies.

- **[GAv6_MOFSPP.md](GAv6_MOFSPP.md)** - Multi-Objective Fog Service Placement Problem solver. Implements multi-objective optimization (Response Time, Cost, Energy) with Pareto-optimal solution generation.

- **[GAv7_MOFSPP_histnorm.md](GAv7_MOFSPP_histnorm.md)** - Version 7 with histogram normalization for improved fitness scaling in multi-objective optimization.

**Recommended:** Use `GAv6_MOFSPP.py` for multi-objective optimization scenarios.

---

## Placement Utilities

Utilities for converting model predictions to placement configurations and validating placements.

- **[prediction_to_placement.md](prediction_to_placement.md)** - Converts seq2seq model predictions (CSV) to YAFS placement JSON format. Validates resource constraints (Storage) and handles placement failures.

- **[prediction_to_placement_EC.md](prediction_to_placement_EC.md)** - Extended version with External Context (EC) validation. Includes additional constraint checking and improved error handling.

- **[hopaware_placement.md](hopaware_placement.md)** - Hop-aware placement algorithm that considers network hop distance in placement decisions.

**Recommended:** Use `prediction_to_placement_EC.py` for production deployments with enhanced validation.

---

## Preprocessing Utilities

Data preprocessing and conversion utilities for preparing training data and working with JSON files.

- **[preprocess_4seq2seq.md](preprocess_4seq2seq.md)** - Preprocessing utilities for seq2seq model training. Converts JSON data to DataFrames, extracts node and application specifications, and creates training data format.

---

## YAFS Integration

Python utilities for working with YAFS (Yet Another Fog Simulator) simulation data and file generation.

### YAFS Utilities

- **[pYAFS_trinity.md](pYAFS_trinity.md)** - Main Python wrapper for YAFS Trinity. Provides helper functions for working with YAFS simulation data, including JSON to DataFrame conversion utilities.

- **[pYAFS_trinity_before.md](pYAFS_trinity_before.md)** - Previous version of pYAFS_trinity utilities (legacy).

- **[pYAFS_trinity_energy.md](pYAFS_trinity_energy.md)** - Energy-aware version with energy consumption calculations and tracking.

- **[pYAFSv3.md](pYAFSv3.md)** - Version 3 of YAFS Python utilities (legacy).

### File Generation

- **[YAFSfilegenerationv2.md](YAFSfilegenerationv2.md)** - YAFS file generation utilities. Creates YAFS-compatible JSON files for applications, topology, and placements. Includes network graph generation and centrality calculations.

**Recommended:** Use `pYAFS_trinity.py` for current implementations, or `pYAFS_trinity_energy.py` if energy tracking is needed.

---

## Other Utilities

Additional utility modules for specialized tasks.

*(Currently no additional utilities documented)*

---

## Quick Start Guide

### For Seq2seq Models

1. Start with `FAPP_seq2seqV9_ECRP.py` or `FAPP_seq2seqV9_EDRP.py` for the latest features
2. Use `preprocess_4seq2seq.py` to prepare your training data
3. Train the model using the provided training functions
4. Use `prediction_to_placement_EC.py` to convert predictions to placement JSON

### For Genetic Algorithms

1. Use `GAv6_MOFSPP.py` for multi-objective optimization
2. Initialize with your topology, application, and population data
3. Run evolution for desired number of generations
4. Extract Pareto-optimal solutions

### For YAFS Integration

1. Use `YAFSfilegenerationv2.py` to generate YAFS-compatible files
2. Use `pYAFS_trinity.py` to process simulation results
3. Use `pYAFS_trinity_energy.py` if energy metrics are needed

---

## Dependencies

### Core Dependencies
- `torch` - PyTorch for neural network models
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `networkx` - Graph and network analysis
- `json` - JSON file handling

### Additional Dependencies
- `tokenizers` - For BPE and Unigram tokenization
- `matplotlib` - For visualization (some modules)

---

## Version History

### Seq2seq Models
- **V4**: Basic implementation with multiple tokenizer support
- **V7**: Added external context validation (RPEC)
- **V8**: Improved RPEC with better error handling
- **V9**: Latest versions with specialized strategies (E3RP, ECRP, EDRP)

### Genetic Algorithms
- **V3**: Basic GA implementation
- **V5**: Enhanced fitness functions
- **V6**: Multi-objective optimization (MOFSPP)
- **V7**: Histogram normalization for fitness scaling

---

## Contributing

When adding new modules to this library:

1. Create a corresponding `.md` documentation file
2. Update this README.md with a link to the new documentation
3. Include usage examples and parameter descriptions
4. Document dependencies and requirements

---

## Notes

- All seq2seq models use CUDA if available, otherwise fall back to CPU
- Genetic algorithms support both single and multi-objective optimization
- YAFS utilities are compatible with YAFS Trinity simulator
- Most modules include error handling and validation

---

## Contact

Michael Moses, msmpakpahan@gmail.com
