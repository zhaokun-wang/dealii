# poisson code change

## File Modification Checklist (English Version)

| **File Name** | **Modification Type** | **Specific Content** | **Line Numbers** |
| --- | --- | --- | --- |
| `agglomeration_handler.h` | Delete methods | `define_agglomerate_with_check` | 293 |
|  |  | `print_agglomeration` | 363 |
|  |  | `get_mesh_size` | 315 |
|  |  | `n_agglomerated_faces_per_cell` | 391 |
|  | Delete hierarchy management | `connect_hierarchy`function | 529, 1184 |
|  |  | `parent_child_info`member variable | 873 |
| `agglomeration_handler.cc` | Delete implementation | `define_agglomerate_with_check` | 110 |
|  |  | `print_agglomeration` |  |
|  |  | `get_mesh_size` |  |
|  |  | `n_agglomerated_faces_per_cell` | 178 |
| `agglomeration_accessor.h` | Delete hierarchy navigation | `parent()` | 178, 779 |
|  |  | `child(unsigned int i)` |  |
|  |  | `n_children()` |  |
|  |  | `level()` |  |
|  |  | `has_children()` |  |
| `poly_utils.h` | Simplify functions (keep only`interpolate_to_fine_grid`,`compute_global_error`,`constexpr_pow`) | `partition_locally_owned_regions` | 555, 629 |
|  |  | `compute_quality_metrics` | 711 |
|  |  | `write_to_matrix_market_format` | 906 |
|  |  | `construct_agglomerated_levels` | 1571 |
|  |  | `assemble_dg_matrix` | 1810 |
|  |  | `assemble_dg_matrix_on_standard_mesh` | 2014 |
|  |  | `assemble_local_jumps_and_averages` | 1680 |
|  |  | `assemble_local_jumps_and_averages_ghost` | 1740 |
|  |  | `compute_h_orthogonal` | 401 |
|  |  | `Rtree_visitor` | 149, 241, 337, 265 |
|  |  | `extract_children_of_level` | 359 |
|  |  | `fill_interpolation_matrix` | 1279 |
| `fe_agglodgp.h/.cc` | **Keep unchanged** | Keep all | - |
| `mapping_box.h/.cc` | **Keep unchanged** | Keep all | - |
| `agglomerator.h` | Delete hierarchy relationships | Delete all`parent_node_to_children_nodes`related content | 135, 160, 236, 284, 372, 428, 464, 340 |
| `utils.h` | Evaluate deletion | Can be completely deleted if`agglomeration_handler`no longer depends on it |  |

## Reduction Principles Summary

1. **FE and Mapping**: Do not change a single character, this is the minimal complete unit
2. **Iterator/Accessor**: Delete all `parent`, `child`, `level` related navigation functions, keep only topology queries (neighbors, faces) and geometric queries (indices, DoF, diameter)
3. **Handler**: Delete `connect_hierarchy` and its data structures
4. **Utility functions**: Keep only functions actually called by `poisson.cc`