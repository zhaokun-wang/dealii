# CMakeLists.txt changing

- **Resolved missing `.inst` files during compilation:**
    - Modified `source/grid/CMakeLists.txt` and `source/fe/CMakeLists.txt`.
    - Explicitly added the corresponding `.inst.in` files to the `_inst` list to ensure the `expand_instantiations` tool correctly captures and generates them.
- **Standardized include paths:**
    - Removed hardcoded `template class` instantiations.
    - Use standard deal.II syntax instead (e.g., `#include "fe/fe_agglodgp.inst"`).