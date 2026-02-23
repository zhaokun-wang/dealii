# compile flag control

- **Created CMake Configuration File**
Added a new file named `configure_50_polydeal.cmake` (or `configure_50_agglomeration.cmake`) within the `cmake/configure/` directory.
- **Resolved External Dependency Misidentification**
Because the `CONFIGURE_FEATURE` macro defaults to searching for third-party external libraries, we injected an interception macro (`FEATURE_AGGLOMERATION_FIND_EXTERNAL` forced to return `TRUE`). This successfully compelled CMake to recognize it as a **pure internal feature**, resolving the "missing library" error.
- **Implemented Conditional Compilation**
    - **Build Scripts:** Wrapped the newly added source files (`.cc`) and template instantiation files (`.inst.in`) inside `if(DEAL_II_WITH_AGGLOMERATION)` blocks within the relevant `CMakeLists.txt` files.
    - **C++ Source & Headers:** Macro-isolated all class definitions and implementations using `#ifdef DEAL_II_WITH_AGGLOMERATION` after explicitly including the `<deal.II/base/config.h>` configuration header.