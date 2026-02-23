MACRO(FEATURE_AGGLOMERATION_FIND_EXTERNAL var)
  SET(${var} TRUE)
ENDMACRO()

CONFIGURE_FEATURE(AGGLOMERATION
  "Enable support for Polytope/Agglomeration methods"
  ON
  )