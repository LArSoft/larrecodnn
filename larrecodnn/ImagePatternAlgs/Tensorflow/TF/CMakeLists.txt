cet_make_library(SOURCE
  tf_graph.cc
  LIBRARIES PRIVATE
  TensorFlow::cc
  TensorFlow::framework
)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
    CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "12.5")
  # GCC 12.5 (and above) aggressively/spuriously warns about array-bounds issues.
  target_compile_options(larrecodnn_ImagePatternAlgs_Tensorflow_TF
                         PRIVATE "-Wno-sign-compare")
endif()
install_headers()
install_source()
