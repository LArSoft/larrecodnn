cet_enable_asserts()

set( nugraph_tool_lib_list
       lardataobj::RecoBase
       lardataobj::AnalysisBase
       larrecodnn_ImagePatternAlgs_NuSonic_Triton
       TorchScatter::TorchScatter
       art::Framework_Core
       hep_concurrency::hep_concurrency
)

cet_build_plugin(FilterDecoder art::tool
  LIBRARIES PRIVATE
  ${nugraph_tool_lib_list}
)

cet_build_plugin(SemanticDecoder art::tool
  LIBRARIES PRIVATE
  ${nugraph_tool_lib_list}
)

cet_build_plugin(StandardLoader art::tool
  LIBRARIES PRIVATE
  ${nugraph_tool_lib_list}
)

cet_build_plugin(VertexDecoder art::tool
  LIBRARIES PRIVATE
  ${nugraph_tool_lib_list}
)

install_headers()
install_fhicl()
install_source()
