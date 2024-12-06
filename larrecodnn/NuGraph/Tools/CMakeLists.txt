cet_enable_asserts()

set( nugraph_tool_lib_list
       lardataobj::RecoBase
       lardataobj::AnalysisBase
       larrecodnn_ImagePatternAlgs_NuSonic_Triton
       TorchScatter::TorchScatter
       art::Framework_Core
       hep_concurrency::hep_concurrency
)
art_make(TOOL_LIBRARIES ${nugraph_tool_lib_list} )

install_headers()
install_fhicl()
install_source()

