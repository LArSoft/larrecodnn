include (lar::PointIdAlgorithm)

cet_build_plugin(PointIdAlgSonicTriton lar::PointIdAlgorithm
  LIBRARIES PRIVATE
  larrecodnn_ImagePatternAlgs_NuSonic_Triton
  messagefacility::MF_MessageLogger
  fhiclcpp::types
  fhiclcpp::fhiclcpp
)

cet_build_plugin(PointIdAlgTriton lar::PointIdAlgorithm
  LIBRARIES PRIVATE
  TritonClient::grpcclient
  gRPC::grpc # Should be a transitive (INTERFACE) dependency of TritonClient::grpcclient
  messagefacility::MF_MessageLogger
  fhiclcpp::types
  fhiclcpp::fhiclcpp
  cetlib_except::cetlib_except
)
cet_build_plugin(WireframeRecogTriton art::tool
  LIBRARIES PRIVATE
    TritonClient::grpcclient
    gRPC::grpc   #larrecodnn_ImagePatternAlgs_NuSonic_Triton
    messagefacility::MF_MessageLogger
    fhiclcpp::types
    fhiclcpp::fhiclcpp
    cetlib_except::cetlib_except
    canvas::canvas
)

install_source()
