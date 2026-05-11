add_subdirectory(Tables)

include_directories($ENV{HEP_HPC_INC})

cet_build_plugin(HDF5Maker art::EDAnalyzer
                 LIBRARIES PRIVATE
                 larrecodnn_HDF5Maker_Tables
                 ${HEP_HPC_HDF5}
                 HDF5::HDF5
)

install_headers()
install_fhicl()
install_source()

