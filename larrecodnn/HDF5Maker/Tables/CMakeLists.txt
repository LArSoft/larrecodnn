include_directories($ENV{HEP_HPC_INC})

cet_set_compiler_flags(DIAGS VIGILANT DWARF_VER 4
  WERROR
  NO_UNDEFINED
  EXTRA_FLAGS -pedantic)

cet_make_library(SOURCE
  Table.h
  EDep.cxx
  Event.cxx
  Hit.cxx
  Neutrino.cxx
  LIBRARIES PRIVATE
  art::Framework_Principal
  art::Framework_Services_Registry
  nusimdata::SimulationBase
  larsim::MCCheater_BackTrackerService_service
  lardata::DetectorInfoServices_DetectorClocksServiceStandard_service
  ${HEP_HPC_HDF5}
  HDF5::HDF5
)

install_headers()
install_source()

# cet_make_library(SOURCE
#   Event.cxx
#   Table.h
#   LIBRARIES
#   PUBLIC
#   nusimdata::SimulationBase
#   art::Framework_Principal
#   ROOT::Hist
#   PRIVATE
#   ${HEP_HPC_HDF5}
#   HDF5::HDF5
#   lardataobj::RecoBase
#   larsim::MCCheater_BackTrackerService_service
#   larsim::MCCheater_ParticleInventoryService_service
#   art::Framework_Services_Registry
#   art::Persistency_Provenance
#   art::Persistency_Common
#   art::Utilities
#   messagefacility::MF_MessageLogger
#   canvas::canvas
# )
