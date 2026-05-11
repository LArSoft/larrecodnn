include_directories($ENV{HEP_HPC_INC})

cet_set_compiler_flags(DIAGS VIGILANT DWARF_VER 4
  WERROR
  NO_UNDEFINED
  EXTRA_FLAGS -pedantic)

cet_make_library(SOURCE
  ChargeHit.cxx
  EDep.cxx
  Event.cxx
  Flash.cxx
  LDep.cxx
  Neutrino.cxx
  OpHit.cxx
  Particle.cxx
  Spacepoint.cxx
  LIBRARIES PRIVATE
  art::Framework_Principal
  art::Framework_Services_Registry
  nusimdata::SimulationBase
  larsim::MCCheater_BackTrackerService_service
  larsim::MCCheater_ParticleInventoryService_service
  larsim::MCCheater_PhotonBackTrackerService_service
  lardata::DetectorInfoServices_DetectorClocksServiceStandard_service
  ${HEP_HPC_HDF5}
  HDF5::HDF5
)

install_headers()
install_source()
