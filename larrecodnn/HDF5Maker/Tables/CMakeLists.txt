include_directories($ENV{HEP_HPC_INC})

cet_set_compiler_flags(DIAGS VIGILANT DWARF_VER 4
  WERROR
  NO_UNDEFINED
  EXTRA_FLAGS -pedantic)

cet_make_library(SOURCE
  ChargeHitTable.cxx
  EDepTable.cxx
  EventTable.cxx
  FlashTable.cxx
  LDepTable.cxx
  NeutrinoTable.cxx
  OpHitTable.cxx
  ParticleTable.cxx
  SpacepointTable.cxx
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

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
    CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "12.5")
  # GCC 12.5 (and above) aggressively/spuriously warns about array-bounds issues.
  target_compile_options(larrecodnn_HDF5Maker_Tables
                         PRIVATE "-Wno-array-bounds;-Wno-stringop-overread;-Wno-stringop-overflow;-Wno-sign-compare")
endif()

install_headers()
install_source()
