#pragma once

#include "larrecodnn/HDF5Maker/Tables/Table.h"

namespace ng {

class ParticleTable : public Table<
  unsigned int /*run*/, unsigned int /*subrun*/, unsigned int /*event*/,
  unsigned int /*g4_id*/, int /*nu_id*/, unsigned int /*pdg_code*/,
  unsigned int /*parent_id*/, float /*momentum_x*/, float /*momentum_y*/,
  float /*momentum_z*/, float /*momentum_e*/, float /*start_x*/,
  float /*start_y*/, float /*start_z*/, float /*start_t*/, float /*end_x*/,
  float /*end_y*/, float /*end_z*/, float /*end_t*/,
  std::string /*start_process*/, std::string /*end_process*/
>{
public:
  // particle table constructor
  ParticleTable(std::string const& wireHitLabel,
                std::vector<Row> const& data={});

  // function to fill table from event
  void Fill(art::Event const& evt);

private:
  std::string fWireHitLabel; ///< Label for wire hit data product

}; // class ParticleTable

} // namespace ng
