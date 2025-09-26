#pragma once

#include "larrecodnn/HDF5Maker/Tables/Table.h"

namespace ng {

class ParticleTable : public Table
<int /*run*/, int /*subrun*/, int /*event*/,
int /*g4_id*/, int /*type*/, int /*parent_id*/,
float /*mom_x*/, float /*mom_y*/, float /*mom_z*/, float /*mom_e*/,
float /*start_x*/, float /*start_y*/, float /*start_z*/, float /*start_t*/, 
float /*end_x*/, float /*end_y*/, float /*end_z*/, float /*end_t*/, 
std::string /*start_process*/, std::string /*end_process*/>
{
public:
  // particle table constructor
  ParticleTable(std::string const& hitLabel, std::vector<Row> const& data={});

  // function to fill table from event
  void Fill(art::Event const& evt);

private:
  std::string fHitLabel; ///< Label for hit data product

}; // class ParticleTable

} // namespace ng
