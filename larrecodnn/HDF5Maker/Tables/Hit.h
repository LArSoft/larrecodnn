#pragma once

#include "larrecodnn/HDF5Maker/Tables/Table.h"

namespace ng {

class HitTable : public Table<
  unsigned int /*run*/, unsigned int /*subrun*/, unsigned int /*event*/,
  unsigned int /*hit_id*/, float /*integral*/, float /*rms*/,
  unsigned int /*tpc_id*/, unsigned int /*plane*/, float /*wire*/,
  float /*time*/, unsigned int /*view*/, float /*proj*/, float /*drift*/
>{
public:
  // hit table constructor
  HitTable(std::string const& hitLabel, std::vector<Row> const& data={});

  // function to fill table from event
  void Fill(art::Event const& evt);

private:
  std::string fHitLabel; ///< Label for hit data product

}; // class HitTable

} // namespace ng
