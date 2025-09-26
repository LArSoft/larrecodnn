#pragma once

#include "larrecodnn/HDF5Maker/Tables/Table.h"

namespace ng {

class WireHitTable : public Table<
  unsigned int /*run*/, unsigned int /*subrun*/, unsigned int /*event*/,
  unsigned int /*wh_id*/, float /*integral*/, float /*rms*/,
  unsigned int /*tpc_id*/, unsigned int /*plane*/, float /*wire*/,
  float /*time*/, unsigned int /*view*/, float /*proj*/, float /*drift*/
>{
public:
  // wire hit table constructor
  WireHitTable(std::string const& wireHitLabel,
               std::vector<Row> const& data={});

  // function to fill table from event
  void Fill(art::Event const& evt);

private:
  std::string fWireHitLabel; ///< Label for wire hit data product

}; // class WireHitTable

} // namespace ng
