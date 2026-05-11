#pragma once

#include "larrecodnn/HDF5Maker/Tables/Table.h"

namespace ng {

class ChargeHitTable : public Table<
  unsigned int /*run*/, unsigned int /*subrun*/, unsigned int /*event*/,
  unsigned int /*ch_id*/, float /*integral*/, float /*rms*/,
  unsigned int /*tpc_id*/, unsigned int /*plane*/, float /*wire*/,
  float /*time*/, unsigned int /*view*/, float /*proj*/, float /*drift*/
>{
public:
  // charge hit table constructor
  ChargeHitTable(std::string const& chargeHitLabel,
                 std::vector<Row> const& data={});

  // function to fill table from event
  void Fill(art::Event const& evt);

private:
  std::string fChargeHitLabel; ///< Label for charge hit data product

}; // class ChargeHitTable

} // namespace ng
