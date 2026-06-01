#pragma once

#include "larrecodnn/HDF5Maker/Tables/Table.h"

namespace ng {

  class EDepTable : public Table<unsigned int /*run*/,
                                 unsigned int /*subrun*/,
                                 unsigned int /*event*/,
                                 unsigned int /*ch_id*/,
                                 unsigned int /*g4_id*/,
                                 float /*energy*/,
                                 float /*position_x*/,
                                 float /*position_y*/,
                                 float /*position_z*/
                                 > {
  public:
    // edep table constructor
    EDepTable(std::string const& chargeHitLabel, std::vector<Row> const& data = {});

    // function to fill table from event
    void Fill(art::Event const& evt);

  private:
    std::string fChargeHitLabel; ///< Label for charge hit data product

  }; // class EDepTable

} // namespace ng
