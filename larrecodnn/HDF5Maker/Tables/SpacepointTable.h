#pragma once

#include "larrecodnn/HDF5Maker/Tables/Table.h"

namespace ng {

  class SpacepointTable : public Table<unsigned int /*run*/,
                                       unsigned int /*subrun*/,
                                       unsigned int /*event*/,
                                       unsigned int /*sp_id*/,
                                       float /*position_x*/,
                                       float /*position_y*/,
                                       float /*position_z*/,
                                       int /*wh_id_1*/,
                                       int /*wh_id_2*/,
                                       int /*wh_id_3*/
                                       > {
  public:
    // spacepoint table constructor
    SpacepointTable(std::string const& spacepointLabel, std::vector<Row> const& data = {});

    // function to fill table from event
    void Fill(art::Event const& evt);

  private:
    std::string fSpacepointLabel; ///< Label for spacepoint data product

  }; // class SpacepointTable

} // namespace ng
