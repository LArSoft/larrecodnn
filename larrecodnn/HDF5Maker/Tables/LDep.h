#pragma once

#include "larrecodnn/HDF5Maker/Tables/Table.h"

namespace ng {

class LDepTable : public Table<
  unsigned int /*run*/, unsigned int /*subrun*/, unsigned int /*event*/,
  unsigned int /*oh_id*/, unsigned int /*g4_id*/, float /*energy*/,
  float /*position_x*/, float /*position_y*/, float /*position_z*/
>{
public:
  // light deposit table constructor
  LDepTable(std::string const& opHitLabel,
            std::vector<Row> const& data={});

  // function to fill table from event
  void Fill(art::Event const& evt);

private:
  std::string fOpHitLabel; ///< Label for optical hit data product

}; // class LDepTable

} // namespace ng
