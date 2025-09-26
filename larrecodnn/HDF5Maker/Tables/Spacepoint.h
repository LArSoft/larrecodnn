#pragma once

#include "larrecodnn/HDF5Maker/Tables/Table.h"

namespace ng {

class SpacepointTable : public Table
<int, int, int, int, float, float, float, int, int, int>
{
public:
  // spacepoint table constructor
  SpacepointTable(std::string const& hitLabel,
                  std::string const& spacepointLabel,
                  std::vector<Row> const& data={});

  // function to fill table from event
  void Fill(art::Event const& evt);

private:
  std::string fHitLabel; ///< Label for hit data product
  std::string fSpacepointLabel; ///< Label for spacepoint data product

}; // class SpacepointTable

} // namespace ng
