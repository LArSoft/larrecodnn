#pragma once

#include "larrecodnn/HDF5Maker/Tables/Table.h"

namespace ng {

class EDepTable : public Table<int, int, int, int, int, float, float, float, float>
{
public:
  // edep table constructor
  EventTable(std::vector<Row> const& data={});

  // function to fill table from event
  void Fill(art::Event const& evt);
}; // class EDepTable

} // namespace ng
