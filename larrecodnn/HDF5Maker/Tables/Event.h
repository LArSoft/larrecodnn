#pragma once

#include "larrecodnn/HDF5Maker/Tables/Table.h"

namespace ng {

class EventTable : public Table<int, int, int>
{
public:
  // event table constructor
  EventTable(std::vector<Row> const& data={});

  // function to fill table from event
  void Fill(art::Event const& evt);
}; // class EventTable

} // namespace ng
