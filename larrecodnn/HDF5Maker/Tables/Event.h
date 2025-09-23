#pragma once

#include "larrecodnn/HDF5Maker/Tables/Table.h"

namespace ng {

class EventTable : public Table<int, int, int>
{
public:
  // event table constructor
  EventTable(const std::vector<Row>& data={});

  // function to fill table from event
  void Fill(const art::Event& evt);
}; // class EventTable

} // namespace ng
