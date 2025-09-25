#include "larrecodnn/HDF5Maker/Tables/Event.h"

namespace ng {

//-----------------------------------------------------------------------------
// names of columns in event table
std::vector<std::string> static const EventColumns
{
  "run", "subrun", "event"
};

//-----------------------------------------------------------------------------
// event table constructor
EventTable::EventTable(std::vector<Row> const& data)
  : Table("events", EventColumns, data)
{}

//-----------------------------------------------------------------------------
void EventTable::Fill(art::Event const& evt)
{
  // get event ID
  art::EventID const& id = evt.id();

  // fill table row
  fData.push_back({id.run(), id.subRun(), id.event()});

} // function EventTable::Fill

} // namespace ng
