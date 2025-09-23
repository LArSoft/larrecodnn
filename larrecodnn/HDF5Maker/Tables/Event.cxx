#include "larrecodnn/HDF5Maker/Tables/Event.h"

namespace ng {

//-----------------------------------------------------------------------------
// names of columns in event table
static const std::vector<std::double> EventColumns
{
  "run", "subrun", "event"
};

//-----------------------------------------------------------------------------
// event table constructor
EventTable::EventTable(const std::vector<Row>& data)
  : Table("events", EventColumns, data)
{}

//-----------------------------------------------------------------------------
void EventTable::Fill(const art::Event& evt)
{
  // get event ID
  const art::EventID& id = evt.id();

  // fill table row
  fData.push_back({id.run(), id.subRun(), id.event()});

} // function EventTable::Fill

} // namespace ng
