#include "larrecodnn/HDF5Maker/Tables/OpHit.h"

#include "lardataobj/RecoBase/OpHit.h"

namespace ng {

//-----------------------------------------------------------------------------
// names of columns in optical hit table
std::vector<std::string> static const OpHitColumns
{
  "run", "subrun", "event", "oh_id", "channel", "time", "area", "width",
  "amplitude"
};

//-----------------------------------------------------------------------------
// optical hit table constructor
OpHitTable::OpHitTable(std::string const& opHitLabel,
                       std::vector<Row> const& data)
  : Table("ophits", OpHitColumns, data), fOpHitLabel(opHitLabel)
{}

//-----------------------------------------------------------------------------
void OpHitTable::Fill(art::Event const& evt)
{
  // get event ID
  art::EventID const& id = evt.id();
  
  // loop over hits
  auto hits = evt.getHandle<std::vector<recob::OpHit>>(fOpHitLabel);
  for (size_t oh_id = 0; oh_id < hits->size(); ++oh_id) {
    recob::OpHit const& hit = hits->at(oh_id);

    fData.push_back({
      id.run(), id.subRun(), id.event(),       // event ID
      oh_id, hit.OpChannel(), hit.PeakTime(),  // oh_id, channel, time
      hit.Area(), hit.Width(), hit.Amplitude() // area, width, amplitude
    });

  } // for hit

} // function OpHitTable::Fill

} // namespace ng
