#include "larrecodnn/HDF5Maker/Tables/OpHitTable.h"

#include "canvas/Persistency/Common/FindManyP.h"
#include "lardataobj/RecoBase/OpFlash.h"
#include "lardataobj/RecoBase/OpHit.h"

namespace nugraph {

  //-----------------------------------------------------------------------------
  // names of columns in optical hit table
  std::vector<std::string> static const OpHitColumns{"run",
                                                     "subrun",
                                                     "event",
                                                     "oh_id",
                                                     "flash_id",
                                                     "channel",
                                                     "time",
                                                     "area",
                                                     "width",
                                                     "amplitude"};

  //-----------------------------------------------------------------------------
  // optical hit table constructor
  OpHitTable::OpHitTable(std::string const& opHitLabel,
                         std::string const& flashLabel,
                         std::vector<Row> const& data)
    : Table("ophits", OpHitColumns, data), fOpHitLabel(opHitLabel), fFlashLabel(flashLabel)
  {}

  //-----------------------------------------------------------------------------
  void OpHitTable::Fill(art::Event const& evt)
  {
    // get event ID
    art::EventID const& id = evt.id();

    // loop over hits
    auto hits = evt.getHandle<std::vector<recob::OpHit>>(fOpHitLabel);
    art::FindManyP<recob::OpFlash> fmp(hits, evt, fFlashLabel);
    for (size_t oh_id = 0; oh_id < hits->size(); ++oh_id) {
      recob::OpHit const& hit = hits->at(oh_id);

      // get associated flash id
      int flash_id = -1;
      auto const& flashes = fmp.at(oh_id);
      if (flashes.size() > 1) {
        throw std::runtime_error("Optical hit is associated with multiple "
                                 "flashes");
      }
      else if (flashes.size() == 1) {
        flash_id = flashes[0].key();
      }

      fData.push_back({
        id.run(),
        id.subRun(),
        id.event(),
        oh_id, // event ID, oh_id
        flash_id,
        hit.OpChannel(),
        hit.PeakTime(), // flash_id, channel, time
        hit.Area(),
        hit.Width(),
        hit.Amplitude() // area, width, amplitude
      });

    } // for hit

  } // function OpHitTable::Fill

} // namespace nugraph
