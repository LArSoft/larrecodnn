#include "larrecodnn/HDF5Maker/Tables/FlashTable.h"

#include "lardataobj/RecoBase/OpFlash.h"

namespace nugraph {

  //-----------------------------------------------------------------------------
  // names of columns in flash table
  std::vector<std::string> static const FlashColumns{"run",
                                                     "subrun",
                                                     "event",
                                                     "flash_id",
                                                     "pe",
                                                     "position_y",
                                                     "position_z",
                                                     "position_t",
                                                     "width_y",
                                                     "width_z",
                                                     "width_t"};

  //-----------------------------------------------------------------------------
  // flash table constructor
  FlashTable::FlashTable(std::string const& flashLabel, std::vector<Row> const& data)
    : Table("flashes", FlashColumns, data), fFlashLabel(flashLabel)
  {}

  //-----------------------------------------------------------------------------
  void FlashTable::Fill(art::Event const& evt)
  {
    // get event ID
    art::EventID const& id = evt.id();

    // loop over flashes
    auto flashes = evt.getHandle<std::vector<recob::OpFlash>>(fFlashLabel);
    for (unsigned int flash_id = 0; flash_id < flashes->size(); ++flash_id) {
      recob::OpFlash const& f = flashes->at(flash_id);

      fData.push_back({
        id.run(),
        id.subRun(),
        id.event(), // event ID
        flash_id,
        f.TotalPE(), // flash_id, pe
        f.YCenter(),
        f.ZCenter(),
        f.Time(), // position
        f.YWidth(),
        f.ZWidth(),
        f.TimeWidth() // width
      });

    } // for flash

  } // function FlashTable::Fill

} // namespace nugraph
