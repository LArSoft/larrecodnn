#pragma once

#include "art/Framework/Principal/Event.h"

#include "hep_hpc/hdf5/File.hpp"
#include "hep_hpc/hdf5/Ntuple.hpp"

namespace ng {

// virtual base class
class ITable {
public:
  virtual ~ITable() {};
  virtual void Fill(const art::Event&) = 0;
  virtual void InitNtuple(hep_hpc::hdf5::File&) = 0;
  virtual void WriteNtuple(bool) = 0;
  virtual void DestroyNtuple() = 0;
}; // class ITable

// template table class
template <typename... Args>
class Table : public ITable
{
public:

  using Row = std::tuple<Args...>;
  using Ntuple = hep_hpc::hdf5::Ntuple<Args...>;

  // no default constructor
  Table() = delete;

  // default destructor
  ~Table() override = default;

  /// Table constructor
  Table(const std::string& name, const std::vector<std::string>& columns,
        const std::vector<Row>& data={})
    : fName(name), fColumns(columns), fData(data)
  {
    if (sizeof...(Args) != fColumns.size()) {
      throw std::runtime_error("Number of column names does not match number "
                               "of types in table.");
    } // if column mismatch
  } // Table constructor

  /// Function to empty contents of table
  void Clear()
  {
    fData.clear();
  } // function Table::Clear

  /// HDF5 interface
  void InitNtuple(hep_hpc::hdf5::File& file) override
  {
    fNtuple = std::make_unique<Ntuple>(file, fName, fColumns);
  } // function Table::InitNtuple

  void WriteNtuple(bool clear=true) override
  {
    for (const Row& row : fData) {
      // fNtuple->insert(row);
    } // for table row

    if (clear) {
      Clear();
    } // if clearing data
  } // function Table::WriteNtuple

  void DestroyNtuple() override
  {
    fNtuple.reset();
  } // function Table::DestroyNtuple

protected:
  std::string fName;
  std::vector<std::string> fColumns;
  std::vector<Row> fData;
  std::unique_ptr<Ntuple> fNtuple;

}; // template table class

} // namespace ng
