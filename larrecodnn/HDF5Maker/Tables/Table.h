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
  virtual void WriteNtuple(bool=false) = 0;
  virtual void DestroyNtuple() = 0;
}; // class ITable

// template table class
template <typename... Types>
class Table : public ITable
{
public:

  using Row = std::tuple<Types...>;
  using Ntuple = hep_hpc::hdf5::Ntuple<Types...>;

  // no default constructor
  Table() = delete;

  // default destructor
  ~Table() override = default;

  /// Table constructor
  Table(const std::string& name, const std::vector<std::string>& columns,
        const std::vector<Row>& data={})
    : fName(name), fColumns(columns), fData(data)
  {
    if (sizeof...(Types) != fColumns.size()) {
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

    fNtuple = std::make_unique<Ntuple>(file, fName, form_input_arguments<Row>(fColumns));
  } // function Table::InitNtuple

  template <typename InputTypes, std::size_t... Is>
  auto form_input_arguments_impl(const std::vector<std::string>& names, std::index_sequence<Is...>)
  {
    return std::make_tuple(hep_hpc::hdf5::Column<std::tuple_element_t<Is, InputTypes>, 1>(names[Is])...);
  }

  template <typename InputTypes>
  auto form_input_arguments(const std::vector<std::string>& names)
  {
    constexpr auto N = std::tuple_size_v<InputTypes>;
    return form_input_arguments_impl<InputTypes>(names, std::make_index_sequence<N>{});
  }


  void WriteNtuple(bool clear=true) override
  {
    // for (const Row& row : fData) {
    //   fNtuple->insert(row); // needs to unpack the elements
    // } // for table row

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
