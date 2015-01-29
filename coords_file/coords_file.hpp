#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <memory>

extern "C" {
  // use xdrfile library to read/write xtc and trr files from gromacs
  #include "xdrfile/xdrfile.h"
  #include "xdrfile/xdrfile_xtc.h"
  #include "xdrfile/xdrfile_trr.h"
}

namespace CoordsFile {

class Handler {
 public:
  virtual std::vector<float> next() = 0;
  virtual void write(std::vector<float> row) = 0;
  virtual bool eof() = 0;
};

class AsciiHandler : public Handler {
 public:
  AsciiHandler(std::string fname, std::string mode);
  std::vector<float> next();
  void write(std::vector<float> row);
  bool eof();
 protected:
  std::ifstream _ifs;
  std::ofstream _ofs;
  bool _eof;
  std::string _mode;
};

class XtcHandler : public Handler {
 public:
  XtcHandler(std::string fname, std::string mode);
  ~XtcHandler();
  std::vector<float> next();
  void write(std::vector<float> row);
  bool eof();
 protected:
  bool _eof;
  std::string _mode;
  int _natoms;
  int _nrow;
  XDRFILE* _xdr;
  rvec* _coord_buf;
};

typedef std::unique_ptr<Handler> FilePointer;

template <typename T> std::vector<T>
split(std::string s);

FilePointer
open(std::string fname, std::string mode);

} // end namespace 'CoordsFile'

