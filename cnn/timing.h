#ifndef _TIMING_H_
#define _TIMING_H_

#include <iostream>
#include <string>

#include "boost/chrono/system_clocks.hpp"

namespace cnn {

struct Timer {
  Timer(const std::string& msg) : msg(msg), start(boost::chrono::high_resolution_clock::now()) {}
  ~Timer() {
    boost::chrono::high_resolution_clock::time_point stop = boost::chrono::high_resolution_clock::now();
    std::cerr << '[' << msg << ' ' << boost::chrono::duration<double, boost::milli>(stop-start).count() << " ms]\n";
  }
  std::string msg;
  boost::chrono::high_resolution_clock::time_point start;
};

} // namespace cnn

#endif
