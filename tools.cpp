
#include "tools.hpp"

#include <stdarg.h>


//// from: https://github.com/lettis/Kubix
/**
behaves like sprintf(char*, ...), but with c++ strings and returns the result

\param str pattern to be printed to
\return resulting string
The function internally calls sprintf, but converts the result to a c++ string and returns that one.
Problems of memory allocation are taken care of automatically.
*/
std::string stringprintf(const std::string& str, ...) {
  unsigned int size = 256;
  va_list args;
  char* buf = (char*) malloc(size * sizeof(char));
  va_start(args, str);
  while (size <= (unsigned int) vsnprintf(buf, size, str.c_str(), args)) {
    size *= 2;
    buf = (char*) realloc(buf, size * sizeof(char));
  }
  va_end(args);
  std::string result(buf);
  free(buf);
  return result;
}

