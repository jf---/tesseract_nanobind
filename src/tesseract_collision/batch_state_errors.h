#pragma once

#include <stdexcept>

struct InvalidBatch : std::runtime_error
{
  using std::runtime_error::runtime_error;
};
struct UnsupportedContext : std::runtime_error
{
  using std::runtime_error::runtime_error;
};
