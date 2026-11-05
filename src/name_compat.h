#pragma once

#include <string>
#include <vector>

namespace tesseract_nb
{
template <typename Ids>
std::vector<std::string> names(const Ids& ids)
{
  std::vector<std::string> result;
  result.reserve(ids.size());
  for (const auto& id : ids)
    result.push_back(id.name());
  return result;
}
}
