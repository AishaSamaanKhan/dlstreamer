#pragma once
#include <string>
#include <vector>
#include <unordered_map>

std::string ctc_decode(
    const std::vector<int64_t>& tokens,
    const std::unordered_map<int, std::string>& vocab,
    const std::vector<std::string>& default_alphabet,
    int blank_id = 0);
