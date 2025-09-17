#include "ctc_decode.h"

std::string ctc_decode(
    const std::vector<int64_t>& tokens,
    const std::unordered_map<int, std::string>& vocab,
    const std::vector<std::string>& default_alphabet,
    int blank_id) 
{
    std::string result;
    int64_t prev = -1;
    for (auto t : tokens) {
        if (t == blank_id || t == prev) {
            prev = t;
            continue;
        }
        std::string token;
        if (!vocab.empty()) {
            if (vocab.find(t) == vocab.end()) continue;
            token = vocab.at(t);
        } else {
            if (t >= static_cast<int>(default_alphabet.size())) continue;
            token = default_alphabet[t];
        }
        if (token == "|") token = " ";  // Convert "|" to space
        result += token;
        prev = t;
    }
    return result;
}
