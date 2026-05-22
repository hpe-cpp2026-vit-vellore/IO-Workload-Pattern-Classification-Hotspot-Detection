#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <cstring>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

// A simple flat JSON parser/writer
// Parses a flat JSON object string, clips specific fields if they are in the bounds list, and outputs the updated JSON string.
extern "C" {
    EXPORT bool parse_and_clip_json(
        const char* json_str,
        const char** keys,
        const double* low_bounds,
        const double* high_bounds,
        int bounds_count,
        char* out_json,
        int out_len
    ) {
        if (!json_str || !out_json || out_len <= 0) {
            return false;
        }

        // Build key-to-bounds map
        std::unordered_map<std::string, std::pair<double, double>> bounds;
        for (int i = 0; i < bounds_count; ++i) {
            if (keys[i]) {
                bounds[keys[i]] = {low_bounds[i], high_bounds[i]};
            }
        }

        std::string input(json_str);
        size_t start = input.find('{');
        size_t end = input.rfind('}');
        if (start == std::string::npos || end == std::string::npos || end <= start) {
            return false;
        }

        std::string body = input.substr(start + 1, end - start - 1);
        std::stringstream ss;
        ss << "{";

        size_t pos = 0;
        bool first = true;

        while (pos < body.length()) {
            // Find key starting quote
            size_t key_start = body.find('"', pos);
            if (key_start == std::string::npos) break;
            size_t key_end = body.find('"', key_start + 1);
            if (key_end == std::string::npos) break;

            std::string key = body.substr(key_start + 1, key_end - key_start - 1);

            // Find colon
            size_t colon_pos = body.find(':', key_end + 1);
            if (colon_pos == std::string::npos) break;

            // Find value start (skip whitespace)
            size_t val_start = colon_pos + 1;
            while (val_start < body.length() && (body[val_start] == ' ' || body[val_start] == '\t' || body[val_start] == '\r' || body[val_start] == '\n')) {
                val_start++;
            }

            if (val_start >= body.length()) break;

            std::string raw_val;
            bool is_string = false;
            size_t next_comma_or_end = std::string::npos;

            if (body[val_start] == '"') {
                // String value
                is_string = true;
                size_t val_end = body.find('"', val_start + 1);
                if (val_end == std::string::npos) break;
                raw_val = body.substr(val_start + 1, val_end - val_start - 1);
                next_comma_or_end = body.find(',', val_end + 1);
                pos = (next_comma_or_end == std::string::npos) ? body.length() : next_comma_or_end + 1;
            } else {
                // Number, bool, or null
                next_comma_or_end = body.find(',', val_start);
                if (next_comma_or_end == std::string::npos) {
                    raw_val = body.substr(val_start);
                    pos = body.length();
                } else {
                    raw_val = body.substr(val_start, next_comma_or_end - val_start);
                    pos = next_comma_or_end + 1;
                }
                // Trim trailing whitespace/newlines
                while (!raw_val.empty() && (raw_val.back() == ' ' || raw_val.back() == '\t' || raw_val.back() == '\r' || raw_val.back() == '\n' || raw_val.back() == '}')) {
                    raw_val.pop_back();
                }
            }

            if (!first) {
                ss << ", ";
            }
            first = false;

            ss << "\"" << key << "\": ";
            if (is_string) {
                ss << "\"" << raw_val << "\"";
            } else {
                auto it = bounds.find(key);
                if (it != bounds.end()) {
                    try {
                        size_t parsed_len = 0;
                        double val = std::stod(raw_val, &parsed_len);
                        double low = it->second.first;
                        double high = it->second.second;
                        if (val < low) {
                            val = low;
                        } else if (val > high) {
                            val = high;
                        }
                        // Write clipped value
                        ss << val;
                    } catch (...) {
                        ss << raw_val;
                    }
                } else {
                    ss << raw_val;
                }
            }
        }

        ss << "}";
        std::string result = ss.str();
        if (result.length() >= static_cast<size_t>(out_len)) {
            return false;
        }

        std::strcpy(out_json, result.c_str());
        return true;
    }
}
