#include <iostream>
#include <string>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main(int argc, char* argv[]) {
    std::string bounds_str = "";
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--bounds" && i + 1 < argc) {
            bounds_str = argv[i + 1];
            break;
        }
    }

    std::string input_line;
    if (!std::getline(std::cin, input_line) || input_line.empty()) {
        return 0;
    }

    try {
        json data = json::parse(input_line);
        if (!bounds_str.empty()) {
            json bounds = json::parse(bounds_str);
            auto low_b = bounds.value("low", json::object());
            auto high_b = bounds.value("high", json::object());

            for (auto it = data.begin(); it != data.end(); ++it) {
                std::string key = it.key();
                if (low_b.contains(key) && high_b.contains(key) && !it.value().is_null()) {
                    try {
                        if (it.value().is_number()) {
                            double val = it.value().get<double>();
                            double low = low_b[key].get<double>();
                            double high = high_b[key].get<double>();
                            if (val < low) {
                                val = low;
                            } else if (val > high) {
                                val = high;
                            }
                            if (it.value().is_number_integer()) {
                                *it = static_cast<long long>(val);
                            } else {
                                *it = val;
                            }
                        }
                    } catch (...) {
                        // Skip if conversion fails
                    }
                }
            }
        }
        std::cout << data.dump() << std::endl;
    } catch (...) {
        // On any parsing/processing error, write original input to stdout unchanged and exit 0
        std::cout << input_line << std::endl;
    }
    return 0;
}
