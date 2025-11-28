#pragma once
#include <string>
#include <vector>
#include <fstream>

namespace dip {
class CsvWriter {
public:
    explicit CsvWriter(const std::string& path);
    bool good() const;
    void write_header(const std::vector<std::string>& cols);
    void write_row(const std::vector<std::string>& cols);
private:
    std::ofstream out_;
    static std::string escape(const std::string& s);
};
}

