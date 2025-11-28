#include "common/csv_writer.h"

namespace dip {
CsvWriter::CsvWriter(const std::string& path) : out_(path, std::ios::binary) {}
bool CsvWriter::good() const { return out_.good(); }

std::string CsvWriter::escape(const std::string& s) {
    bool need_quotes = false;
    for (char c : s) { if (c == ',' || c == '"' || c == '\n' || c=='\r') { need_quotes = true; break; } }
    if (!need_quotes && !s.empty()) return s;
    std::string t; t.reserve(s.size()+2);
    t.push_back('"');
    for (char c : s) { if (c == '"') t.push_back('"'), t.push_back('"'); else t.push_back(c); }
    t.push_back('"');
    return t;
}

void CsvWriter::write_header(const std::vector<std::string>& cols) {
    bool first = true;
    for (auto& c : cols) { if (!first) out_ << ","; first = false; out_ << escape(c); }
    out_ << "\n";
}

void CsvWriter::write_row(const std::vector<std::string>& cols) {
    bool first = true;
    for (auto& c : cols) { if (!first) out_ << ","; first = false; out_ << escape(c); }
    out_ << "\n";
}
}

