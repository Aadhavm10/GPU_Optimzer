#pragma once

#include <string>
#include <memory>
#include <mutex>
#include <fstream>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "config.h"

class Logger {
public:
    static void init(LogLevel level, const std::string& log_file = "");
    static void set_level(LogLevel level);
    
    template<typename... Args>
    static void debug(const std::string& format, Args&&... args) {
        log(LogLevel::DEBUG, format, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    static void info(const std::string& format, Args&&... args) {
        log(LogLevel::INFO, format, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    static void warn(const std::string& format, Args&&... args) {
        log(LogLevel::WARN, format, std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    static void error(const std::string& format, Args&&... args) {
        log(LogLevel::ERROR, format, std::forward<Args>(args)...);
    }

private:
    static LogLevel current_level_;
    static std::unique_ptr<std::ofstream> log_file_;
    static std::mutex log_mutex_;
    
    static std::string get_timestamp();
    static std::string level_to_string(LogLevel level);
    
    template<typename... Args>
    static void log(LogLevel level, const std::string& format, Args&&... args) {
        if (level < current_level_) return;
        
        std::lock_guard<std::mutex> lock(log_mutex_);
        
        std::string message = format_string(format, std::forward<Args>(args)...);
        std::string log_line = "[" + get_timestamp() + "] [" + level_to_string(level) + "] " + message + "\n";
        
        // Output to console
        std::cout << log_line;
        std::cout.flush();
        
        // Output to file if configured
        if (log_file_ && log_file_->is_open()) {
            *log_file_ << log_line;
            log_file_->flush();
        }
    }
    
    template<typename... Args>
    static std::string format_string(const std::string& format, Args&&... args) {
        // Simple format string implementation
        // In production, consider using fmt library or similar
        std::ostringstream oss;
        format_string_impl(oss, format, std::forward<Args>(args)...);
        return oss.str();
    }
    
    template<typename T, typename... Args>
    static void format_string_impl(std::ostringstream& oss, const std::string& format, T&& value, Args&&... args) {
        size_t pos = format.find("{}");
        if (pos != std::string::npos) {
            oss << format.substr(0, pos) << value;
            format_string_impl(oss, format.substr(pos + 2), std::forward<Args>(args)...);
        } else {
            oss << format;
        }
    }
    
    static void format_string_impl(std::ostringstream& oss, const std::string& format) {
        oss << format;
    }
};
