#include "logger.h"

LogLevel Logger::current_level_ = LogLevel::INFO;
std::unique_ptr<std::ofstream> Logger::log_file_ = nullptr;
std::mutex Logger::log_mutex_;

void Logger::init(LogLevel level, const std::string& log_file) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    current_level_ = level;
    
    if (!log_file.empty()) {
        log_file_ = std::make_unique<std::ofstream>(log_file, std::ios::app);
        if (!log_file_->is_open()) {
            std::cerr << "Failed to open log file: " << log_file << std::endl;
            log_file_.reset();
        }
    }
}

void Logger::set_level(LogLevel level) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    current_level_ = level;
}

std::string Logger::get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    oss << "." << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

std::string Logger::level_to_string(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO ";
        case LogLevel::WARN:  return "WARN ";
        case LogLevel::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}





