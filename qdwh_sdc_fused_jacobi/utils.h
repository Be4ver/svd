#pragma once

#include <acl/acl.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>

#define CHECK_ACL(x) { aclError __ret = x; if (__ret != ACL_SUCCESS) std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; } 0

class Timer
{
public:
    Timer() : start_time{}, end_time{}
    {}

    void start()
    {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void end()
    {
        end_time = std::chrono::high_resolution_clock::now();
    }

    auto get_duration_us() const
    {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        return duration;
    }

private:
    decltype(std::chrono::high_resolution_clock::now()) start_time;
    decltype(std::chrono::high_resolution_clock::now()) end_time;
};

static void ReadFile(const char *file_path, void *buffer, size_t length)
{
    std::ifstream file(file_path);
    if (!file.is_open())
        throw std::runtime_error(std::string{} + "cannot open file " + file_path);
    if (!file.read(static_cast<char *>(buffer), length))
        throw std::runtime_error("cannot read " + std::to_string(length) + " bytes");
}
