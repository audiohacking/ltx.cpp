// ltx_perf.hpp — optional background resource-usage monitor
//
// Prints CPU%, process RSS, system free RAM, and (if available) GPU memory
// to stderr every N seconds.  Designed to be zero-overhead when disabled.
//
// Usage:
//   LtxPerfMonitor mon(10, backend_name, gpu_weight_mb);
//   // ... inference ...
//   mon.stop();   // or destructor cleans up

#pragma once
#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

#if defined(__APPLE__)
#  include <mach/mach.h>
#  include <mach/mach_host.h>
#  include <mach/task_info.h>
#elif defined(__linux__)
#  include <cstring>
#  include <fstream>
#  include <sys/resource.h>
#  include <unistd.h>
#endif

struct LtxPerfMonitor {
    std::thread worker;
    std::atomic<bool> running{false};

    // backend_name: e.g. "Metal" or "" for CPU-only.
    // gpu_weight_mb: total MB moved to GPU (from ggml_backend_buffer_get_size).
    LtxPerfMonitor(int interval_s, const std::string & backend_name, size_t gpu_weight_mb) {
        running = true;
        worker = std::thread([this, interval_s, backend_name, gpu_weight_mb]() {
            PerfState prev = sample_process();
            auto prev_time = std::chrono::steady_clock::now();

            while (running.load()) {
                // Sleep in short chunks so stop() is responsive.
                for (int i = 0; i < interval_s * 10 && running.load(); ++i)
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                if (!running.load()) break;

                auto now = std::chrono::steady_clock::now();
                PerfState cur = sample_process();
                double wall_s = std::chrono::duration<double>(now - prev_time).count();

                double cpu_pct = cpu_percent(prev, cur, wall_s);
                size_t rss_mb  = cur.rss_bytes / (1024 * 1024);
                size_t free_mb = system_free_mb();

                char gpu_buf[64] = "";
                if (!backend_name.empty())
                    snprintf(gpu_buf, sizeof(gpu_buf), "  gpu-weights=%zu MB (%s)",
                             gpu_weight_mb, backend_name.c_str());

                fprintf(stderr, "\n[ltx/perf] cpu=%.1f%%  rss=%zu MB  sys-free=%zu MB%s\n",
                        cpu_pct, rss_mb, free_mb, gpu_buf);
                fflush(stderr);

                prev = cur;
                prev_time = now;
            }
        });
    }

    void stop() {
        running = false;
        if (worker.joinable()) worker.join();
    }

    ~LtxPerfMonitor() { stop(); }

    // ── platform internals ────────────────────────────────────────────────────

    struct PerfState {
        uint64_t cpu_us = 0;   // accumulated process CPU microseconds
        size_t   rss_bytes = 0;
    };

#if defined(__APPLE__)
    static PerfState sample_process() {
        PerfState s;
        mach_task_basic_info_data_t info{};
        mach_msg_type_number_t cnt = MACH_TASK_BASIC_INFO_COUNT;
        if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                      reinterpret_cast<task_info_t>(&info), &cnt) == KERN_SUCCESS) {
            s.rss_bytes = info.resident_size;
        }
        task_thread_times_info_data_t times{};
        mach_msg_type_number_t tcnt = TASK_THREAD_TIMES_INFO_COUNT;
        if (task_info(mach_task_self(), TASK_THREAD_TIMES_INFO,
                      reinterpret_cast<task_info_t>(&times), &tcnt) == KERN_SUCCESS) {
            s.cpu_us = ((uint64_t)times.user_time.seconds   + times.system_time.seconds)   * 1000000ULL
                     +  (uint64_t)times.user_time.microseconds + times.system_time.microseconds;
        }
        return s;
    }

    static size_t system_free_mb() {
        vm_statistics64_data_t vmstat{};
        mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
        if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                              reinterpret_cast<host_info64_t>(&vmstat), &cnt) != KERN_SUCCESS)
            return 0;
        vm_size_t page_size = 4096;
        host_page_size(mach_host_self(), &page_size);
        return (size_t)(vmstat.free_count + vmstat.speculative_count) * page_size / (1024 * 1024);
    }

#elif defined(__linux__)
    static PerfState sample_process() {
        PerfState s;
        // RSS from /proc/self/status
        std::ifstream f("/proc/self/status");
        std::string line;
        while (std::getline(f, line)) {
            if (line.rfind("VmRSS:", 0) == 0) {
                s.rss_bytes = (size_t)std::stoul(line.substr(6)) * 1024;
                break;
            }
        }
        // CPU time from getrusage (process-wide user+sys)
        struct rusage ru{};
        if (getrusage(RUSAGE_SELF, &ru) == 0) {
            s.cpu_us = ((uint64_t)ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) * 1000000ULL
                     +  (uint64_t)ru.ru_utime.tv_usec + ru.ru_stime.tv_usec;
        }
        return s;
    }

    static size_t system_free_mb() {
        std::ifstream f("/proc/meminfo");
        std::string line;
        size_t free_kb = 0, avail_kb = 0;
        while (std::getline(f, line)) {
            if (line.rfind("MemFree:", 0) == 0)  free_kb  = std::stoul(line.substr(8));
            if (line.rfind("MemAvailable:", 0) == 0) { avail_kb = std::stoul(line.substr(13)); break; }
        }
        return (avail_kb ? avail_kb : free_kb) / 1024;
    }

#else
    static PerfState sample_process() { return {}; }
    static size_t system_free_mb() { return 0; }
#endif

    static double cpu_percent(const PerfState & a, const PerfState & b, double wall_s) {
        if (wall_s <= 0.0) return 0.0;
        int ncpu = (int)std::thread::hardware_concurrency();
        if (ncpu < 1) ncpu = 1;
        double delta_cpu_s = (double)(b.cpu_us - a.cpu_us) / 1e6;
        return (delta_cpu_s / (wall_s * ncpu)) * 100.0;
    }
};
