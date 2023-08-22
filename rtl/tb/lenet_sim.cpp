#include <cmath>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include <verilated.h>
#include <verilated_vcd_c.h>
#include <queue>

#include "Vlenet_sim.h"
#include "Vlenet_sim__Dpi.h"

using namespace std;

// axil interface
enum axil_op {
  axil_op_nop,
  axil_op_read,
  axil_op_write,
  axil_op_finish,
};

struct axil_request {
    axil_op op;
    uint8_t addr;
    uint32_t data;
    axil_request() : op(axil_op_nop), addr(0), data(0) {}
    axil_request(axil_op op, uint8_t addr, uint32_t data = 0) : op(op), addr(addr), data(data) {}
};

std::queue<axil_request> request_queue;
std::queue<axil_request> rdata_queue;
std::queue<axil_request> wdata_queue;
std::queue<axil_request> bdata_queue;
std::queue<axil_request> response_queue;

#define MAX_SIM_TIME 100000  // nanoseconds
vluint64_t sim_time = 0;

// register/memory read/write statustics
struct ReadWriteStats {
    uint64_t write_count;
    uint64_t read_count;
    uint32_t bitwidth;

    ReadWriteStats() : write_count(0), read_count(0), bitwidth(0) {}
    ReadWriteStats(const ReadWriteStats &o) : write_count(o.write_count), read_count(o.read_count), bitwidth(o.bitwidth) {}
};

struct RegValue {
    uint16_t cycle_count;
    uint16_t index_count;
    uint16_t value;

    RegValue() : cycle_count(0), index_count(0), value(0) {}
    RegValue(const RegValue &o) : cycle_count(o.cycle_count), index_count(o.index_count), value(o.value) {}
};

std::map<std::string, ReadWriteStats> reg_stats;
std::map<std::string, ReadWriteStats> ram_stats;
std::map<std::string, RegValue> register_value;


void dpi_reg_read(const char *reg_name, int width) {
    ReadWriteStats &stats = reg_stats[reg_name];
    stats.bitwidth = width;
    stats.read_count++;
}

void dpi_reg_write(const char *reg_name, int width) {
    ReadWriteStats &stats = reg_stats[reg_name];
    stats.bitwidth = width;
    stats.write_count++;
}

void dpi_ram_read(const char *ram_name, int width) {
    ReadWriteStats &stats = ram_stats[ram_name];
    stats.bitwidth = width;
    stats.read_count++;
}

void dpi_ram_write(const char *ram_name, int width) {
    ReadWriteStats &stats = ram_stats[ram_name];
    stats.bitwidth = width;
    stats.write_count++;
}

void dpi_reg_output(const char *reg_name, int cycle_count, int index_count, int value) {
    RegValue &reg_v = register_value[reg_name];
    reg_v.cycle_count = cycle_count;
    reg_v.index_count = index_count;
    reg_v.value = value;
}

void dumpReadWriteStats() {
    FILE *statsout = fopen("tb_lenet_sim_read_write_stats.csv", "w");
    fprintf(statsout, "type, name, bitwidthname, bitwidth, readsname, reads, writesname, writes, \n");
    for (auto it = reg_stats.begin(); it != reg_stats.end(); ++it) {
        const std::string name = it->first;
        const ReadWriteStats &stats = it->second;
        fprintf(statsout, "REG, %s, bitwidth, %d, reads, %lu, writes, %lu \n", name.c_str(), stats.bitwidth, stats.read_count, stats.write_count);
    }
    for (auto it = ram_stats.begin(); it != ram_stats.end(); ++it) {
        const std::string name = it->first;
        const ReadWriteStats &stats = it->second;
        fprintf(statsout, "RAM, %s, bitwidth, %d, reads, %lu, writes, %lu \n", name.c_str(), stats.bitwidth, stats.read_count, stats.write_count);
    }
    fclose(statsout);
}

void dumpRegValue() {
    FILE *statsout = fopen("tb_lenet_sim_reg_values.csv", "w");
    fprintf(statsout, "name, cycle, value, \n");
    for (auto it = register_value.begin(); it != register_value.end(); ++it) {
        const std::string name = it->first;
        const RegValue &value = it->second;
        fprintf(statsout, "%s, %u, %u, %d \n", name.c_str(), value.cycle_count, value.index_count, value.value);
    }
    fclose(statsout);
}


// main function
int main(int argc, char** argv, char** env) {
    Vlenet_sim *dut = new Vlenet_sim;

    Verilated::traceEverOn(true);
    VerilatedVcdC *m_trace = new VerilatedVcdC;
    dut->trace(m_trace, 5);
    m_trace->open("tb_lenet_sim.vcd");

    dut->rst = 1;
    dut->clk = 1;

    int tvalid = 0;

    axil_request request;

    fprintf(stderr, "pushing some requests\n");
    request_queue.push(axil_request(axil_op_write, 0, strtol(argv[1], &argv[1], 10)));  // input image
    request_queue.push(axil_request(axil_op_write, 17, 0x0001));  // optical loss,
    request_queue.push(axil_request(axil_op_write, 18, 0x000a));  // calibration length,
    request_queue.push(axil_request(axil_op_write, 19, 0x0004));  // calibration type,
    request_queue.push(axil_request(axil_op_write, 20, 0x000a));  // estimated photonic slack time in cycles
    request_queue.push(axil_request(axil_op_write, 27, 0x0060));  // preamble monitor length >= detection length + photonic slack + 9
    request_queue.push(axil_request(axil_op_write, 28, 0x000a));  // preamble detection length
    request_queue.push(axil_request(axil_op_write, 29, 0x0007));  // propagation_cycle_delay_between_modulators
    request_queue.push(axil_request(axil_op_write, 30, 0x0003));  // propagation_cycle_shift_between_modulators
    fprintf(stderr, "pushed some requests\n");

    // start inference
    fprintf(stderr, "===================\n");
    fprintf(stderr, "Start Calibration\n");
    request_queue.push(axil_request(axil_op_write, 3, 0x0002));  // calibration start signal, only running calibration

    dut->s_axil_user_arvalid = 0;
    dut->s_axil_user_awvalid = 0;
    dut->s_axil_user_wstrb = 0xf;
    dut->s_axil_user_wvalid = 0;
    dut->s_axil_user_bready = 1;

    while (sim_time < MAX_SIM_TIME) {
        if (sim_time > 2 && !dut->clk) {
            dut->rst = 0;
        }

        if (sim_time == 1000 ) {
            fprintf(stderr, "[sim_time == 1000]===================\n");
            request_queue.push(axil_request(axil_op_write, 3, 0x0000));  // inference refresh signal 101010
        }
        if (sim_time == 2000) {
            fprintf(stderr, "[sim_time == 2000] Start Inference on image index %lx\n", strtol(argv[1], &argv[1], 10));
            request_queue.push(axil_request(axil_op_write, 3, 0x0009));  // inference start signal with sparsity
        }

        if (!dut->clk) {
        
            // process request_queue
            if (request.op == axil_op_finish && request_queue.empty() && rdata_queue.empty() && wdata_queue.empty() && bdata_queue.empty()) {
                fprintf(stderr, "  finish\n");
                break;
            }

            if (request.op == axil_op_nop && request_queue.size()) {
                request = request_queue.front();
                request_queue.pop();
            }

            if (dut->s_axil_user_rvalid && dut->s_axil_user_rready) {
                        fprintf(stderr, "  rdata\n");
                auto &request = rdata_queue.front();
                request.data = dut->s_axil_user_rdata;
                response_queue.push(request);
                rdata_queue.pop();
            }

            if (dut->s_axil_user_arvalid && dut->s_axil_user_arready) {
                        fprintf(stderr, "  araddr\n");
                rdata_queue.push(request);
                request.op = axil_op_nop;
                dut->s_axil_user_arvalid = 0;
            }

            if (request.op == axil_op_read) {
                        fprintf(stderr, "  op_read\n");
                dut->s_axil_user_araddr = request.addr;
                dut->s_axil_user_arvalid = 1;
                dut->s_axil_user_rready = 1;
            }

            if (dut->s_axil_user_bvalid && dut->s_axil_user_bready) {
                        fprintf(stderr, "  bdata\n");
                auto &request = bdata_queue.front();
                response_queue.push(request);
                bdata_queue.pop();
            }

            if (dut->s_axil_user_wvalid && dut->s_axil_user_wready) {
                        fprintf(stderr, "  wdata\n");
                // done
                auto &request = wdata_queue.front();
                bdata_queue.push(request);
                wdata_queue.pop();
                dut->s_axil_user_wvalid = 0;
            }

            if (dut->s_axil_user_awvalid && dut->s_axil_user_awready) {
                        fprintf(stderr, "  awaddr\n");
                wdata_queue.push(request);
                request.op = axil_op_nop;
                dut->s_axil_user_awvalid = 0;
                dut->s_axil_user_wvalid = 1;
                dut->s_axil_user_wdata = request.data;
                fprintf(stderr, "  awaddr %x should be %x\n", dut->s_axil_user_wdata, request.data);
            }

            if (request.op == axil_op_write) {
                fprintf(stderr, "  op_write\n");
                dut->s_axil_user_awaddr = request.addr;
                dut->s_axil_user_awvalid = 1;
            }
        }

        dut->clk ^= 1;
        dut->eval();
        m_trace->dump(sim_time);
        sim_time++;
    }

    m_trace->close();
    delete dut;
    dumpReadWriteStats();
    dumpRegValue();
    exit(EXIT_SUCCESS);
}
