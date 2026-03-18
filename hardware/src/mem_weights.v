module mem_weights #(
    parameter VOCAB_SIZE = 512,
    parameter DIM = 64,
    parameter HIDDEN_DIM = 172,
    parameter N_LAYERS = 5,
    parameter N_HEADS = 8,
    parameter N_KV_HEADS = 4,
    parameter MAX_SEQ_LEN = 512
) (
    input wire clk,
    input wire rst_n,
    input wire sync_start,
    output reg busy,
    output reg done,
    input wire wgt_rd_en,
    input wire [31:0] wgt_rd_addr,
    output reg [31:0] wgt_rd_data
);

localparam KV_DIM = (DIM * N_KV_HEADS) / N_HEADS;

`include "memory_map.vh"

localparam DEPTH = `WGT_RMS_FINAL_BASE + `WGT_RMS_FINAL_SIZE;

reg [31:0] mem [0:DEPTH-1];
reg synced;

initial begin
    synced = 1'b0;
    busy = 1'b0;
    done = 1'b0;
    wgt_rd_data = 32'd0;
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        busy <= 1'b0;
        done <= 1'b0;
        synced <= 1'b0;
        wgt_rd_data <= 32'd0;
    end else begin
        done <= 1'b0;

        if (sync_start && !synced) begin
            busy <= 1'b1;
            synced <= 1'b1;
            busy <= 1'b0;
            done <= 1'b1;
        end else begin
            busy <= 1'b0;
        end

        if (wgt_rd_en && (wgt_rd_addr < DEPTH)) begin
            wgt_rd_data <= mem[wgt_rd_addr];
        end else begin
            wgt_rd_data <= 32'd0;
        end
    end
end

endmodule
