module mem_activation #(
    parameter VOCAB_SIZE = 512,
    parameter DIM = 64,
    parameter HIDDEN_DIM = 172,
    parameter N_HEADS = 8,
    parameter MAX_SEQ_LEN = 512,
    parameter N_KV_HEADS = 4
) (
    input wire clk,
    input wire rst_n,
    input wire rd_en,
    input wire [31:0] rd_addr,
    output reg [31:0] rd_data,
    input wire wr_en,
    input wire [31:0] wr_addr,
    input wire [31:0] wr_data
);

localparam KV_DIM = (DIM * N_KV_HEADS) / N_HEADS;

`include "memory_map.vh"

localparam DEPTH = `ACT_LOGITS_BASE + `ACT_LOGITS_SIZE;

reg [31:0] mem [0:DEPTH-1];
reg [31:0] tag [0:DEPTH-1];
reg [31:0] epoch;

initial begin
    epoch = 32'd1;
    rd_data = 32'd0;
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        epoch <= epoch + 32'd1;
        rd_data <= 32'd0;
    end else begin
        if (wr_en && (wr_addr < DEPTH)) begin
            mem[wr_addr] <= wr_data;
            tag[wr_addr] <= epoch;
        end

        if (rd_en && (rd_addr < DEPTH) && (tag[rd_addr] == epoch)) begin
            rd_data <= mem[rd_addr];
        end else begin
            rd_data <= 32'd0;
        end
    end
end

endmodule
