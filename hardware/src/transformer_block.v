module transformer_block #(
    parameter DIM = 64,
    parameter HIDDEN_DIM = 172,
    parameter N_HEADS = 8,
    parameter N_KV_HEADS = 4,
    parameter MAX_SEQ_LEN = 512
) (
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [2:0] layer_idx,
    input wire [9:0] pos_idx,
    input wire [31:0] block_input_base_addr,
    input wire [31:0] block_output_base_addr,
    output wire busy,
    output reg done
);

localparam STATE_IDLE       = 3'd0;
localparam STATE_ATTN_START = 3'd1;
localparam STATE_ATTN_WAIT  = 3'd2;
localparam STATE_FFN_START  = 3'd3;
localparam STATE_FFN_WAIT   = 3'd4;
localparam STATE_DONE       = 3'd5;
localparam VOCAB_SIZE = 512;
localparam N_LAYERS = 5;

reg [2:0] state;

wire attn_start;
wire attn_busy;
wire attn_done;
wire attn_kv_rd_en;
wire [31:0] attn_kv_rd_addr;
wire [63:0] attn_kv_rd_data;
wire attn_kv_wr_en;
wire [31:0] attn_kv_wr_addr;
wire [63:0] attn_kv_wr_data;
wire ffn_start;
wire ffn_busy;
wire ffn_done;

localparam KV_DIM = (DIM * N_KV_HEADS) / N_HEADS;

`include "memory_map.vh"

function [63:0] kv_read_bits;
    input [31:0] addr;
    begin
        if (addr < `KV_CACHE_SIZE) begin
            kv_read_bits = $realtobits(top_level_module.u_mem_kv_cache.key_cache[addr]);
        end else begin
            kv_read_bits = $realtobits(top_level_module.u_mem_kv_cache.value_cache[addr - `KV_CACHE_SIZE]);
        end
    end
endfunction

task read_kv_local;
    input integer addr;
    output real value;
    begin
        value = $bitstoreal(kv_read_bits(addr));
    end
endtask

task write_kv_local;
    input integer addr;
    input real value;
    begin
        if (addr < `KV_CACHE_SIZE) begin
            top_level_module.u_mem_kv_cache.key_cache[addr] = value;
        end else begin
            top_level_module.u_mem_kv_cache.value_cache[addr - `KV_CACHE_SIZE] = value;
        end
    end
endtask

assign attn_kv_rd_data = kv_read_bits(attn_kv_rd_addr);

always @(*) begin
    if (attn_kv_wr_en) begin
        write_kv_local(attn_kv_wr_addr, $bitstoreal(attn_kv_wr_data));
    end
end

assign busy = (state != STATE_IDLE);
assign attn_start = (state == STATE_ATTN_START);
assign ffn_start = (state == STATE_FFN_START);

attn #(
    .DIM(DIM),
    .N_HEADS(N_HEADS),
    .N_KV_HEADS(N_KV_HEADS),
    .MAX_SEQ_LEN(MAX_SEQ_LEN)
) u_attn (
    .clk(clk),
    .rst_n(rst_n),
    .start(attn_start),
    .layer_idx(layer_idx),
    .pos_idx(pos_idx),
    .kv_rd_en(attn_kv_rd_en),
    .kv_rd_addr(attn_kv_rd_addr),
    .kv_rd_data(attn_kv_rd_data),
    .kv_wr_en(attn_kv_wr_en),
    .kv_wr_addr(attn_kv_wr_addr),
    .kv_wr_data(attn_kv_wr_data),
    .busy(attn_busy),
    .done(attn_done)
);

ffn #(
    .DIM(DIM),
    .HIDDEN_DIM(HIDDEN_DIM),
    .N_HEADS(N_HEADS),
    .N_KV_HEADS(N_KV_HEADS)
) u_ffn (
    .clk(clk),
    .rst_n(rst_n),
    .start(ffn_start),
    .layer_idx(layer_idx),
    .busy(ffn_busy),
    .done(ffn_done)
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= STATE_IDLE;
        done <= 1'b0;
    end else begin
        done <= 1'b0;
        case (state)
            STATE_IDLE: begin
                if (start) begin
                    state <= STATE_ATTN_START;
                end
            end

            STATE_ATTN_START: begin
                state <= STATE_ATTN_WAIT;
            end

            STATE_ATTN_WAIT: begin
                if (attn_done) begin
                    state <= STATE_FFN_START;
                end
            end

            STATE_FFN_START: begin
                state <= STATE_FFN_WAIT;
            end

            STATE_FFN_WAIT: begin
                if (ffn_done) begin
                    state <= STATE_DONE;
                end
            end

            STATE_DONE: begin
                done <= 1'b1;
                state <= STATE_IDLE;
            end

            default: begin
                state <= STATE_IDLE;
            end
        endcase
    end
end

endmodule
