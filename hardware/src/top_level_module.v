module top_level_module #(
    parameter VOCAB_SIZE = 512,
    parameter TOKEN_W = 9,
    parameter LOGIT_W = 32,
    parameter LOGITS_W = VOCAB_SIZE * LOGIT_W,
    parameter DIM = 64,
    parameter HIDDEN_DIM = 172,
    parameter N_LAYERS = 5,
    parameter N_HEADS = 8,
    parameter N_KV_HEADS = 4,
    parameter MAX_SEQ_LEN = 512
) (
    input wire clk,
    input wire rst_n,
    input wire in_valid,
    input wire [TOKEN_W-1:0] in_token_id,
    input wire is_prompt_token,
    output wire in_ready,
    output wire out_valid,
    output reg [TOKEN_W-1:0] next_token_id,
    output wire logits_valid,
    output reg [LOGITS_W-1:0] logits,
    output wire busy
);

localparam FSM_IDLE            = 4'd0;
localparam FSM_EMBED           = 4'd1;
localparam FSM_BLOCK_START     = 4'd2;
localparam FSM_BLOCK_WAIT      = 4'd3;
localparam FSM_FINAL_RMS_START = 4'd4;
localparam FSM_FINAL_RMS_WAIT  = 4'd5;
localparam FSM_CLS_START       = 4'd6;
localparam FSM_CLS_WAIT        = 4'd7;
localparam FSM_DONE            = 4'd8;

localparam KV_DIM = (DIM * N_KV_HEADS) / N_HEADS;

`include "memory_map.vh"

reg [TOKEN_W-1:0] current_token_id;
reg [9:0] seq_pos;
reg [9:0] current_pos;
reg weights_initialized;

integer embed_idx;
integer embed_base_idx;
real embed_value;

wire [3:0] fsm_state;
wire [2:0] fsm_layer_idx;
wire fsm_busy;
wire fsm_out_valid;
wire fsm_in_ready;
wire start_step;
wire embed_done;
wire block_start;
wire block_busy;
wire block_done;
wire final_rms_start;
wire final_rms_busy;
wire final_rms_done;
wire cls_start;
wire cls_busy;
wire cls_done;
wire weights_sync_start;
wire weights_sync_busy;
wire weights_sync_done;
wire [TOKEN_W-1:0] cls_next_token_id;
wire [LOGITS_W-1:0] cls_logits;

wire final_act_rd_en;
wire [31:0] final_act_rd_addr;
wire [63:0] final_act_rd_data;
wire final_act_wr_en;
wire [31:0] final_act_wr_addr;
wire [63:0] final_act_wr_data;
wire final_wgt_rd_en;
wire [31:0] final_wgt_rd_addr;
wire [63:0] final_wgt_rd_data;

wire cls_act_rd_en;
wire [31:0] cls_act_rd_addr;
wire [63:0] cls_act_rd_data;
wire cls_act_wr_en;
wire [31:0] cls_act_wr_addr;
wire [63:0] cls_act_wr_data;
wire cls_wgt_rd_en;
wire [31:0] cls_wgt_rd_addr;
wire [63:0] cls_wgt_rd_data;

function [63:0] act_read_bits;
    input [31:0] addr;
    begin
        if (addr < DIM) begin
            act_read_bits = $realtobits(u_mem_activation.x[addr]);
        end else if (addr >= `ACT_LOGITS_BASE) begin
            act_read_bits = $realtobits(u_mem_activation.logits_real[addr - `ACT_LOGITS_BASE]);
        end else begin
            act_read_bits = 64'd0;
        end
    end
endfunction

function [63:0] final_wgt_read_bits;
    input [31:0] addr;
    begin
        final_wgt_read_bits = $realtobits(u_mem_weights.rms_final_weight[addr]);
    end
endfunction

function [63:0] cls_wgt_read_bits;
    input [31:0] addr;
    begin
        cls_wgt_read_bits = $realtobits(u_mem_weights.token_embedding[addr - `WGT_TOKEN_EMBED_BASE]);
    end
endfunction

task write_act_local;
    input integer addr;
    input real value;
    begin
        if (addr >= `ACT_X_BASE && addr < `ACT_X_BASE + `ACT_X_SIZE) begin
            u_mem_activation.x[addr - `ACT_X_BASE] = value;
        end else if (addr >= `ACT_LOGITS_BASE && addr < `ACT_LOGITS_BASE + `ACT_LOGITS_SIZE) begin
            u_mem_activation.logits_real[addr - `ACT_LOGITS_BASE] = value;
        end
    end
endtask

task read_wgt_local;
    input integer addr;
    output real value;
    begin
        value = $bitstoreal(cls_wgt_read_bits(addr));
    end
endtask

assign start_step = in_valid & fsm_in_ready;
assign in_ready = fsm_in_ready;
assign busy = fsm_busy;
assign out_valid = fsm_out_valid;
assign logits_valid = fsm_out_valid;
assign embed_done = (fsm_state == FSM_EMBED) && weights_initialized;
assign block_start = (fsm_state == FSM_BLOCK_START);
assign final_rms_start = (fsm_state == FSM_FINAL_RMS_START);
assign cls_start = (fsm_state == FSM_CLS_START);
assign weights_sync_start = start_step && !weights_initialized;
assign final_act_rd_data = act_read_bits(final_act_rd_addr);
assign cls_act_rd_data = act_read_bits(cls_act_rd_addr);
assign final_wgt_rd_data = final_wgt_read_bits(final_wgt_rd_addr);
assign cls_wgt_rd_data = cls_wgt_read_bits(cls_wgt_rd_addr);

always @(*) begin
    if (final_act_wr_en) begin
        write_act_local(final_act_wr_addr, $bitstoreal(final_act_wr_data));
    end
    if (cls_act_wr_en) begin
        write_act_local(cls_act_wr_addr, $bitstoreal(cls_act_wr_data));
    end
end

fsm #(
    .N_LAYERS(N_LAYERS)
) u_fsm (
    .clk(clk),
    .rst_n(rst_n),
    .start(start_step),
    .embed_done(embed_done),
    .block_done(block_done),
    .final_rms_done(final_rms_done),
    .cls_done(cls_done),
    .state(fsm_state),
    .layer_idx(fsm_layer_idx),
    .in_ready(fsm_in_ready),
    .busy(fsm_busy),
    .out_valid(fsm_out_valid)
);

mem_weights #(
    .VOCAB_SIZE(VOCAB_SIZE),
    .DIM(DIM),
    .HIDDEN_DIM(HIDDEN_DIM),
    .N_LAYERS(N_LAYERS),
    .N_HEADS(N_HEADS),
    .N_KV_HEADS(N_KV_HEADS)
) u_mem_weights (
    .clk(clk),
    .rst_n(rst_n),
    .sync_start(weights_sync_start),
    .busy(weights_sync_busy),
    .done(weights_sync_done)
);

mem_activation #(
    .VOCAB_SIZE(VOCAB_SIZE),
    .DIM(DIM),
    .HIDDEN_DIM(HIDDEN_DIM),
    .N_HEADS(N_HEADS),
    .MAX_SEQ_LEN(MAX_SEQ_LEN),
    .N_KV_HEADS(N_KV_HEADS)
) u_mem_activation (
    .rst_n(rst_n)
);

mem_kv_cache #(
    .DIM(DIM),
    .N_LAYERS(N_LAYERS),
    .N_HEADS(N_HEADS),
    .N_KV_HEADS(N_KV_HEADS),
    .MAX_SEQ_LEN(MAX_SEQ_LEN)
) u_mem_kv_cache (
    .rst_n(rst_n)
);

transformer_block #(
    .DIM(DIM),
    .HIDDEN_DIM(HIDDEN_DIM),
    .N_HEADS(N_HEADS),
    .N_KV_HEADS(N_KV_HEADS),
    .MAX_SEQ_LEN(MAX_SEQ_LEN)
) u_block (
    .clk(clk),
    .rst_n(rst_n),
    .start(block_start),
    .layer_idx(fsm_layer_idx),
    .pos_idx(current_pos),
    .block_input_base_addr(32'd0),
    .block_output_base_addr(32'd0),
    .busy(block_busy),
    .done(block_done)
);

kernel_rmsnorm #(
    .DIM(DIM),
    .CONTROL_MODE(2)
) u_final_rms (
    .clk(clk),
    .rst_n(rst_n),
    .start(final_rms_start),
    .layer_idx(32'd0),
    .busy(final_rms_busy),
    .done(final_rms_done),
    .act_rd_en(final_act_rd_en),
    .act_rd_addr(final_act_rd_addr),
    .act_rd_data(final_act_rd_data),
    .act_wr_en(final_act_wr_en),
    .act_wr_addr(final_act_wr_addr),
    .act_wr_data(final_act_wr_data),
    .wgt_rd_en(final_wgt_rd_en),
    .wgt_rd_addr(final_wgt_rd_addr),
    .wgt_rd_data(final_wgt_rd_data)
);

kernel_matmul #(
    .VOCAB_SIZE(VOCAB_SIZE),
    .TOKEN_W(TOKEN_W),
    .LOGIT_W(LOGIT_W),
    .LOGITS_W(LOGITS_W),
    .DIM(DIM),
    .HIDDEN_DIM(HIDDEN_DIM),
    .N_LAYERS(N_LAYERS),
    .N_HEADS(N_HEADS),
    .N_KV_HEADS(N_KV_HEADS),
    .MAX_SEQ_LEN(MAX_SEQ_LEN),
    .CONTROL_MODE(4)
) u_cls_matmul (
    .clk(clk),
    .rst_n(rst_n),
    .start(cls_start),
    .layer_idx(32'd0),
    .busy(cls_busy),
    .done(cls_done),
    .next_token(cls_next_token_id),
    .flat_logits(cls_logits),
    .act_rd_en(cls_act_rd_en),
    .act_rd_addr(cls_act_rd_addr),
    .act_rd_data(cls_act_rd_data),
    .act_wr_en(cls_act_wr_en),
    .act_wr_addr(cls_act_wr_addr),
    .act_wr_data(cls_act_wr_data),
    .wgt_rd_en(cls_wgt_rd_en),
    .wgt_rd_addr(cls_wgt_rd_addr),
    .wgt_rd_data(cls_wgt_rd_data)
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        current_token_id <= {TOKEN_W{1'b0}};
        seq_pos <= 10'd0;
        current_pos <= 10'd0;
        weights_initialized <= 1'b0;
        next_token_id <= {TOKEN_W{1'b0}};
        logits <= {LOGITS_W{1'b0}};
    end else begin
        if (weights_sync_done) begin
            weights_initialized <= 1'b1;
        end

        if (start_step) begin
            current_token_id <= in_token_id;
            current_pos <= seq_pos;
        end

        if (fsm_state == FSM_EMBED && weights_initialized) begin
            embed_base_idx = current_token_id * DIM;
            for (embed_idx = 0; embed_idx < DIM; embed_idx = embed_idx + 1) begin
                read_wgt_local(`WGT_TOKEN_EMBED_BASE + embed_base_idx + embed_idx, embed_value);
                write_act_local(`ACT_X_BASE + embed_idx, embed_value);
            end
        end

        if (cls_done) begin
            next_token_id <= cls_next_token_id;
            logits <= cls_logits;
        end

        if (fsm_state == FSM_DONE) begin
            seq_pos <= seq_pos + 10'd1;
        end
    end
end

endmodule
