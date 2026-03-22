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
localparam RMS_OP_FINAL        = 2'd3;
localparam MATMUL_OP_CLASSIFY  = 3'd5;

localparam KV_DIM = (DIM * N_KV_HEADS) / N_HEADS;

`include "memory_map.vh"

reg [TOKEN_W-1:0] current_token_id;
reg [9:0] seq_pos;
reg [9:0] current_pos;
reg weights_initialized;

integer embed_idx;
integer embed_base_idx;
reg [31:0] embed_bits;
reg local_act_wr_en;
reg [31:0] local_act_wr_addr;
reg [31:0] local_act_wr_data;
reg embed_done_reg;
reg local_wgt_rd_en;
reg [31:0] local_wgt_rd_addr;
reg [2:0] embed_state;
localparam EMBED_IDLE   = 3'd0;
localparam EMBED_WGT_REQ= 3'd1;
localparam EMBED_WGT_WAIT=3'd2;
localparam EMBED_WGT_CAP= 3'd3;
localparam EMBED_ACT_WR = 3'd4;
localparam EMBED_DONE   = 3'd5;

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
wire [31:0] final_act_rd_data;
wire final_act_wr_en;
wire [31:0] final_act_wr_addr;
wire [31:0] final_act_wr_data;
wire final_wgt_rd_en;
wire [31:0] final_wgt_rd_addr;
wire [31:0] final_wgt_rd_data;

wire cls_act_rd_en;
wire [31:0] cls_act_rd_addr;
wire [31:0] cls_act_rd_data;
wire cls_act_wr_en;
wire [31:0] cls_act_wr_addr;
wire [31:0] cls_act_wr_data;
wire cls_wgt_rd_en;
wire [31:0] cls_wgt_rd_addr;
wire [31:0] cls_wgt_rd_data;

wire block_act_rd_en;
wire [31:0] block_act_rd_addr;
wire [31:0] block_act_rd_data;
wire block_act_wr_en;
wire [31:0] block_act_wr_addr;
wire [31:0] block_act_wr_data;
wire block_wgt_rd_en;
wire [31:0] block_wgt_rd_addr;
wire [31:0] block_wgt_rd_data;
wire block_kv_rd_en;
wire [31:0] block_kv_rd_addr;
wire [31:0] block_kv_rd_data;
wire block_kv_wr_en;
wire [31:0] block_kv_wr_addr;
wire [31:0] block_kv_wr_data;
reg mem_act_rd_en;
reg [31:0] mem_act_rd_addr;
wire [31:0] mem_act_rd_data;
reg mem_act_wr_en;
reg [31:0] mem_act_wr_addr;
reg [31:0] mem_act_wr_data;
reg mem_kv_rd_en;
reg [31:0] mem_kv_rd_addr;
wire [31:0] mem_kv_rd_data;
reg mem_kv_wr_en;
reg [31:0] mem_kv_wr_addr;
reg [31:0] mem_kv_wr_data;
reg mem_wgt_rd_en;
reg [31:0] mem_wgt_rd_addr;
wire [31:0] mem_wgt_rd_data;

assign start_step = in_valid & fsm_in_ready;
assign in_ready = fsm_in_ready;
assign busy = fsm_busy;
assign out_valid = fsm_out_valid;
assign logits_valid = fsm_out_valid;
assign embed_done = embed_done_reg;
assign block_start = (fsm_state == FSM_BLOCK_START);
assign final_rms_start = (fsm_state == FSM_FINAL_RMS_START);
assign cls_start = (fsm_state == FSM_CLS_START);
assign weights_sync_start = start_step && !weights_initialized;
assign final_act_rd_data = mem_act_rd_data;
assign cls_act_rd_data = mem_act_rd_data;
assign block_act_rd_data = mem_act_rd_data;
assign final_wgt_rd_data = mem_wgt_rd_data;
assign cls_wgt_rd_data = mem_wgt_rd_data;
assign block_wgt_rd_data = mem_wgt_rd_data;
assign block_kv_rd_data = mem_kv_rd_data;

always @(*) begin
    mem_act_rd_en = 1'b0;
    mem_act_rd_addr = 32'd0;
    mem_act_wr_en = 1'b0;
    mem_act_wr_addr = 32'd0;
    mem_act_wr_data = 32'd0;
    mem_kv_rd_en = 1'b0;
    mem_kv_rd_addr = 32'd0;
    mem_kv_wr_en = 1'b0;
    mem_kv_wr_addr = 32'd0;
    mem_kv_wr_data = 32'd0;
    mem_wgt_rd_en = 1'b0;
    mem_wgt_rd_addr = 32'd0;

    case (fsm_state)
        FSM_EMBED: begin
            mem_act_wr_en = local_act_wr_en;
            mem_act_wr_addr = local_act_wr_addr;
            mem_act_wr_data = local_act_wr_data;
            mem_wgt_rd_en = local_wgt_rd_en;
            mem_wgt_rd_addr = local_wgt_rd_addr;
        end

        FSM_BLOCK_START,
        FSM_BLOCK_WAIT: begin
            mem_act_rd_en = block_act_rd_en;
            mem_act_rd_addr = block_act_rd_addr;
            mem_act_wr_en = block_act_wr_en;
            mem_act_wr_addr = block_act_wr_addr;
            mem_act_wr_data = block_act_wr_data;
            mem_kv_rd_en = block_kv_rd_en;
            mem_kv_rd_addr = block_kv_rd_addr;
            mem_kv_wr_en = block_kv_wr_en;
            mem_kv_wr_addr = block_kv_wr_addr;
            mem_kv_wr_data = block_kv_wr_data;
            mem_wgt_rd_en = block_wgt_rd_en;
            mem_wgt_rd_addr = block_wgt_rd_addr;
        end

        FSM_FINAL_RMS_START,
        FSM_FINAL_RMS_WAIT: begin
            mem_act_rd_en = final_act_rd_en;
            mem_act_rd_addr = final_act_rd_addr;
            mem_act_wr_en = final_act_wr_en;
            mem_act_wr_addr = final_act_wr_addr;
            mem_act_wr_data = final_act_wr_data;
            mem_wgt_rd_en = final_wgt_rd_en;
            mem_wgt_rd_addr = `WGT_RMS_FINAL_BASE + final_wgt_rd_addr;
        end

        FSM_CLS_START,
        FSM_CLS_WAIT: begin
            mem_act_rd_en = cls_act_rd_en;
            mem_act_rd_addr = cls_act_rd_addr;
            mem_act_wr_en = cls_act_wr_en;
            mem_act_wr_addr = cls_act_wr_addr;
            mem_act_wr_data = cls_act_wr_data;
            mem_wgt_rd_en = cls_wgt_rd_en;
            mem_wgt_rd_addr = cls_wgt_rd_addr;
        end

        default: begin
        end
    endcase
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
    .N_KV_HEADS(N_KV_HEADS),
    .MAX_SEQ_LEN(MAX_SEQ_LEN)
) u_mem_weights (
    .clk(clk),
    .rst_n(rst_n),
    .sync_start(weights_sync_start),
    .busy(weights_sync_busy),
    .done(weights_sync_done),
    .wgt_rd_en(mem_wgt_rd_en),
    .wgt_rd_addr(mem_wgt_rd_addr),
    .wgt_rd_data(mem_wgt_rd_data)
);

mem_activation #(
    .VOCAB_SIZE(VOCAB_SIZE),
    .DIM(DIM),
    .HIDDEN_DIM(HIDDEN_DIM),
    .N_HEADS(N_HEADS),
    .MAX_SEQ_LEN(MAX_SEQ_LEN),
    .N_KV_HEADS(N_KV_HEADS)
) u_mem_activation (
    .clk(clk),
    .rst_n(rst_n),
    .rd_en(mem_act_rd_en),
    .rd_addr(mem_act_rd_addr),
    .rd_data(mem_act_rd_data),
    .wr_en(mem_act_wr_en),
    .wr_addr(mem_act_wr_addr),
    .wr_data(mem_act_wr_data)
);

mem_kv_cache #(
    .DIM(DIM),
    .N_LAYERS(N_LAYERS),
    .N_HEADS(N_HEADS),
    .N_KV_HEADS(N_KV_HEADS),
    .MAX_SEQ_LEN(MAX_SEQ_LEN)
) u_mem_kv_cache (
    .clk(clk),
    .rst_n(rst_n),
    .rd_en(mem_kv_rd_en),
    .rd_addr(mem_kv_rd_addr),
    .rd_data(mem_kv_rd_data),
    .wr_en(mem_kv_wr_en),
    .wr_addr(mem_kv_wr_addr),
    .wr_data(mem_kv_wr_data)
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
    .act_rd_en(block_act_rd_en),
    .act_rd_addr(block_act_rd_addr),
    .act_rd_data(block_act_rd_data),
    .act_wr_en(block_act_wr_en),
    .act_wr_addr(block_act_wr_addr),
    .act_wr_data(block_act_wr_data),
    .wgt_rd_en(block_wgt_rd_en),
    .wgt_rd_addr(block_wgt_rd_addr),
    .wgt_rd_data(block_wgt_rd_data),
    .kv_rd_en(block_kv_rd_en),
    .kv_rd_addr(block_kv_rd_addr),
    .kv_rd_data(block_kv_rd_data),
    .kv_wr_en(block_kv_wr_en),
    .kv_wr_addr(block_kv_wr_addr),
    .kv_wr_data(block_kv_wr_data),
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
    .op_code(RMS_OP_FINAL),
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
    .op_code(MATMUL_OP_CLASSIFY),
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
        local_act_wr_en <= 1'b0;
        local_act_wr_addr <= 32'd0;
        local_act_wr_data <= 32'd0;
        embed_done_reg <= 1'b0;
        local_wgt_rd_en <= 1'b0;
        local_wgt_rd_addr <= 32'd0;
        embed_state <= EMBED_IDLE;
        embed_idx <= 0;
        embed_base_idx <= 0;
        embed_bits <= 32'd0;
    end else begin
        embed_done_reg <= 1'b0;
        local_act_wr_en <= 1'b0;
        local_wgt_rd_en <= 1'b0;

        if (weights_sync_done) begin
            weights_initialized <= 1'b1;
        end

        if (start_step) begin
            current_token_id <= in_token_id;
            current_pos <= seq_pos;
        end

        if (fsm_state != FSM_EMBED) begin
            embed_state <= EMBED_IDLE;
        end else if (weights_initialized) begin
            case (embed_state)
                EMBED_IDLE: begin
                    embed_base_idx <= current_token_id * DIM;
                    embed_idx <= 0;
                    embed_state <= EMBED_WGT_REQ;
                end

                EMBED_WGT_REQ: begin
                    local_wgt_rd_en <= 1'b1;
                    local_wgt_rd_addr <= `WGT_TOKEN_EMBED_BASE + embed_base_idx + embed_idx;
                    embed_state <= EMBED_WGT_WAIT;
                end

                EMBED_WGT_WAIT: begin
                    embed_state <= EMBED_WGT_CAP;
                end

                EMBED_WGT_CAP: begin
                    embed_bits <= mem_wgt_rd_data;
                    embed_state <= EMBED_ACT_WR;
                end

                EMBED_ACT_WR: begin
                    local_act_wr_en <= 1'b1;
                    local_act_wr_addr <= `ACT_X_BASE + embed_idx;
                    local_act_wr_data <= embed_bits;
                    if (embed_idx == (DIM - 1)) begin
                        embed_state <= EMBED_DONE;
                    end else begin
                        embed_idx <= embed_idx + 1;
                        embed_state <= EMBED_WGT_REQ;
                    end
                end

                EMBED_DONE: begin
                    embed_done_reg <= 1'b1;
                    embed_state <= EMBED_IDLE;
                end

                default: begin
                    embed_state <= EMBED_IDLE;
                end
            endcase
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
