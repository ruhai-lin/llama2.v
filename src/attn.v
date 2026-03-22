module attn #(
    parameter DIM = 64,
    parameter N_HEADS = 8,
    parameter N_KV_HEADS = 4,
    parameter MAX_SEQ_LEN = 512
) (
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [2:0] layer_idx,
    input wire [9:0] pos_idx,
    output reg act_rd_en,
    output reg [31:0] act_rd_addr,
    input wire [31:0] act_rd_data,
    output reg act_wr_en,
    output reg [31:0] act_wr_addr,
    output reg [31:0] act_wr_data,
    output reg wgt_rd_en,
    output reg [31:0] wgt_rd_addr,
    input wire [31:0] wgt_rd_data,
    output reg kv_rd_en,
    output reg [31:0] kv_rd_addr,
    input wire [31:0] kv_rd_data,
    output reg kv_wr_en,
    output reg [31:0] kv_wr_addr,
    output reg [31:0] kv_wr_data,
    output reg busy,
    output reg done
);

localparam VOCAB_SIZE = 512;
localparam N_LAYERS = 5;
localparam HIDDEN_DIM = 172;
localparam HEAD_SIZE = DIM / N_HEADS;
localparam KV_DIM = (DIM * N_KV_HEADS) / N_HEADS;
localparam KV_MUL = N_HEADS / N_KV_HEADS;

`include "memory_map.vh"

localparam STATE_IDLE            = 6'd0;
localparam STATE_RMS_START       = 6'd1;
localparam STATE_RMS_WAIT        = 6'd2;
localparam STATE_QKV_START       = 6'd3;
localparam STATE_QKV_WAIT        = 6'd4;
localparam STATE_ROPE_START      = 6'd5;
localparam STATE_ROPE_WAIT       = 6'd6;
localparam STATE_KV_K_RD_REQ     = 6'd7;
localparam STATE_KV_K_RD_WAIT    = 6'd8;
localparam STATE_KV_K_RD_CAP     = 6'd9;
localparam STATE_KV_K_WR         = 6'd10;
localparam STATE_KV_V_RD_REQ     = 6'd11;
localparam STATE_KV_V_RD_WAIT    = 6'd12;
localparam STATE_KV_V_RD_CAP     = 6'd13;
localparam STATE_KV_V_WR         = 6'd14;
localparam STATE_CLEAR_ALL_WR    = 6'd15;
localparam STATE_HEAD_PREP       = 6'd16;
localparam STATE_SCORE_Q_REQ     = 6'd17;
localparam STATE_SCORE_Q_WAIT    = 6'd18;
localparam STATE_SCORE_Q_CAP     = 6'd19;
localparam STATE_SCORE_KV_REQ    = 6'd20;
localparam STATE_SCORE_KV_WAIT   = 6'd21;
localparam STATE_SCORE_KV_CAP    = 6'd22;
localparam STATE_SCORE_WRITE     = 6'd23;
localparam STATE_SOFTMAX_START   = 6'd24;
localparam STATE_SOFTMAX_WAIT    = 6'd25;
localparam STATE_CLEAR_HEAD_WR   = 6'd26;
localparam STATE_MERGE_XB_REQ    = 6'd27;
localparam STATE_MERGE_XB_WAIT   = 6'd28;
localparam STATE_MERGE_XB_CAP    = 6'd29;
localparam STATE_MERGE_ATT_REQ   = 6'd30;
localparam STATE_MERGE_ATT_WAIT  = 6'd31;
localparam STATE_MERGE_ATT_CAP   = 6'd32;
localparam STATE_MERGE_KV_REQ    = 6'd33;
localparam STATE_MERGE_KV_WAIT   = 6'd34;
localparam STATE_MERGE_KV_CAP    = 6'd35;
localparam STATE_MERGE_WRITE     = 6'd36;
localparam STATE_ATTN_OUT_START  = 6'd37;
localparam STATE_ATTN_OUT_WAIT   = 6'd38;
localparam STATE_DONE            = 6'd39;
localparam STATE_SCORE_ACCUM     = 6'd40;
localparam STATE_SCORE_SCALE     = 6'd41;
localparam STATE_MERGE_ACCUM     = 6'd42;

localparam RMS_OP_ATTN           = 2'd1;
localparam MATMUL_OP_QKV         = 3'd1;
localparam MATMUL_OP_ATTN_OUT    = 3'd2;

wire rms_act_rd_en;
wire [31:0] rms_act_rd_addr;
wire [31:0] rms_act_rd_data;
wire rms_act_wr_en;
wire [31:0] rms_act_wr_addr;
wire [31:0] rms_act_wr_data;
wire rms_wgt_rd_en;
wire [31:0] rms_wgt_rd_addr;
wire [31:0] rms_wgt_rd_data;
wire rms_busy;
wire rms_done;

wire matmul_act_rd_en;
wire [31:0] matmul_act_rd_addr;
wire [31:0] matmul_act_rd_data;
wire matmul_act_wr_en;
wire [31:0] matmul_act_wr_addr;
wire [31:0] matmul_act_wr_data;
wire matmul_wgt_rd_en;
wire [31:0] matmul_wgt_rd_addr;
wire [31:0] matmul_wgt_rd_data;
wire matmul_busy;
wire matmul_done;

wire rope_act_rd_en;
wire [31:0] rope_act_rd_addr;
wire [31:0] rope_act_rd_data;
wire rope_act_wr_en;
wire [31:0] rope_act_wr_addr;
wire [31:0] rope_act_wr_data;
wire rope_busy;
wire rope_done;

wire softmax_act_rd_en;
wire [31:0] softmax_act_rd_addr;
wire [31:0] softmax_act_rd_data;
wire softmax_act_wr_en;
wire [31:0] softmax_act_wr_addr;
wire [31:0] softmax_act_wr_data;
wire softmax_busy;
wire softmax_done;

reg local_act_rd_en;
reg [31:0] local_act_rd_addr;
reg local_act_wr_en;
reg [31:0] local_act_wr_addr;
reg [31:0] local_act_wr_data;
reg [5:0] state;

integer current_head;
integer loop_i;
integer timestep;
integer head_base;
integer kv_head;
integer cache_base;
integer att_base;
integer loff;
reg [31:0] kv_bits;
reg [31:0] q_bits;
reg [31:0] score_bits;
reg [31:0] scaled_score_bits;
reg [31:0] xb_bits;
reg [31:0] att_bits;
reg [31:0] v_bits;
reg [31:0] mul_a;
reg [31:0] mul_b;
reg [31:0] add_a;
reg [31:0] add_b;
wire [31:0] mul_y;
wire [31:0] add_y;

localparam [31:0] INV_SCALE_BITS = 32'h3eb504f3;  // 1/sqrt(8)

assign rms_act_rd_data = act_rd_data;
assign matmul_act_rd_data = act_rd_data;
assign rope_act_rd_data = act_rd_data;
assign softmax_act_rd_data = act_rd_data;
assign rms_wgt_rd_data = wgt_rd_data;
assign matmul_wgt_rd_data = wgt_rd_data;

always @(*) begin
    act_rd_en = 1'b0;
    act_rd_addr = 32'd0;
    act_wr_en = 1'b0;
    act_wr_addr = 32'd0;
    act_wr_data = 32'd0;
    wgt_rd_en = 1'b0;
    wgt_rd_addr = 32'd0;

    if (local_act_rd_en) begin
        act_rd_en = 1'b1;
        act_rd_addr = local_act_rd_addr;
    end else if (softmax_act_rd_en) begin
        act_rd_en = 1'b1;
        act_rd_addr = softmax_act_rd_addr;
    end else if (rope_act_rd_en) begin
        act_rd_en = 1'b1;
        act_rd_addr = rope_act_rd_addr;
    end else if (matmul_act_rd_en) begin
        act_rd_en = 1'b1;
        act_rd_addr = matmul_act_rd_addr;
    end else if (rms_act_rd_en) begin
        act_rd_en = 1'b1;
        act_rd_addr = rms_act_rd_addr;
    end

    if (local_act_wr_en) begin
        act_wr_en = 1'b1;
        act_wr_addr = local_act_wr_addr;
        act_wr_data = local_act_wr_data;
    end else if (softmax_act_wr_en) begin
        act_wr_en = 1'b1;
        act_wr_addr = softmax_act_wr_addr;
        act_wr_data = softmax_act_wr_data;
    end else if (rope_act_wr_en) begin
        act_wr_en = 1'b1;
        act_wr_addr = rope_act_wr_addr;
        act_wr_data = rope_act_wr_data;
    end else if (matmul_act_wr_en) begin
        act_wr_en = 1'b1;
        act_wr_addr = matmul_act_wr_addr;
        act_wr_data = matmul_act_wr_data;
    end else if (rms_act_wr_en) begin
        act_wr_en = 1'b1;
        act_wr_addr = rms_act_wr_addr;
        act_wr_data = rms_act_wr_data;
    end

    if (matmul_wgt_rd_en) begin
        wgt_rd_en = 1'b1;
        wgt_rd_addr = matmul_wgt_rd_addr;
    end else if (rms_wgt_rd_en) begin
        wgt_rd_en = 1'b1;
        wgt_rd_addr = `WGT_RMS_ATT_BASE + rms_wgt_rd_addr;
    end

    mul_a = 32'd0;
    mul_b = 32'd0;
    add_a = 32'd0;
    add_b = 32'd0;

    case (state)
        STATE_SCORE_ACCUM: begin
            mul_a = q_bits;
            mul_b = kv_bits;
            add_a = score_bits;
            add_b = mul_y;
        end
        STATE_SCORE_SCALE: begin
            mul_a = score_bits;
            mul_b = INV_SCALE_BITS;
        end
        STATE_MERGE_ACCUM: begin
            mul_a = att_bits;
            mul_b = v_bits;
            add_a = xb_bits;
            add_b = mul_y;
        end
        default: begin
        end
    endcase
end

kernel_mul u_local_mul (
    .a(mul_a),
    .b(mul_b),
    .y(mul_y)
);

kernel_add u_local_add (
    .a(add_a),
    .b(add_b),
    .y(add_y)
);

kernel_rmsnorm #(
    .DIM(DIM)
) u_rmsnorm (
    .clk(clk),
    .rst_n(rst_n),
    .start(state == STATE_RMS_START),
    .layer_idx({29'd0, layer_idx}),
    .op_code(RMS_OP_ATTN),
    .busy(rms_busy),
    .done(rms_done),
    .act_rd_en(rms_act_rd_en),
    .act_rd_addr(rms_act_rd_addr),
    .act_rd_data(rms_act_rd_data),
    .act_wr_en(rms_act_wr_en),
    .act_wr_addr(rms_act_wr_addr),
    .act_wr_data(rms_act_wr_data),
    .wgt_rd_en(rms_wgt_rd_en),
    .wgt_rd_addr(rms_wgt_rd_addr),
    .wgt_rd_data(rms_wgt_rd_data)
);

kernel_matmul #(
    .DIM(DIM),
    .N_HEADS(N_HEADS),
    .N_KV_HEADS(N_KV_HEADS),
    .MAX_SEQ_LEN(MAX_SEQ_LEN)
) u_matmul (
    .clk(clk),
    .rst_n(rst_n),
    .start((state == STATE_QKV_START) || (state == STATE_ATTN_OUT_START)),
    .layer_idx({29'd0, layer_idx}),
    .op_code((state == STATE_QKV_START) ? MATMUL_OP_QKV : MATMUL_OP_ATTN_OUT),
    .busy(matmul_busy),
    .done(matmul_done),
    .next_token(),
    .flat_logits(),
    .act_rd_en(matmul_act_rd_en),
    .act_rd_addr(matmul_act_rd_addr),
    .act_rd_data(matmul_act_rd_data),
    .act_wr_en(matmul_act_wr_en),
    .act_wr_addr(matmul_act_wr_addr),
    .act_wr_data(matmul_act_wr_data),
    .wgt_rd_en(matmul_wgt_rd_en),
    .wgt_rd_addr(matmul_wgt_rd_addr),
    .wgt_rd_data(matmul_wgt_rd_data)
);

kernel_rope #(
    .DIM(DIM),
    .N_HEADS(N_HEADS),
    .N_KV_HEADS(N_KV_HEADS),
    .ACT_Q_BASE(`ACT_Q_BASE),
    .ACT_K_BASE(`ACT_K_BASE)
) u_rope (
    .clk(clk),
    .rst_n(rst_n),
    .start(state == STATE_ROPE_START),
    .pos_idx(pos_idx),
    .busy(rope_busy),
    .done(rope_done),
    .act_rd_en(rope_act_rd_en),
    .act_rd_addr(rope_act_rd_addr),
    .act_rd_data(rope_act_rd_data),
    .act_wr_en(rope_act_wr_en),
    .act_wr_addr(rope_act_wr_addr),
    .act_wr_data(rope_act_wr_data)
);

kernel_softmax #(
    .N_HEADS(N_HEADS),
    .MAX_SEQ_LEN(MAX_SEQ_LEN),
    .ACT_ATT_BASE(`ACT_ATT_BASE)
) u_softmax (
    .clk(clk),
    .rst_n(rst_n),
    .start(state == STATE_SOFTMAX_START),
    .head_idx(current_head),
    .pos_idx(pos_idx),
    .busy(softmax_busy),
    .done(softmax_done),
    .act_rd_en(softmax_act_rd_en),
    .act_rd_addr(softmax_act_rd_addr),
    .act_rd_data(softmax_act_rd_data),
    .act_wr_en(softmax_act_wr_en),
    .act_wr_addr(softmax_act_wr_addr),
    .act_wr_data(softmax_act_wr_data)
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        local_act_rd_en <= 1'b0;
        local_act_rd_addr <= 32'd0;
        local_act_wr_en <= 1'b0;
        local_act_wr_addr <= 32'd0;
        local_act_wr_data <= 32'd0;
        kv_rd_en <= 1'b0;
        kv_rd_addr <= 32'd0;
        kv_wr_en <= 1'b0;
        kv_wr_addr <= 32'd0;
        kv_wr_data <= 32'd0;
        state <= STATE_IDLE;
        current_head <= 0;
        busy <= 1'b0;
        done <= 1'b0;
        loop_i <= 0;
        timestep <= 0;
        head_base <= 0;
        kv_head <= 0;
        cache_base <= 0;
        att_base <= 0;
        loff <= 0;
        kv_bits <= 32'd0;
        q_bits <= 32'd0;
        score_bits <= 32'd0;
        scaled_score_bits <= 32'd0;
        xb_bits <= 32'd0;
        att_bits <= 32'd0;
        v_bits <= 32'd0;
    end else begin
        done <= 1'b0;
        local_act_rd_en <= 1'b0;
        local_act_wr_en <= 1'b0;
        kv_rd_en <= 1'b0;
        kv_wr_en <= 1'b0;

        case (state)
            STATE_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    busy <= 1'b1;
                    state <= STATE_RMS_START;
                end
            end
            STATE_RMS_START: begin busy <= 1'b1; state <= STATE_RMS_WAIT; end
            STATE_RMS_WAIT: begin busy <= 1'b1; if (rms_done) state <= STATE_QKV_START; end
            STATE_QKV_START: begin busy <= 1'b1; state <= STATE_QKV_WAIT; end
            STATE_QKV_WAIT: begin busy <= 1'b1; if (matmul_done) state <= STATE_ROPE_START; end
            STATE_ROPE_START: begin busy <= 1'b1; state <= STATE_ROPE_WAIT; end
            STATE_ROPE_WAIT: begin
                busy <= 1'b1;
                if (rope_done) begin
                    loff <= ({29'd0, layer_idx} * MAX_SEQ_LEN + pos_idx) * KV_DIM;
                    loop_i <= 0;
                    state <= STATE_KV_K_RD_REQ;
                end
            end

            STATE_KV_K_RD_REQ: begin
                busy <= 1'b1;
                local_act_rd_en <= 1'b1;
                local_act_rd_addr <= `ACT_K_BASE + loop_i;
                state <= STATE_KV_K_RD_WAIT;
            end
            STATE_KV_K_RD_WAIT: begin busy <= 1'b1; state <= STATE_KV_K_RD_CAP; end
            STATE_KV_K_RD_CAP: begin
                busy <= 1'b1;
                kv_bits <= act_rd_data;
                state <= STATE_KV_K_WR;
            end
            STATE_KV_K_WR: begin
                busy <= 1'b1;
                kv_wr_en <= 1'b1;
                kv_wr_addr <= `KV_KEY_BASE + loff + loop_i;
                kv_wr_data <= kv_bits;
                state <= STATE_KV_V_RD_REQ;
            end
            STATE_KV_V_RD_REQ: begin
                busy <= 1'b1;
                local_act_rd_en <= 1'b1;
                local_act_rd_addr <= `ACT_V_BASE + loop_i;
                state <= STATE_KV_V_RD_WAIT;
            end
            STATE_KV_V_RD_WAIT: begin busy <= 1'b1; state <= STATE_KV_V_RD_CAP; end
            STATE_KV_V_RD_CAP: begin
                busy <= 1'b1;
                kv_bits <= act_rd_data;
                state <= STATE_KV_V_WR;
            end
            STATE_KV_V_WR: begin
                busy <= 1'b1;
                kv_wr_en <= 1'b1;
                kv_wr_addr <= `KV_VALUE_BASE + loff + loop_i;
                kv_wr_data <= kv_bits;
                if (loop_i == (KV_DIM - 1)) begin
                    loop_i <= 0;
                    state <= STATE_CLEAR_ALL_WR;
                end else begin
                    loop_i <= loop_i + 1;
                    state <= STATE_KV_K_RD_REQ;
                end
            end

            STATE_CLEAR_ALL_WR: begin
                busy <= 1'b1;
                local_act_wr_en <= 1'b1;
                local_act_wr_addr <= `ACT_XB_BASE + loop_i;
                local_act_wr_data <= 32'd0;
                if (loop_i == (DIM - 1)) begin
                    current_head <= 0;
                    state <= STATE_HEAD_PREP;
                end else begin
                    loop_i <= loop_i + 1;
                end
            end

            STATE_HEAD_PREP: begin
                busy <= 1'b1;
                head_base <= current_head * HEAD_SIZE;
                att_base <= current_head * MAX_SEQ_LEN;
                kv_head <= current_head / KV_MUL;
                loff <= {29'd0, layer_idx} * MAX_SEQ_LEN * KV_DIM;
                timestep <= 0;
                loop_i <= 0;
                score_bits <= 32'd0;
                state <= STATE_SCORE_Q_REQ;
            end
            STATE_SCORE_Q_REQ: begin
                busy <= 1'b1;
                cache_base <= loff + timestep * KV_DIM + kv_head * HEAD_SIZE;
                local_act_rd_en <= 1'b1;
                local_act_rd_addr <= `ACT_Q_BASE + head_base + loop_i;
                state <= STATE_SCORE_Q_WAIT;
            end
            STATE_SCORE_Q_WAIT: begin busy <= 1'b1; state <= STATE_SCORE_Q_CAP; end
            STATE_SCORE_Q_CAP: begin
                busy <= 1'b1;
                q_bits <= act_rd_data;
                state <= STATE_SCORE_KV_REQ;
            end
            STATE_SCORE_KV_REQ: begin
                busy <= 1'b1;
                kv_rd_en <= 1'b1;
                kv_rd_addr <= `KV_KEY_BASE + cache_base + loop_i;
                state <= STATE_SCORE_KV_WAIT;
            end
            STATE_SCORE_KV_WAIT: begin busy <= 1'b1; state <= STATE_SCORE_KV_CAP; end
            STATE_SCORE_KV_CAP: begin
                busy <= 1'b1;
                kv_bits <= kv_rd_data;
                state <= STATE_SCORE_ACCUM;
            end
            STATE_SCORE_ACCUM: begin
                busy <= 1'b1;
                score_bits <= add_y;
                if (loop_i == (HEAD_SIZE - 1)) begin
                    state <= STATE_SCORE_SCALE;
                end else begin
                    loop_i <= loop_i + 1;
                    state <= STATE_SCORE_Q_REQ;
                end
            end
            STATE_SCORE_SCALE: begin
                busy <= 1'b1;
                scaled_score_bits <= mul_y;
                state <= STATE_SCORE_WRITE;
            end
            STATE_SCORE_WRITE: begin
                busy <= 1'b1;
                local_act_wr_en <= 1'b1;
                local_act_wr_addr <= `ACT_ATT_BASE + att_base + timestep;
                local_act_wr_data <= scaled_score_bits;
                if (timestep == pos_idx) begin
                    state <= STATE_SOFTMAX_START;
                end else begin
                    timestep <= timestep + 1;
                    loop_i <= 0;
                    score_bits <= 32'd0;
                    state <= STATE_SCORE_Q_REQ;
                end
            end

            STATE_SOFTMAX_START: begin busy <= 1'b1; state <= STATE_SOFTMAX_WAIT; end
            STATE_SOFTMAX_WAIT: begin
                busy <= 1'b1;
                if (softmax_done) begin
                    head_base <= current_head * HEAD_SIZE;
                    att_base <= current_head * MAX_SEQ_LEN;
                    kv_head <= current_head / KV_MUL;
                    loff <= {29'd0, layer_idx} * MAX_SEQ_LEN * KV_DIM;
                    loop_i <= 0;
                    state <= STATE_CLEAR_HEAD_WR;
                end
            end

            STATE_CLEAR_HEAD_WR: begin
                busy <= 1'b1;
                local_act_wr_en <= 1'b1;
                local_act_wr_addr <= `ACT_XB_BASE + head_base + loop_i;
                local_act_wr_data <= 32'd0;
                if (loop_i == (HEAD_SIZE - 1)) begin
                    timestep <= 0;
                    loop_i <= 0;
                    state <= STATE_MERGE_XB_REQ;
                end else begin
                    loop_i <= loop_i + 1;
                end
            end

            STATE_MERGE_XB_REQ: begin
                busy <= 1'b1;
                cache_base <= loff + timestep * KV_DIM + kv_head * HEAD_SIZE;
                local_act_rd_en <= 1'b1;
                local_act_rd_addr <= `ACT_XB_BASE + head_base + loop_i;
                state <= STATE_MERGE_XB_WAIT;
            end
            STATE_MERGE_XB_WAIT: begin busy <= 1'b1; state <= STATE_MERGE_XB_CAP; end
            STATE_MERGE_XB_CAP: begin busy <= 1'b1; xb_bits <= act_rd_data; state <= STATE_MERGE_ATT_REQ; end
            STATE_MERGE_ATT_REQ: begin
                busy <= 1'b1;
                local_act_rd_en <= 1'b1;
                local_act_rd_addr <= `ACT_ATT_BASE + att_base + timestep;
                state <= STATE_MERGE_ATT_WAIT;
            end
            STATE_MERGE_ATT_WAIT: begin busy <= 1'b1; state <= STATE_MERGE_ATT_CAP; end
            STATE_MERGE_ATT_CAP: begin busy <= 1'b1; att_bits <= act_rd_data; state <= STATE_MERGE_KV_REQ; end
            STATE_MERGE_KV_REQ: begin
                busy <= 1'b1;
                kv_rd_en <= 1'b1;
                kv_rd_addr <= `KV_VALUE_BASE + cache_base + loop_i;
                state <= STATE_MERGE_KV_WAIT;
            end
            STATE_MERGE_KV_WAIT: begin busy <= 1'b1; state <= STATE_MERGE_KV_CAP; end
            STATE_MERGE_KV_CAP: begin busy <= 1'b1; v_bits <= kv_rd_data; state <= STATE_MERGE_ACCUM; end
            STATE_MERGE_ACCUM: begin busy <= 1'b1; local_act_wr_data <= add_y; state <= STATE_MERGE_WRITE; end
            STATE_MERGE_WRITE: begin
                busy <= 1'b1;
                local_act_wr_en <= 1'b1;
                local_act_wr_addr <= `ACT_XB_BASE + head_base + loop_i;
                if (loop_i == (HEAD_SIZE - 1)) begin
                    if (timestep == pos_idx) begin
                        if (current_head == (N_HEADS - 1)) begin
                            state <= STATE_ATTN_OUT_START;
                        end else begin
                            current_head <= current_head + 1;
                            state <= STATE_HEAD_PREP;
                        end
                    end else begin
                        timestep <= timestep + 1;
                        loop_i <= 0;
                        state <= STATE_MERGE_XB_REQ;
                    end
                end else begin
                    loop_i <= loop_i + 1;
                    state <= STATE_MERGE_XB_REQ;
                end
            end

            STATE_ATTN_OUT_START: begin busy <= 1'b1; state <= STATE_ATTN_OUT_WAIT; end
            STATE_ATTN_OUT_WAIT: begin busy <= 1'b1; if (matmul_done) state <= STATE_DONE; end
            STATE_DONE: begin busy <= 1'b0; done <= 1'b1; state <= STATE_IDLE; end
            default: begin busy <= 1'b0; state <= STATE_IDLE; end
        endcase
    end
end

endmodule
