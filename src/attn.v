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

`include "real_fp32_helpers.vh"

localparam VOCAB_SIZE = 512;
localparam N_LAYERS = 5;
localparam HIDDEN_DIM = 172;
localparam HEAD_SIZE = DIM / N_HEADS;
localparam KV_DIM = (DIM * N_KV_HEADS) / N_HEADS;
localparam KV_MUL = N_HEADS / N_KV_HEADS;

`include "memory_map.vh"

localparam STATE_IDLE            = 5'd0;
localparam STATE_RMS_START       = 5'd1;
localparam STATE_RMS_WAIT        = 5'd2;
localparam STATE_QKV_START       = 5'd3;
localparam STATE_QKV_WAIT        = 5'd4;
localparam STATE_ROPE_START      = 5'd5;
localparam STATE_ROPE_WAIT       = 5'd6;
localparam STATE_KV_COPY         = 5'd7;
localparam STATE_HEAD_SCORE      = 5'd8;
localparam STATE_SOFTMAX_START   = 5'd9;
localparam STATE_SOFTMAX_WAIT    = 5'd10;
localparam STATE_HEAD_MERGE      = 5'd11;
localparam STATE_ATTN_OUT_START  = 5'd12;
localparam STATE_ATTN_OUT_WAIT   = 5'd13;
localparam STATE_DONE            = 5'd14;

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
reg [4:0] state;
integer current_head;

integer i;
integer timestep;
integer head_base;
integer kv_head;
integer cache_base;
integer att_base;
integer loff;
real kv_value;
real score;
real inv_scale;
real act_value_a;
real act_value_b;

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
end

task read_kv;
    input integer addr;
    output real value;
    begin
        kv_rd_addr = addr;
        kv_rd_en = 1'b1;
        @(posedge clk);
        kv_rd_en = 1'b0;
        @(negedge clk);
        value = fp32_to_real(kv_rd_data);
    end
endtask

task write_kv;
    input integer addr;
    input real value;
    begin
        kv_wr_addr = addr;
        kv_wr_data = real_to_fp32_bits(value);
        kv_wr_en = 1'b1;
        @(posedge clk);
        kv_wr_en = 1'b0;
    end
endtask

task read_act_local;
    input integer addr;
    output real value;
    begin
        local_act_rd_addr = addr;
        local_act_rd_en = 1'b1;
        @(posedge clk);
        local_act_rd_en = 1'b0;
        @(negedge clk);
        value = fp32_to_real(act_rd_data);
    end
endtask

task write_act_local;
    input integer addr;
    input real value;
    begin
        local_act_wr_addr = addr;
        local_act_wr_data = real_to_fp32_bits(value);
        local_act_wr_en = 1'b1;
        @(posedge clk);
        local_act_wr_en = 1'b0;
    end
endtask

task copy_kv_cache;
    input integer task_layer_idx;
    input integer task_pos_idx;
    begin
        loff = (task_layer_idx * MAX_SEQ_LEN + task_pos_idx) * KV_DIM;
        for (i = 0; i < KV_DIM; i = i + 1) begin
            read_act_local(`ACT_K_BASE + i, kv_value);
            write_kv(`KV_KEY_BASE + loff + i, kv_value);
            read_act_local(`ACT_V_BASE + i, kv_value);
            write_kv(`KV_VALUE_BASE + loff + i, kv_value);
        end
    end
endtask

task clear_xb_all;
    begin
        for (i = 0; i < DIM; i = i + 1) begin
            write_act_local(`ACT_XB_BASE + i, 0.0);
        end
    end
endtask

task compute_scores_for_head;
    input integer task_layer_idx;
    input integer task_pos_idx;
    input integer head_idx;
    begin
        inv_scale = 1.0 / $sqrt(HEAD_SIZE);
        head_base = head_idx * HEAD_SIZE;
        att_base = head_idx * MAX_SEQ_LEN;
        kv_head = head_idx / KV_MUL;
        loff = task_layer_idx * MAX_SEQ_LEN * KV_DIM;

        for (timestep = 0; timestep <= task_pos_idx; timestep = timestep + 1) begin
            cache_base = loff + timestep * KV_DIM + kv_head * HEAD_SIZE;
            score = 0.0;
            for (i = 0; i < HEAD_SIZE; i = i + 1) begin
                read_act_local(`ACT_Q_BASE + head_base + i, act_value_a);
                read_kv(`KV_KEY_BASE + cache_base + i, kv_value);
                score = score + (act_value_a * kv_value);
            end
            write_act_local(`ACT_ATT_BASE + att_base + timestep, fp32_round(score * inv_scale));
        end
    end
endtask

task clear_xb_head;
    input integer head_idx;
    begin
        head_base = head_idx * HEAD_SIZE;
        for (i = 0; i < HEAD_SIZE; i = i + 1) begin
            write_act_local(`ACT_XB_BASE + head_base + i, 0.0);
        end
    end
endtask

task merge_head_values;
    input integer task_layer_idx;
    input integer task_pos_idx;
    input integer head_idx;
    begin
        head_base = head_idx * HEAD_SIZE;
        att_base = head_idx * MAX_SEQ_LEN;
        kv_head = head_idx / KV_MUL;
        loff = task_layer_idx * MAX_SEQ_LEN * KV_DIM;

        for (timestep = 0; timestep <= task_pos_idx; timestep = timestep + 1) begin
            cache_base = loff + timestep * KV_DIM + kv_head * HEAD_SIZE;
            for (i = 0; i < HEAD_SIZE; i = i + 1) begin
                read_act_local(`ACT_XB_BASE + head_base + i, act_value_a);
                read_act_local(`ACT_ATT_BASE + att_base + timestep, act_value_b);
                read_kv(`KV_VALUE_BASE + cache_base + i, kv_value);
                write_act_local(
                    `ACT_XB_BASE + head_base + i,
                    fp32_round(act_value_a + (act_value_b * kv_value))
                );
            end
        end
    end
endtask

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
    end else begin
        done <= 1'b0;
        case (state)
            STATE_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    busy <= 1'b1;
                    state <= STATE_RMS_START;
                end
            end

            STATE_RMS_START: begin
                busy <= 1'b1;
                state <= STATE_RMS_WAIT;
            end

            STATE_RMS_WAIT: begin
                busy <= 1'b1;
                if (rms_done) begin
                    state <= STATE_QKV_START;
                end
            end

            STATE_QKV_START: begin
                busy <= 1'b1;
                state <= STATE_QKV_WAIT;
            end

            STATE_QKV_WAIT: begin
                busy <= 1'b1;
                if (matmul_done) begin
                    state <= STATE_ROPE_START;
                end
            end

            STATE_ROPE_START: begin
                busy <= 1'b1;
                state <= STATE_ROPE_WAIT;
            end

            STATE_ROPE_WAIT: begin
                busy <= 1'b1;
                if (rope_done) begin
                    state <= STATE_KV_COPY;
                end
            end

            STATE_KV_COPY: begin
                busy <= 1'b1;
                copy_kv_cache(layer_idx, pos_idx);
                clear_xb_all();
                current_head <= 0;
                state <= STATE_HEAD_SCORE;
            end

            STATE_HEAD_SCORE: begin
                busy <= 1'b1;
                compute_scores_for_head(layer_idx, pos_idx, current_head);
                state <= STATE_SOFTMAX_START;
            end

            STATE_SOFTMAX_START: begin
                busy <= 1'b1;
                state <= STATE_SOFTMAX_WAIT;
            end

            STATE_SOFTMAX_WAIT: begin
                busy <= 1'b1;
                if (softmax_done) begin
                    state <= STATE_HEAD_MERGE;
                end
            end

            STATE_HEAD_MERGE: begin
                busy <= 1'b1;
                clear_xb_head(current_head);
                merge_head_values(layer_idx, pos_idx, current_head);
                if (current_head == (N_HEADS - 1)) begin
                    state <= STATE_ATTN_OUT_START;
                end else begin
                    current_head <= current_head + 1;
                    state <= STATE_HEAD_SCORE;
                end
            end

            STATE_ATTN_OUT_START: begin
                busy <= 1'b1;
                state <= STATE_ATTN_OUT_WAIT;
            end

            STATE_ATTN_OUT_WAIT: begin
                busy <= 1'b1;
                if (matmul_done) begin
                    state <= STATE_DONE;
                end
            end

            STATE_DONE: begin
                busy <= 1'b0;
                done <= 1'b1;
                state <= STATE_IDLE;
            end

            default: begin
                busy <= 1'b0;
                state <= STATE_IDLE;
            end
        endcase
    end
end

endmodule
