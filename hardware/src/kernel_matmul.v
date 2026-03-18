module kernel_matmul #(
    parameter VOCAB_SIZE = 512,
    parameter TOKEN_W = 9,
    parameter LOGIT_W = 32,
    parameter LOGITS_W = VOCAB_SIZE * LOGIT_W,
    parameter DIM = 64,
    parameter HIDDEN_DIM = 172,
    parameter N_LAYERS = 5,
    parameter N_HEADS = 8,
    parameter N_KV_HEADS = 4,
    parameter MAX_SEQ_LEN = 512,
    parameter CONTROL_MODE = 0
) (
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [31:0] layer_idx,
    output reg busy,
    output reg done,
    output reg [TOKEN_W-1:0] next_token,
    output reg [LOGITS_W-1:0] flat_logits,
    output reg act_rd_en,
    output reg [31:0] act_rd_addr,
    input wire [31:0] act_rd_data,
    output reg act_wr_en,
    output reg [31:0] act_wr_addr,
    output reg [31:0] act_wr_data,
    output reg wgt_rd_en,
    output reg [31:0] wgt_rd_addr,
    input wire [31:0] wgt_rd_data
);

`include "real_fp32_helpers.vh"

localparam KV_DIM = (DIM * N_KV_HEADS) / N_HEADS;

`include "memory_map.vh"

integer i;
integer j;
integer base_idx;
real acc;
real max_logit;
integer best_token;
real act_value;
real wgt_value;

task read_act;
    input integer addr;
    output real value;
    begin
        act_rd_addr = addr;
        act_rd_en = 1'b1;
        @(posedge clk);
        act_rd_en = 1'b0;
        @(negedge clk);
        value = fp32_to_real(act_rd_data);
    end
endtask

task write_act;
    input integer addr;
    input real value;
    begin
        act_wr_addr = addr;
        act_wr_data = real_to_fp32_bits(value);
        act_wr_en = 1'b1;
        @(posedge clk);
        act_wr_en = 1'b0;
    end
endtask

task read_wgt;
    input integer addr;
    output real value;
    begin
        wgt_rd_addr = addr;
        wgt_rd_en = 1'b1;
        @(posedge clk);
        wgt_rd_en = 1'b0;
        @(negedge clk);
        value = fp32_to_real(wgt_rd_data);
    end
endtask

task project_qkv;
    input integer task_layer_idx;
    integer base_q;
    integer base_k;
    integer base_v;
    begin
        base_q = `WGT_WQ_BASE + task_layer_idx * DIM * DIM;
        base_k = `WGT_WK_BASE + task_layer_idx * DIM * KV_DIM;
        base_v = `WGT_WV_BASE + task_layer_idx * DIM * KV_DIM;

        for (i = 0; i < DIM; i = i + 1) begin
            acc = 0.0;
            for (j = 0; j < DIM; j = j + 1) begin
                read_wgt(base_q + i * DIM + j, wgt_value);
                read_act(`ACT_XB_BASE + j, act_value);
                acc = acc + (wgt_value * act_value);
            end
            write_act(`ACT_Q_BASE + i, fp32_round(acc));
        end

        for (i = 0; i < KV_DIM; i = i + 1) begin
            acc = 0.0;
            for (j = 0; j < DIM; j = j + 1) begin
                read_wgt(base_k + i * DIM + j, wgt_value);
                read_act(`ACT_XB_BASE + j, act_value);
                acc = acc + (wgt_value * act_value);
            end
            write_act(`ACT_K_BASE + i, fp32_round(acc));
        end

        for (i = 0; i < KV_DIM; i = i + 1) begin
            acc = 0.0;
            for (j = 0; j < DIM; j = j + 1) begin
                read_wgt(base_v + i * DIM + j, wgt_value);
                read_act(`ACT_XB_BASE + j, act_value);
                acc = acc + (wgt_value * act_value);
            end
            write_act(`ACT_V_BASE + i, fp32_round(acc));
        end
    end
endtask

task project_attention_output;
    input integer task_layer_idx;
    integer base_o;
    real xb_val;
    begin
        base_o = `WGT_WO_BASE + task_layer_idx * DIM * DIM;
        for (i = 0; i < DIM; i = i + 1) begin
            acc = 0.0;
            for (j = 0; j < DIM; j = j + 1) begin
                read_wgt(base_o + i * DIM + j, wgt_value);
                read_act(`ACT_XB_BASE + j, act_value);
                acc = acc + (wgt_value * act_value);
            end
            write_act(`ACT_XB2_BASE + i, fp32_round(acc));
        end

        for (i = 0; i < DIM; i = i + 1) begin
            read_act(`ACT_X_BASE + i, act_value);
            read_act(`ACT_XB2_BASE + i, xb_val);
            write_act(`ACT_X_BASE + i, fp32_round(act_value + xb_val));
        end
    end
endtask

task project_w1_w3;
    input integer task_layer_idx;
    integer base_w1;
    integer base_w3;
    begin
        base_w1 = `WGT_W1_BASE + task_layer_idx * DIM * HIDDEN_DIM;
        base_w3 = `WGT_W3_BASE + task_layer_idx * DIM * HIDDEN_DIM;
        for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
            acc = 0.0;
            for (j = 0; j < DIM; j = j + 1) begin
                read_wgt(base_w1 + i * DIM + j, wgt_value);
                read_act(`ACT_XB_BASE + j, act_value);
                acc = acc + (wgt_value * act_value);
            end
            write_act(`ACT_HB_BASE + i, fp32_round(acc));

            acc = 0.0;
            for (j = 0; j < DIM; j = j + 1) begin
                read_wgt(base_w3 + i * DIM + j, wgt_value);
                read_act(`ACT_XB_BASE + j, act_value);
                acc = acc + (wgt_value * act_value);
            end
            write_act(`ACT_HB2_BASE + i, fp32_round(acc));
        end
    end
endtask

task project_w2;
    input integer task_layer_idx;
    integer base_w2;
    real xb_val;
    begin
        base_w2 = `WGT_W2_BASE + task_layer_idx * HIDDEN_DIM * DIM;
        for (i = 0; i < DIM; i = i + 1) begin
            acc = 0.0;
            for (j = 0; j < HIDDEN_DIM; j = j + 1) begin
                read_wgt(base_w2 + i * HIDDEN_DIM + j, wgt_value);
                read_act(`ACT_HB_BASE + j, act_value);
                acc = acc + (wgt_value * act_value);
            end
            write_act(`ACT_XB_BASE + i, fp32_round(acc));
        end

        for (i = 0; i < DIM; i = i + 1) begin
            read_act(`ACT_X_BASE + i, act_value);
            read_act(`ACT_XB_BASE + i, xb_val);
            write_act(`ACT_X_BASE + i, fp32_round(act_value + xb_val));
        end
    end
endtask

task classify;
    output [TOKEN_W-1:0] task_next_token;
    output [LOGITS_W-1:0] task_flat_logits;
    begin
        best_token = 0;
        max_logit = 0.0;
        for (i = 0; i < VOCAB_SIZE; i = i + 1) begin
            acc = 0.0;
            base_idx = `WGT_TOKEN_EMBED_BASE + i * DIM;
            for (j = 0; j < DIM; j = j + 1) begin
                read_wgt(base_idx + j, wgt_value);
                read_act(`ACT_X_BASE + j, act_value);
                acc = acc + (wgt_value * act_value);
            end
            act_value = fp32_round(acc);
            write_act(`ACT_LOGITS_BASE + i, act_value);
            task_flat_logits[i * LOGIT_W +: LOGIT_W] = real_to_fp32_bits(act_value);
            if ((i == 0) || (act_value > max_logit)) begin
                max_logit = act_value;
                best_token = i;
            end
        end
        task_next_token = best_token[TOKEN_W-1:0];
    end
endtask

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        busy <= 1'b0;
        done <= 1'b0;
        next_token <= {TOKEN_W{1'b0}};
        flat_logits <= {LOGITS_W{1'b0}};
        act_rd_en <= 1'b0;
        act_rd_addr <= 32'd0;
        act_wr_en <= 1'b0;
        act_wr_addr <= 32'd0;
        act_wr_data <= 32'd0;
        wgt_rd_en <= 1'b0;
        wgt_rd_addr <= 32'd0;
    end else begin
        done <= 1'b0;
        if (start) begin
            busy <= 1'b1;
            if (CONTROL_MODE == 4) begin
                classify(next_token, flat_logits);
            end
            busy <= 1'b0;
            done <= 1'b1;
        end else begin
            busy <= 1'b0;
        end
    end
end

endmodule
