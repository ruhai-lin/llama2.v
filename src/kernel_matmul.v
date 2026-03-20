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
    parameter CONTROL_MODE = 0,
    parameter TM = 8,
    parameter TN = 1,
    parameter TK = 172,
    parameter K_MAX = 172
) (
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [31:0] layer_idx,
    input wire [2:0] op_code,
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

localparam PHASE_IDLE        = 3'd0;
localparam PHASE_LOAD_X      = 3'd1;
localparam PHASE_LOAD_W_TILE = 3'd2;
localparam PHASE_COMPUTE     = 3'd3;
localparam PHASE_WRITE_C     = 3'd4;
localparam OP_QKV            = 3'd1;
localparam OP_ATTN_OUT       = 3'd2;
localparam OP_W1_W3          = 3'd3;
localparam OP_W2             = 3'd4;
localparam OP_CLASSIFY       = 3'd5;

integer i;
integer j;
integer base_idx;
real acc;
real max_logit;
integer best_token;
real act_value;
real wgt_value;
reg [2:0] phase;

reg [31:0] x_buf [0:K_MAX-1];
reg [31:0] w_buf [0:TM-1][0:K_MAX-1];
reg [31:0] acc_buf [0:TM-1];

task read_act_bits;
    input integer addr;
    output [31:0] bits;
    begin
        act_rd_addr = addr;
        act_rd_en = 1'b1;
        @(posedge clk);
        act_rd_en = 1'b0;
        @(negedge clk);
        bits = act_rd_data;
    end
endtask

task write_act_bits;
    input integer addr;
    input [31:0] bits;
    begin
        act_wr_addr = addr;
        act_wr_data = bits;
        act_wr_en = 1'b1;
        @(posedge clk);
        act_wr_en = 1'b0;
    end
endtask

task read_act;
    input integer addr;
    output real value;
    reg [31:0] bits;
    begin
        read_act_bits(addr, bits);
        value = fp32_to_real(bits);
    end
endtask

task write_act;
    input integer addr;
    input real value;
    begin
        write_act_bits(addr, real_to_fp32_bits(value));
    end
endtask

task read_wgt_bits;
    input integer addr;
    output [31:0] bits;
    begin
        wgt_rd_addr = addr;
        wgt_rd_en = 1'b1;
        @(posedge clk);
        wgt_rd_en = 1'b0;
        @(negedge clk);
        bits = wgt_rd_data;
    end
endtask

task read_wgt;
    input integer addr;
    output real value;
    reg [31:0] bits;
    begin
        read_wgt_bits(addr, bits);
        value = fp32_to_real(bits);
    end
endtask

task load_x_buffer;
    input integer input_base_addr;
    input integer k_size;
    integer k;
    reg [31:0] bits;
    begin
        phase = PHASE_LOAD_X;
        for (k = 0; k < k_size; k = k + 1) begin
            read_act_bits(input_base_addr + k, bits);
            x_buf[k] = bits;
        end
    end
endtask

task load_w_tile;
    input integer weight_base_addr;
    input integer row_base;
    input integer tile_rows;
    input integer k_size;
    integer row_offset;
    integer k;
    reg [31:0] bits;
    begin
        phase = PHASE_LOAD_W_TILE;
        for (row_offset = 0; row_offset < tile_rows; row_offset = row_offset + 1) begin
            for (k = 0; k < k_size; k = k + 1) begin
                read_wgt_bits(weight_base_addr + (row_base + row_offset) * k_size + k, bits);
                w_buf[row_offset][k] = bits;
            end
        end
    end
endtask

task compute_tile;
    input integer tile_rows;
    input integer k_size;
    integer row_offset;
    integer k;
    real acc_real;
    begin
        phase = PHASE_COMPUTE;
        for (row_offset = 0; row_offset < tile_rows; row_offset = row_offset + 1) begin
            acc_real = 0.0;
            for (k = 0; k < k_size; k = k + 1) begin
                acc_real = acc_real
                    + (fp32_to_real(w_buf[row_offset][k]) * fp32_to_real(x_buf[k]));
            end
            acc_buf[row_offset] = real_to_fp32_bits(fp32_round(acc_real));
        end
    end
endtask

task write_c_tile;
    input integer output_base_addr;
    input integer row_base;
    input integer tile_rows;
    integer row_offset;
    begin
        phase = PHASE_WRITE_C;
        for (row_offset = 0; row_offset < tile_rows; row_offset = row_offset + 1) begin
            write_act_bits(output_base_addr + row_base + row_offset, acc_buf[row_offset]);
        end
    end
endtask

task matvec_tiled_from_xbuf;
    input integer weight_base_addr;
    input integer output_base_addr;
    input integer m_size;
    input integer k_size;
    integer row_base;
    integer tile_rows;
    begin
        for (row_base = 0; row_base < m_size; row_base = row_base + TM) begin
            if ((row_base + TM) <= m_size) begin
                tile_rows = TM;
            end else begin
                tile_rows = m_size - row_base;
            end
            load_w_tile(weight_base_addr, row_base, tile_rows, k_size);
            compute_tile(tile_rows, k_size);
            write_c_tile(output_base_addr, row_base, tile_rows);
        end
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

        load_x_buffer(`ACT_XB_BASE, DIM);
        matvec_tiled_from_xbuf(base_q, `ACT_Q_BASE, DIM, DIM);
        matvec_tiled_from_xbuf(base_k, `ACT_K_BASE, KV_DIM, DIM);
        matvec_tiled_from_xbuf(base_v, `ACT_V_BASE, KV_DIM, DIM);
    end
endtask

task project_attention_output;
    input integer task_layer_idx;
    integer base_o;
    real xb_val;
    begin
        base_o = `WGT_WO_BASE + task_layer_idx * DIM * DIM;
        load_x_buffer(`ACT_XB_BASE, DIM);
        matvec_tiled_from_xbuf(base_o, `ACT_XB2_BASE, DIM, DIM);

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

        load_x_buffer(`ACT_XB_BASE, DIM);
        matvec_tiled_from_xbuf(base_w1, `ACT_HB_BASE, HIDDEN_DIM, DIM);
        matvec_tiled_from_xbuf(base_w3, `ACT_HB2_BASE, HIDDEN_DIM, DIM);
    end
endtask

task project_w2;
    input integer task_layer_idx;
    integer base_w2;
    real xb_val;
    begin
        base_w2 = `WGT_W2_BASE + task_layer_idx * HIDDEN_DIM * DIM;
        load_x_buffer(`ACT_HB_BASE, HIDDEN_DIM);
        matvec_tiled_from_xbuf(base_w2, `ACT_XB_BASE, DIM, HIDDEN_DIM);

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
    integer row_base;
    integer tile_rows;
    integer row_offset;
    begin
        best_token = 0;
        max_logit = 0.0;
        load_x_buffer(`ACT_X_BASE, DIM);

        for (row_base = 0; row_base < VOCAB_SIZE; row_base = row_base + TM) begin
            if ((row_base + TM) <= VOCAB_SIZE) begin
                tile_rows = TM;
            end else begin
                tile_rows = VOCAB_SIZE - row_base;
            end

            load_w_tile(`WGT_TOKEN_EMBED_BASE, row_base, tile_rows, DIM);
            compute_tile(tile_rows, DIM);
            phase = PHASE_WRITE_C;
            for (row_offset = 0; row_offset < tile_rows; row_offset = row_offset + 1) begin
                write_act_bits(`ACT_LOGITS_BASE + row_base + row_offset, acc_buf[row_offset]);
                act_value = fp32_to_real(acc_buf[row_offset]);
                task_flat_logits[(row_base + row_offset) * LOGIT_W +: LOGIT_W] = acc_buf[row_offset];
                if (((row_base + row_offset) == 0) || (act_value > max_logit)) begin
                    max_logit = act_value;
                    best_token = row_base + row_offset;
                end
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
        phase <= PHASE_IDLE;
    end else begin
        done <= 1'b0;
        if (start) begin
            busy <= 1'b1;
            if ((op_code == OP_CLASSIFY) || (CONTROL_MODE == 4)) begin
                classify(next_token, flat_logits);
            end else if (op_code == OP_QKV) begin
                project_qkv(layer_idx);
            end else if (op_code == OP_ATTN_OUT) begin
                project_attention_output(layer_idx);
            end else if (op_code == OP_W1_W3) begin
                project_w1_w3(layer_idx);
            end else if (op_code == OP_W2) begin
                project_w2(layer_idx);
            end
            phase <= PHASE_IDLE;
            busy <= 1'b0;
            done <= 1'b1;
        end else begin
            phase <= PHASE_IDLE;
            busy <= 1'b0;
        end
    end
end

endmodule
