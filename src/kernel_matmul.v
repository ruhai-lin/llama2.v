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

localparam KV_DIM = (DIM * N_KV_HEADS) / N_HEADS;

`include "memory_map.vh"

localparam OP_QKV       = 3'd1;
localparam OP_ATTN_OUT  = 3'd2;
localparam OP_W1_W3     = 3'd3;
localparam OP_W2        = 3'd4;
localparam OP_CLASSIFY  = 3'd5;

localparam STATE_IDLE               = 5'd0;
localparam STATE_CONFIG             = 5'd1;
localparam STATE_LOAD_X_REQ         = 5'd2;
localparam STATE_LOAD_X_WAIT        = 5'd3;
localparam STATE_LOAD_X_CAP         = 5'd4;
localparam STATE_TILE_PREP          = 5'd5;
localparam STATE_LOAD_W_REQ         = 5'd6;
localparam STATE_LOAD_W_WAIT        = 5'd7;
localparam STATE_LOAD_W_CAP         = 5'd8;
localparam STATE_MAC_INIT           = 5'd9;
localparam STATE_MAC_ACCUM          = 5'd10;
localparam STATE_WRITE_REQ          = 5'd11;
localparam STATE_WRITE_COMMIT       = 5'd12;
localparam STATE_RESID_A_REQ        = 5'd13;
localparam STATE_RESID_A_WAIT       = 5'd14;
localparam STATE_RESID_A_CAP        = 5'd15;
localparam STATE_RESID_B_REQ        = 5'd16;
localparam STATE_RESID_B_WAIT       = 5'd17;
localparam STATE_RESID_B_CAP        = 5'd18;
localparam STATE_RESID_WRITE_REQ    = 5'd19;
localparam STATE_RESID_WRITE_COMMIT = 5'd20;
localparam STATE_DONE               = 5'd21;

integer row_i;
integer col_i;

reg [4:0] state;
reg [2:0] op_code_reg;
reg [31:0] layer_idx_reg;
reg [31:0] seq_index;
reg [31:0] row_base;
reg [31:0] tile_rows_reg;
reg [31:0] row_offset;
reg [31:0] k_index;
reg [31:0] x_index;
reg [31:0] write_index;
reg [31:0] residual_index;
reg [31:0] residual_a_bits;
reg [31:0] residual_b_bits;
reg [31:0] acc_work;
reg [31:0] max_logit_bits;
reg [TOKEN_W-1:0] best_token_reg;

reg [31:0] x_buf [0:K_MAX-1];
reg [31:0] w_buf [0:TM-1][0:K_MAX-1];
reg [31:0] acc_buf [0:TM-1];

reg [31:0] cfg_input_base;
reg [31:0] cfg_input_k_size;
reg [31:0] cfg_weight_base;
reg [31:0] cfg_output_base;
reg [31:0] cfg_m_size;
reg [31:0] cfg_k_size;
reg [31:0] cfg_seq_total;
reg cfg_need_residual;
reg cfg_classify;
reg [31:0] cfg_residual_src_a_base;
reg [31:0] cfg_residual_src_b_base;
reg [31:0] cfg_residual_dst_base;
reg [31:0] cfg_residual_len;

reg [31:0] mac_mul_a;
reg [31:0] mac_mul_b;
reg [31:0] add_in_a;
reg [31:0] add_in_b;

wire [31:0] mac_mul_y;
wire [31:0] add_y;

function fp32_gt;
    input [31:0] lhs;
    input [31:0] rhs;
    begin
        if ((rhs[30:0] == 31'd0) && (lhs[30:0] == 31'd0)) begin
            fp32_gt = 1'b0;
        end else if (lhs[31] != rhs[31]) begin
            fp32_gt = rhs[31] && !lhs[31];
        end else if (!lhs[31]) begin
            if (lhs[30:23] != rhs[30:23]) begin
                fp32_gt = lhs[30:23] > rhs[30:23];
            end else begin
                fp32_gt = lhs[22:0] > rhs[22:0];
            end
        end else begin
            if (lhs[30:23] != rhs[30:23]) begin
                fp32_gt = lhs[30:23] < rhs[30:23];
            end else begin
                fp32_gt = lhs[22:0] < rhs[22:0];
            end
        end
    end
endfunction

always @(*) begin
    cfg_input_base = `ACT_X_BASE;
    cfg_input_k_size = DIM;
    cfg_weight_base = 32'd0;
    cfg_output_base = 32'd0;
    cfg_m_size = 32'd0;
    cfg_k_size = DIM;
    cfg_seq_total = 32'd1;
    cfg_need_residual = 1'b0;
    cfg_classify = 1'b0;
    cfg_residual_src_a_base = 32'd0;
    cfg_residual_src_b_base = 32'd0;
    cfg_residual_dst_base = 32'd0;
    cfg_residual_len = 32'd0;

    case (op_code_reg)
        OP_QKV: begin
            cfg_input_base = `ACT_XB_BASE;
            cfg_input_k_size = DIM;
            cfg_seq_total = 32'd3;
            cfg_k_size = DIM;
            if (seq_index == 32'd0) begin
                cfg_weight_base = `WGT_WQ_BASE + layer_idx_reg * DIM * DIM;
                cfg_output_base = `ACT_Q_BASE;
                cfg_m_size = DIM;
            end else if (seq_index == 32'd1) begin
                cfg_weight_base = `WGT_WK_BASE + layer_idx_reg * DIM * KV_DIM;
                cfg_output_base = `ACT_K_BASE;
                cfg_m_size = KV_DIM;
            end else begin
                cfg_weight_base = `WGT_WV_BASE + layer_idx_reg * DIM * KV_DIM;
                cfg_output_base = `ACT_V_BASE;
                cfg_m_size = KV_DIM;
            end
        end
        OP_ATTN_OUT: begin
            cfg_input_base = `ACT_XB_BASE;
            cfg_input_k_size = DIM;
            cfg_seq_total = 32'd1;
            cfg_weight_base = `WGT_WO_BASE + layer_idx_reg * DIM * DIM;
            cfg_output_base = `ACT_XB2_BASE;
            cfg_m_size = DIM;
            cfg_k_size = DIM;
            cfg_need_residual = 1'b1;
            cfg_residual_src_a_base = `ACT_X_BASE;
            cfg_residual_src_b_base = `ACT_XB2_BASE;
            cfg_residual_dst_base = `ACT_X_BASE;
            cfg_residual_len = DIM;
        end
        OP_W1_W3: begin
            cfg_input_base = `ACT_XB_BASE;
            cfg_input_k_size = DIM;
            cfg_seq_total = 32'd2;
            cfg_k_size = DIM;
            if (seq_index == 32'd0) begin
                cfg_weight_base = `WGT_W1_BASE + layer_idx_reg * DIM * HIDDEN_DIM;
                cfg_output_base = `ACT_HB_BASE;
            end else begin
                cfg_weight_base = `WGT_W3_BASE + layer_idx_reg * DIM * HIDDEN_DIM;
                cfg_output_base = `ACT_HB2_BASE;
            end
            cfg_m_size = HIDDEN_DIM;
        end
        OP_W2: begin
            cfg_input_base = `ACT_HB_BASE;
            cfg_input_k_size = HIDDEN_DIM;
            cfg_seq_total = 32'd1;
            cfg_weight_base = `WGT_W2_BASE + layer_idx_reg * HIDDEN_DIM * DIM;
            cfg_output_base = `ACT_XB_BASE;
            cfg_m_size = DIM;
            cfg_k_size = HIDDEN_DIM;
            cfg_need_residual = 1'b1;
            cfg_residual_src_a_base = `ACT_X_BASE;
            cfg_residual_src_b_base = `ACT_XB_BASE;
            cfg_residual_dst_base = `ACT_X_BASE;
            cfg_residual_len = DIM;
        end
        OP_CLASSIFY: begin
            cfg_input_base = `ACT_X_BASE;
            cfg_input_k_size = DIM;
            cfg_seq_total = 32'd1;
            cfg_weight_base = `WGT_TOKEN_EMBED_BASE;
            cfg_output_base = `ACT_LOGITS_BASE;
            cfg_m_size = VOCAB_SIZE;
            cfg_k_size = DIM;
            cfg_classify = 1'b1;
        end
        default: begin
        end
    endcase
end

always @(*) begin
    mac_mul_a = 32'd0;
    mac_mul_b = 32'd0;
    add_in_a = 32'd0;
    add_in_b = 32'd0;

    if (state == STATE_MAC_ACCUM) begin
        mac_mul_a = w_buf[row_offset][k_index];
        mac_mul_b = x_buf[k_index];
        add_in_a = acc_work;
        add_in_b = mac_mul_y;
    end else if (state == STATE_RESID_WRITE_REQ) begin
        add_in_a = residual_a_bits;
        add_in_b = residual_b_bits;
    end
end

kernel_mul u_mul (
    .a(mac_mul_a),
    .b(mac_mul_b),
    .y(mac_mul_y)
);

kernel_add u_add (
    .a(add_in_a),
    .b(add_in_b),
    .y(add_y)
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= STATE_IDLE;
        op_code_reg <= 3'd0;
        layer_idx_reg <= 32'd0;
        seq_index <= 32'd0;
        row_base <= 32'd0;
        tile_rows_reg <= 32'd0;
        row_offset <= 32'd0;
        k_index <= 32'd0;
        x_index <= 32'd0;
        write_index <= 32'd0;
        residual_index <= 32'd0;
        residual_a_bits <= 32'd0;
        residual_b_bits <= 32'd0;
        acc_work <= 32'd0;
        max_logit_bits <= 32'd0;
        best_token_reg <= {TOKEN_W{1'b0}};
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
        for (row_i = 0; row_i < K_MAX; row_i = row_i + 1) begin
            x_buf[row_i] <= 32'd0;
        end
        for (row_i = 0; row_i < TM; row_i = row_i + 1) begin
            acc_buf[row_i] <= 32'd0;
            for (col_i = 0; col_i < K_MAX; col_i = col_i + 1) begin
                w_buf[row_i][col_i] <= 32'd0;
            end
        end
    end else begin
        done <= 1'b0;
        act_rd_en <= 1'b0;
        act_wr_en <= 1'b0;
        wgt_rd_en <= 1'b0;

        case (state)
            STATE_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    busy <= 1'b1;
                    op_code_reg <= ((CONTROL_MODE == 4) && (op_code == 3'd0)) ? OP_CLASSIFY : op_code;
                    layer_idx_reg <= layer_idx;
                    seq_index <= 32'd0;
                    next_token <= {TOKEN_W{1'b0}};
                    flat_logits <= {LOGITS_W{1'b0}};
                    max_logit_bits <= 32'd0;
                    best_token_reg <= {TOKEN_W{1'b0}};
                    state <= STATE_CONFIG;
                end
            end

            STATE_CONFIG: begin
                busy <= 1'b1;
                x_index <= 32'd0;
                row_base <= 32'd0;
                residual_index <= 32'd0;
                state <= STATE_LOAD_X_REQ;
            end

            STATE_LOAD_X_REQ: begin
                busy <= 1'b1;
                act_rd_en <= 1'b1;
                act_rd_addr <= cfg_input_base + x_index;
                state <= STATE_LOAD_X_WAIT;
            end

            STATE_LOAD_X_WAIT: begin
                busy <= 1'b1;
                state <= STATE_LOAD_X_CAP;
            end

            STATE_LOAD_X_CAP: begin
                busy <= 1'b1;
                x_buf[x_index] <= act_rd_data;
                if (x_index == (cfg_input_k_size - 1)) begin
                    state <= STATE_TILE_PREP;
                end else begin
                    x_index <= x_index + 32'd1;
                    state <= STATE_LOAD_X_REQ;
                end
            end

            STATE_TILE_PREP: begin
                busy <= 1'b1;
                row_offset <= 32'd0;
                k_index <= 32'd0;
                write_index <= 32'd0;
                if ((row_base + TM) <= cfg_m_size) begin
                    tile_rows_reg <= TM;
                end else begin
                    tile_rows_reg <= cfg_m_size - row_base;
                end
                state <= STATE_LOAD_W_REQ;
            end

            STATE_LOAD_W_REQ: begin
                busy <= 1'b1;
                wgt_rd_en <= 1'b1;
                wgt_rd_addr <= cfg_weight_base + (row_base + row_offset) * cfg_k_size + k_index;
                state <= STATE_LOAD_W_WAIT;
            end

            STATE_LOAD_W_WAIT: begin
                busy <= 1'b1;
                state <= STATE_LOAD_W_CAP;
            end

            STATE_LOAD_W_CAP: begin
                busy <= 1'b1;
                w_buf[row_offset][k_index] <= wgt_rd_data;
                if (k_index == (cfg_k_size - 1)) begin
                    if (row_offset == (tile_rows_reg - 1)) begin
                        state <= STATE_MAC_INIT;
                    end else begin
                        row_offset <= row_offset + 32'd1;
                        k_index <= 32'd0;
                        state <= STATE_LOAD_W_REQ;
                    end
                end else begin
                    k_index <= k_index + 32'd1;
                    state <= STATE_LOAD_W_REQ;
                end
            end

            STATE_MAC_INIT: begin
                busy <= 1'b1;
                row_offset <= 32'd0;
                k_index <= 32'd0;
                acc_work <= 32'd0;
                state <= STATE_MAC_ACCUM;
            end

            STATE_MAC_ACCUM: begin
                busy <= 1'b1;
                if (k_index == (cfg_k_size - 1)) begin
                    acc_buf[row_offset] <= add_y;
                    if (row_offset == (tile_rows_reg - 1)) begin
                        write_index <= 32'd0;
                        state <= STATE_WRITE_REQ;
                    end else begin
                        row_offset <= row_offset + 32'd1;
                        k_index <= 32'd0;
                        acc_work <= 32'd0;
                    end
                end else begin
                    acc_work <= add_y;
                    k_index <= k_index + 32'd1;
                end
            end

            STATE_WRITE_REQ: begin
                busy <= 1'b1;
                act_wr_en <= 1'b1;
                act_wr_addr <= cfg_output_base + row_base + write_index;
                act_wr_data <= acc_buf[write_index];
                state <= STATE_WRITE_COMMIT;
            end

            STATE_WRITE_COMMIT: begin
                busy <= 1'b1;
                if (cfg_classify) begin
                    flat_logits[(row_base + write_index) * LOGIT_W +: LOGIT_W] <= acc_buf[write_index];
                    if (((row_base + write_index) == 0) || fp32_gt(acc_buf[write_index], max_logit_bits)) begin
                        max_logit_bits <= acc_buf[write_index];
                        best_token_reg <= row_base + write_index;
                    end
                end

                if (write_index == (tile_rows_reg - 1)) begin
                    if ((row_base + tile_rows_reg) >= cfg_m_size) begin
                        if ((seq_index + 32'd1) < cfg_seq_total) begin
                            seq_index <= seq_index + 32'd1;
                            row_base <= 32'd0;
                            state <= STATE_TILE_PREP;
                        end else if (cfg_need_residual) begin
                            residual_index <= 32'd0;
                            state <= STATE_RESID_A_REQ;
                        end else begin
                            if (cfg_classify) begin
                                next_token <= best_token_reg;
                            end
                            state <= STATE_DONE;
                        end
                    end else begin
                        row_base <= row_base + tile_rows_reg;
                        state <= STATE_TILE_PREP;
                    end
                end else begin
                    write_index <= write_index + 32'd1;
                    state <= STATE_WRITE_REQ;
                end
            end

            STATE_RESID_A_REQ: begin
                busy <= 1'b1;
                act_rd_en <= 1'b1;
                act_rd_addr <= cfg_residual_src_a_base + residual_index;
                state <= STATE_RESID_A_WAIT;
            end

            STATE_RESID_A_WAIT: begin
                busy <= 1'b1;
                state <= STATE_RESID_A_CAP;
            end

            STATE_RESID_A_CAP: begin
                busy <= 1'b1;
                residual_a_bits <= act_rd_data;
                state <= STATE_RESID_B_REQ;
            end

            STATE_RESID_B_REQ: begin
                busy <= 1'b1;
                act_rd_en <= 1'b1;
                act_rd_addr <= cfg_residual_src_b_base + residual_index;
                state <= STATE_RESID_B_WAIT;
            end

            STATE_RESID_B_WAIT: begin
                busy <= 1'b1;
                state <= STATE_RESID_B_CAP;
            end

            STATE_RESID_B_CAP: begin
                busy <= 1'b1;
                residual_b_bits <= act_rd_data;
                state <= STATE_RESID_WRITE_REQ;
            end

            STATE_RESID_WRITE_REQ: begin
                busy <= 1'b1;
                act_wr_en <= 1'b1;
                act_wr_addr <= cfg_residual_dst_base + residual_index;
                act_wr_data <= add_y;
                state <= STATE_RESID_WRITE_COMMIT;
            end

            STATE_RESID_WRITE_COMMIT: begin
                busy <= 1'b1;
                if (residual_index == (cfg_residual_len - 1)) begin
                    state <= STATE_DONE;
                end else begin
                    residual_index <= residual_index + 32'd1;
                    state <= STATE_RESID_A_REQ;
                end
            end

            STATE_DONE: begin
                busy <= 1'b0;
                done <= 1'b1;
                next_token <= best_token_reg;
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
