module kernel_rmsnorm #(
    parameter DIM = 64,
    parameter CONTROL_MODE = 0
) (
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [31:0] layer_idx,
    input wire [1:0] op_code,
    output reg busy,
    output reg done,
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

localparam OP_ATTN  = 2'd1;
localparam OP_FFN   = 2'd2;
localparam OP_FINAL = 2'd3;

localparam STATE_IDLE         = 5'd0;
localparam STATE_SUM_REQ      = 5'd1;
localparam STATE_SUM_WAIT     = 5'd2;
localparam STATE_SUM_CAP      = 5'd3;
localparam STATE_SUM_ACCUM    = 5'd4;
localparam STATE_MEAN_MUL     = 5'd5;
localparam STATE_EPS_ADD      = 5'd6;
localparam STATE_LUT_FETCH    = 5'd7;
localparam STATE_WGT_REQ      = 5'd8;
localparam STATE_WGT_WAIT     = 5'd9;
localparam STATE_WGT_CAP      = 5'd10;
localparam STATE_ACT_REQ      = 5'd11;
localparam STATE_ACT_WAIT     = 5'd12;
localparam STATE_ACT_CAP      = 5'd13;
localparam STATE_WRITE_REQ    = 5'd14;
localparam STATE_WRITE_COMMIT = 5'd15;
localparam STATE_DONE         = 5'd16;

localparam [31:0] EPS_BITS = 32'h3727c5ac;
localparam [11:0] RECIP_DIM_IDX = 12'd2128;  // 64.0 -> reciprocal LUT index

reg [4:0] state;
reg [1:0] op_code_reg;
reg [31:0] layer_idx_reg;
reg [31:0] idx_reg;
reg [31:0] weight_base_reg;
reg [31:0] output_base_reg;
reg [31:0] sum_bits;
reg [31:0] mean_bits;
reg [31:0] norm_bits;
reg [31:0] rsqrt_bits;
reg [31:0] act_bits;
reg [31:0] wgt_bits;

reg [31:0] rsqrt_lut [0:4095];
reg [31:0] reciprocal_lut [0:4095];
integer lut_fd;

reg [31:0] mul0_a;
reg [31:0] mul0_b;
reg [31:0] add0_a;
reg [31:0] add0_b;
wire [31:0] mul0_y;
wire [31:0] add0_y;
wire [31:0] mul1_y;
wire [31:0] weighted_y;

function [11:0] lut_index_from_bits;
    input [31:0] bits;
    begin
        lut_index_from_bits = {bits[30:23], bits[22:19]};
    end
endfunction

initial begin
    lut_fd = $fopen("src/LUTs/rsqrt_lut.hex", "r");
    if (lut_fd != 0) begin
        $fclose(lut_fd);
        $readmemh("src/LUTs/rsqrt_lut.hex", rsqrt_lut);
    end else begin
        lut_fd = $fopen("../src/LUTs/rsqrt_lut.hex", "r");
        if (lut_fd != 0) begin
            $fclose(lut_fd);
            $readmemh("../src/LUTs/rsqrt_lut.hex", rsqrt_lut);
        end else begin
            lut_fd = $fopen("../../src/LUTs/rsqrt_lut.hex", "r");
            if (lut_fd != 0) begin
                $fclose(lut_fd);
                $readmemh("../../src/LUTs/rsqrt_lut.hex", rsqrt_lut);
            end else begin
                $display("ERROR: kernel_rmsnorm could not locate rsqrt_lut.hex");
                $finish;
            end
        end
    end

    lut_fd = $fopen("src/LUTs/reciprocal_lut.hex", "r");
    if (lut_fd != 0) begin
        $fclose(lut_fd);
        $readmemh("src/LUTs/reciprocal_lut.hex", reciprocal_lut);
    end else begin
        lut_fd = $fopen("../src/LUTs/reciprocal_lut.hex", "r");
        if (lut_fd != 0) begin
            $fclose(lut_fd);
            $readmemh("../src/LUTs/reciprocal_lut.hex", reciprocal_lut);
        end else begin
            lut_fd = $fopen("../../src/LUTs/reciprocal_lut.hex", "r");
            if (lut_fd != 0) begin
                $fclose(lut_fd);
                $readmemh("../../src/LUTs/reciprocal_lut.hex", reciprocal_lut);
            end else begin
                $display("ERROR: kernel_rmsnorm could not locate reciprocal_lut.hex");
                $finish;
            end
        end
    end
end

always @(*) begin
    mul0_a = 32'd0;
    mul0_b = 32'd0;
    add0_a = 32'd0;
    add0_b = 32'd0;

    case (state)
        STATE_SUM_ACCUM: begin
            mul0_a = act_bits;
            mul0_b = act_bits;
            add0_a = sum_bits;
            add0_b = mul0_y;
        end
        STATE_MEAN_MUL: begin
            mul0_a = sum_bits;
            mul0_b = reciprocal_lut[RECIP_DIM_IDX];
        end
        STATE_EPS_ADD: begin
            add0_a = mean_bits;
            add0_b = EPS_BITS;
        end
        default: begin
        end
    endcase
end

kernel_mul u_mul_square (
    .a(mul0_a),
    .b(mul0_b),
    .y(mul0_y)
);

kernel_add u_add_accum (
    .a(add0_a),
    .b(add0_b),
    .y(add0_y)
);

kernel_mul u_mul_norm_act (
    .a(rsqrt_bits),
    .b(act_bits),
    .y(mul1_y)
);

kernel_mul u_mul_weighted (
    .a(wgt_bits),
    .b(mul1_y),
    .y(weighted_y)
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        busy <= 1'b0;
        done <= 1'b0;
        act_rd_en <= 1'b0;
        act_rd_addr <= 32'd0;
        act_wr_en <= 1'b0;
        act_wr_addr <= 32'd0;
        wgt_rd_en <= 1'b0;
        wgt_rd_addr <= 32'd0;
        state <= STATE_IDLE;
        op_code_reg <= 2'd0;
        layer_idx_reg <= 32'd0;
        idx_reg <= 32'd0;
        weight_base_reg <= 32'd0;
        output_base_reg <= 32'd0;
        sum_bits <= 32'd0;
        mean_bits <= 32'd0;
        norm_bits <= 32'd0;
        rsqrt_bits <= 32'd0;
        act_bits <= 32'd0;
        wgt_bits <= 32'd0;
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
                    op_code_reg <= ((CONTROL_MODE == 2) && (op_code == 2'd0)) ? OP_FINAL : op_code;
                    layer_idx_reg <= layer_idx;
                    idx_reg <= 32'd0;
                    sum_bits <= 32'd0;
                    if (((CONTROL_MODE == 2) && (op_code == 2'd0)) || (op_code == OP_FINAL)) begin
                        weight_base_reg <= 32'd0;
                        output_base_reg <= 32'd0;
                    end else begin
                        weight_base_reg <= layer_idx * DIM;
                        output_base_reg <= DIM;
                    end
                    state <= STATE_SUM_REQ;
                end
            end

            STATE_SUM_REQ: begin
                busy <= 1'b1;
                act_rd_en <= 1'b1;
                act_rd_addr <= idx_reg;
                state <= STATE_SUM_WAIT;
            end

            STATE_SUM_WAIT: begin
                busy <= 1'b1;
                state <= STATE_SUM_CAP;
            end

            STATE_SUM_CAP: begin
                busy <= 1'b1;
                act_bits <= act_rd_data;
                state <= STATE_SUM_ACCUM;
            end

            STATE_SUM_ACCUM: begin
                busy <= 1'b1;
                sum_bits <= add0_y;
                if (idx_reg == (DIM - 1)) begin
                    state <= STATE_MEAN_MUL;
                end else begin
                    idx_reg <= idx_reg + 32'd1;
                    state <= STATE_SUM_REQ;
                end
            end

            STATE_MEAN_MUL: begin
                busy <= 1'b1;
                mean_bits <= mul0_y;
                state <= STATE_EPS_ADD;
            end

            STATE_EPS_ADD: begin
                busy <= 1'b1;
                norm_bits <= add0_y;
                state <= STATE_LUT_FETCH;
            end

            STATE_LUT_FETCH: begin
                busy <= 1'b1;
                rsqrt_bits <= rsqrt_lut[lut_index_from_bits(norm_bits)];
                idx_reg <= 32'd0;
                state <= STATE_WGT_REQ;
            end

            STATE_WGT_REQ: begin
                busy <= 1'b1;
                wgt_rd_en <= 1'b1;
                wgt_rd_addr <= weight_base_reg + idx_reg;
                state <= STATE_WGT_WAIT;
            end

            STATE_WGT_WAIT: begin
                busy <= 1'b1;
                state <= STATE_WGT_CAP;
            end

            STATE_WGT_CAP: begin
                busy <= 1'b1;
                wgt_bits <= wgt_rd_data;
                state <= STATE_ACT_REQ;
            end

            STATE_ACT_REQ: begin
                busy <= 1'b1;
                act_rd_en <= 1'b1;
                act_rd_addr <= idx_reg;
                state <= STATE_ACT_WAIT;
            end

            STATE_ACT_WAIT: begin
                busy <= 1'b1;
                state <= STATE_ACT_CAP;
            end

            STATE_ACT_CAP: begin
                busy <= 1'b1;
                act_bits <= act_rd_data;
                state <= STATE_WRITE_REQ;
            end

            STATE_WRITE_REQ: begin
                busy <= 1'b1;
                act_wr_en <= 1'b1;
                act_wr_addr <= output_base_reg + idx_reg;
                act_wr_data <= weighted_y;
                state <= STATE_WRITE_COMMIT;
            end

            STATE_WRITE_COMMIT: begin
                busy <= 1'b1;
                if (idx_reg == (DIM - 1)) begin
                    state <= STATE_DONE;
                end else begin
                    idx_reg <= idx_reg + 32'd1;
                    state <= STATE_WGT_REQ;
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
