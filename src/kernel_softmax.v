module kernel_softmax #(
    parameter N_HEADS = 8,
    parameter MAX_SEQ_LEN = 512,
    parameter ACT_ATT_BASE = 0
) (
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [31:0] head_idx,
    input wire [9:0] pos_idx,
    output reg busy,
    output reg done,
    output reg act_rd_en,
    output reg [31:0] act_rd_addr,
    input wire [31:0] act_rd_data,
    output reg act_wr_en,
    output reg [31:0] act_wr_addr,
    output reg [31:0] act_wr_data
);

localparam STATE_IDLE         = 5'd0;
localparam STATE_MAX_REQ      = 5'd1;
localparam STATE_MAX_WAIT     = 5'd2;
localparam STATE_MAX_CAP      = 5'd3;
localparam STATE_EXP_REQ      = 5'd4;
localparam STATE_EXP_WAIT     = 5'd5;
localparam STATE_EXP_CAP      = 5'd6;
localparam STATE_EXP_LUT      = 5'd7;
localparam STATE_EXP_WRITE    = 5'd8;
localparam STATE_NORM_REQ     = 5'd9;
localparam STATE_NORM_WAIT    = 5'd10;
localparam STATE_NORM_CAP     = 5'd11;
localparam STATE_NORM_MUL     = 5'd12;
localparam STATE_NORM_WRITE   = 5'd13;
localparam STATE_DONE         = 5'd14;

reg [4:0] state;
reg [31:0] head_idx_reg;
reg [9:0] pos_idx_reg;
reg [31:0] att_base_reg;
reg [31:0] timestep_reg;
reg [31:0] max_bits;
reg [31:0] sum_bits;
reg [31:0] cur_bits;
reg [31:0] exp_bits;
reg [31:0] recip_sum_bits;

reg [31:0] exp_lut [0:4095];
reg [31:0] reciprocal_lut [0:4095];
integer lut_fd;

wire [31:0] neg_max_bits;
wire [31:0] diff_bits;
wire [31:0] sum_next_bits;
wire [31:0] norm_bits;

assign neg_max_bits = {~max_bits[31], max_bits[30:0]};

function [11:0] lut_index_from_bits;
    input [31:0] bits;
    begin
        lut_index_from_bits = {bits[30:23], bits[22:19]};
    end
endfunction

function fp32_is_zero;
    input [31:0] bits;
    begin
        fp32_is_zero = (bits[30:0] == 31'd0);
    end
endfunction

function fp32_gt;
    input [31:0] a;
    input [31:0] b;
    reg [30:0] mag_a;
    reg [30:0] mag_b;
    begin
        mag_a = a[30:0];
        mag_b = b[30:0];
        if (fp32_is_zero(a) && fp32_is_zero(b)) begin
            fp32_gt = 1'b0;
        end else if (a[31] != b[31]) begin
            fp32_gt = b[31];
        end else if (a[31] == 1'b0) begin
            fp32_gt = (mag_a > mag_b);
        end else begin
            fp32_gt = (mag_a < mag_b);
        end
    end
endfunction

function [11:0] exp_lut_index_from_diff;
    input [31:0] bits;
    begin
        if (fp32_is_zero(bits)) begin
            exp_lut_index_from_diff = 12'd0;
        end else if (bits[31] == 1'b0) begin
            exp_lut_index_from_diff = 12'd0;
        end else begin
            exp_lut_index_from_diff = lut_index_from_bits(bits);
        end
    end
endfunction

initial begin
    lut_fd = $fopen("src/LUTs/exp_lut.hex", "r");
    if (lut_fd != 0) begin
        $fclose(lut_fd);
        $readmemh("src/LUTs/exp_lut.hex", exp_lut);
    end else begin
        lut_fd = $fopen("../src/LUTs/exp_lut.hex", "r");
        if (lut_fd != 0) begin
            $fclose(lut_fd);
            $readmemh("../src/LUTs/exp_lut.hex", exp_lut);
        end else begin
            lut_fd = $fopen("../../src/LUTs/exp_lut.hex", "r");
            if (lut_fd != 0) begin
                $fclose(lut_fd);
                $readmemh("../../src/LUTs/exp_lut.hex", exp_lut);
            end else begin
                $display("ERROR: kernel_softmax could not locate exp_lut.hex");
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
                $display("ERROR: kernel_softmax could not locate reciprocal_lut.hex");
                $finish;
            end
        end
    end
end

kernel_add u_sub_max (
    .a(cur_bits),
    .b(neg_max_bits),
    .y(diff_bits)
);

kernel_add u_add_sum (
    .a(sum_bits),
    .b(exp_bits),
    .y(sum_next_bits)
);

kernel_mul u_mul_norm (
    .a(cur_bits),
    .b(recip_sum_bits),
    .y(norm_bits)
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        act_rd_en <= 1'b0;
        act_rd_addr <= 32'd0;
        act_wr_en <= 1'b0;
        act_wr_addr <= 32'd0;
        act_wr_data <= 32'd0;
        busy <= 1'b0;
        done <= 1'b0;
        state <= STATE_IDLE;
        head_idx_reg <= 32'd0;
        pos_idx_reg <= 10'd0;
        att_base_reg <= 32'd0;
        timestep_reg <= 32'd0;
        max_bits <= 32'd0;
        sum_bits <= 32'd0;
        cur_bits <= 32'd0;
        exp_bits <= 32'd0;
        recip_sum_bits <= 32'd0;
    end else begin
        done <= 1'b0;
        act_rd_en <= 1'b0;
        act_wr_en <= 1'b0;

        case (state)
            STATE_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    busy <= 1'b1;
                    head_idx_reg <= head_idx;
                    pos_idx_reg <= pos_idx;
                    att_base_reg <= ACT_ATT_BASE + head_idx * MAX_SEQ_LEN;
                    timestep_reg <= 32'd0;
                    max_bits <= 32'd0;
                    sum_bits <= 32'd0;
                    cur_bits <= 32'd0;
                    exp_bits <= 32'd0;
                    recip_sum_bits <= 32'd0;
                    state <= STATE_MAX_REQ;
                end
            end

            STATE_MAX_REQ: begin
                busy <= 1'b1;
                act_rd_en <= 1'b1;
                act_rd_addr <= att_base_reg + timestep_reg;
                state <= STATE_MAX_WAIT;
            end

            STATE_MAX_WAIT: begin
                busy <= 1'b1;
                state <= STATE_MAX_CAP;
            end

            STATE_MAX_CAP: begin
                busy <= 1'b1;
                cur_bits <= act_rd_data;
                if ((timestep_reg == 32'd0) || fp32_gt(act_rd_data, max_bits)) begin
                    max_bits <= act_rd_data;
                end
                if (timestep_reg == {22'd0, pos_idx_reg}) begin
                    timestep_reg <= 32'd0;
                    sum_bits <= 32'd0;
                    state <= STATE_EXP_REQ;
                end else begin
                    timestep_reg <= timestep_reg + 32'd1;
                    state <= STATE_MAX_REQ;
                end
            end

            STATE_EXP_REQ: begin
                busy <= 1'b1;
                act_rd_en <= 1'b1;
                act_rd_addr <= att_base_reg + timestep_reg;
                state <= STATE_EXP_WAIT;
            end

            STATE_EXP_WAIT: begin
                busy <= 1'b1;
                state <= STATE_EXP_CAP;
            end

            STATE_EXP_CAP: begin
                busy <= 1'b1;
                cur_bits <= act_rd_data;
                state <= STATE_EXP_LUT;
            end

            STATE_EXP_LUT: begin
                busy <= 1'b1;
                exp_bits <= exp_lut[exp_lut_index_from_diff(diff_bits)];
                state <= STATE_EXP_WRITE;
            end

            STATE_EXP_WRITE: begin
                busy <= 1'b1;
                act_wr_en <= 1'b1;
                act_wr_addr <= att_base_reg + timestep_reg;
                act_wr_data <= exp_bits;
                sum_bits <= sum_next_bits;
                if (timestep_reg == {22'd0, pos_idx_reg}) begin
                    timestep_reg <= 32'd0;
                    state <= STATE_NORM_REQ;
                end else begin
                    timestep_reg <= timestep_reg + 32'd1;
                    state <= STATE_EXP_REQ;
                end
            end

            STATE_NORM_REQ: begin
                busy <= 1'b1;
                act_rd_en <= 1'b1;
                act_rd_addr <= att_base_reg + timestep_reg;
                state <= STATE_NORM_WAIT;
            end

            STATE_NORM_WAIT: begin
                busy <= 1'b1;
                state <= STATE_NORM_CAP;
            end

            STATE_NORM_CAP: begin
                busy <= 1'b1;
                cur_bits <= act_rd_data;
                recip_sum_bits <= reciprocal_lut[lut_index_from_bits(sum_bits)];
                state <= STATE_NORM_MUL;
            end

            STATE_NORM_MUL: begin
                busy <= 1'b1;
                act_wr_data <= norm_bits;
                state <= STATE_NORM_WRITE;
            end

            STATE_NORM_WRITE: begin
                busy <= 1'b1;
                act_wr_en <= 1'b1;
                act_wr_addr <= att_base_reg + timestep_reg;
                if (timestep_reg == {22'd0, pos_idx_reg}) begin
                    state <= STATE_DONE;
                end else begin
                    timestep_reg <= timestep_reg + 32'd1;
                    state <= STATE_NORM_REQ;
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
