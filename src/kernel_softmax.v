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

`include "real_fp32_helpers.vh"

localparam STATE_IDLE        = 4'd0;
localparam STATE_MAX_REQ     = 4'd1;
localparam STATE_MAX_WAIT    = 4'd2;
localparam STATE_MAX_CAP     = 4'd3;
localparam STATE_EXP_REQ     = 4'd4;
localparam STATE_EXP_WAIT    = 4'd5;
localparam STATE_EXP_CAP     = 4'd6;
localparam STATE_EXP_WRITE   = 4'd7;
localparam STATE_NORM_REQ    = 4'd8;
localparam STATE_NORM_WAIT   = 4'd9;
localparam STATE_NORM_CAP    = 4'd10;
localparam STATE_NORM_WRITE  = 4'd11;
localparam STATE_DONE        = 4'd12;

reg [3:0] state;
reg [31:0] head_idx_reg;
reg [9:0] pos_idx_reg;
reg [31:0] att_base_reg;
integer timestep;
real max_score;
real sum_exp;
real act_value;

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
        timestep <= 0;
        max_score <= 0.0;
        sum_exp <= 0.0;
        act_value <= 0.0;
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
                    timestep <= 0;
                    max_score <= 0.0;
                    sum_exp <= 0.0;
                    state <= STATE_MAX_REQ;
                end
            end

            STATE_MAX_REQ: begin
                busy <= 1'b1;
                act_rd_en <= 1'b1;
                act_rd_addr <= att_base_reg + timestep;
                state <= STATE_MAX_WAIT;
            end

            STATE_MAX_WAIT: begin
                busy <= 1'b1;
                state <= STATE_MAX_CAP;
            end

            STATE_MAX_CAP: begin
                busy <= 1'b1;
                act_value <= fp32_to_real(act_rd_data);
                if ((timestep == 0) || (fp32_to_real(act_rd_data) > max_score)) begin
                    max_score <= fp32_to_real(act_rd_data);
                end
                if (timestep == pos_idx_reg) begin
                    timestep <= 0;
                    sum_exp <= 0.0;
                    state <= STATE_EXP_REQ;
                end else begin
                    timestep <= timestep + 1;
                    state <= STATE_MAX_REQ;
                end
            end

            STATE_EXP_REQ: begin
                busy <= 1'b1;
                act_rd_en <= 1'b1;
                act_rd_addr <= att_base_reg + timestep;
                state <= STATE_EXP_WAIT;
            end

            STATE_EXP_WAIT: begin
                busy <= 1'b1;
                state <= STATE_EXP_CAP;
            end

            STATE_EXP_CAP: begin
                busy <= 1'b1;
                act_value <= fp32_round($exp(fp32_to_real(act_rd_data) - max_score));
                state <= STATE_EXP_WRITE;
            end

            STATE_EXP_WRITE: begin
                busy <= 1'b1;
                act_wr_en <= 1'b1;
                act_wr_addr <= att_base_reg + timestep;
                act_wr_data <= real_to_fp32_bits(act_value);
                sum_exp <= sum_exp + act_value;
                if (timestep == pos_idx_reg) begin
                    timestep <= 0;
                    state <= STATE_NORM_REQ;
                end else begin
                    timestep <= timestep + 1;
                    state <= STATE_EXP_REQ;
                end
            end

            STATE_NORM_REQ: begin
                busy <= 1'b1;
                act_rd_en <= 1'b1;
                act_rd_addr <= att_base_reg + timestep;
                state <= STATE_NORM_WAIT;
            end

            STATE_NORM_WAIT: begin
                busy <= 1'b1;
                state <= STATE_NORM_CAP;
            end

            STATE_NORM_CAP: begin
                busy <= 1'b1;
                act_value <= fp32_to_real(act_rd_data);
                state <= STATE_NORM_WRITE;
            end

            STATE_NORM_WRITE: begin
                busy <= 1'b1;
                act_wr_en <= 1'b1;
                act_wr_addr <= att_base_reg + timestep;
                act_wr_data <= real_to_fp32_bits(fp32_round(act_value / sum_exp));
                if (timestep == pos_idx_reg) begin
                    state <= STATE_DONE;
                end else begin
                    timestep <= timestep + 1;
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
