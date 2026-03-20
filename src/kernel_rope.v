module kernel_rope #(
    parameter DIM = 64,
    parameter N_HEADS = 8,
    parameter N_KV_HEADS = 4,
    parameter ACT_Q_BASE = 0,
    parameter ACT_K_BASE = 0
) (
    input wire clk,
    input wire rst_n,
    input wire start,
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

localparam HEAD_SIZE = DIM / N_HEADS;
localparam KV_DIM = (DIM * N_KV_HEADS) / N_HEADS;

localparam STATE_IDLE      = 4'd0;
localparam STATE_PREP      = 4'd1;
localparam STATE_READ0_REQ = 4'd2;
localparam STATE_READ0_WAIT= 4'd3;
localparam STATE_READ0_CAP = 4'd4;
localparam STATE_READ1_REQ = 4'd5;
localparam STATE_READ1_WAIT= 4'd6;
localparam STATE_READ1_CAP = 4'd7;
localparam STATE_WRITE0    = 4'd8;
localparam STATE_WRITE1    = 4'd9;
localparam STATE_NEXT_VEC  = 4'd10;
localparam STATE_NEXT_ELEM = 4'd11;
localparam STATE_DONE      = 4'd12;

reg [3:0] state;
reg [9:0] pos_idx_reg;
integer elem_idx;
integer vec_sel;
integer rot_count;
integer head_dim;
real freq;
real angle;
real cos_val;
real sin_val;
real v0;
real v1;

function real rope_freq;
    input integer current_head_dim;
    begin
        case (current_head_dim)
            0: rope_freq = 1.0;
            2: rope_freq = 0.1;
            4: rope_freq = 0.01;
            6: rope_freq = 0.001;
            default: rope_freq = 1.0;
        endcase
    end
endfunction

function [31:0] vec_base_addr;
    input integer vec_idx;
    begin
        if (vec_idx == 0) begin
            vec_base_addr = ACT_Q_BASE;
        end else begin
            vec_base_addr = ACT_K_BASE;
        end
    end
endfunction

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
        pos_idx_reg <= 10'd0;
        elem_idx <= 0;
        vec_sel <= 0;
        rot_count <= 0;
        head_dim <= 0;
        freq <= 0.0;
        angle <= 0.0;
        cos_val <= 0.0;
        sin_val <= 0.0;
        v0 <= 0.0;
        v1 <= 0.0;
    end else begin
        done <= 1'b0;
        act_rd_en <= 1'b0;
        act_wr_en <= 1'b0;

        case (state)
            STATE_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    busy <= 1'b1;
                    pos_idx_reg <= pos_idx;
                    elem_idx <= 0;
                    vec_sel <= 0;
                    state <= STATE_PREP;
                end
            end

            STATE_PREP: begin
                busy <= 1'b1;
                head_dim <= elem_idx % HEAD_SIZE;
                freq <= rope_freq(elem_idx % HEAD_SIZE);
                angle <= pos_idx_reg * rope_freq(elem_idx % HEAD_SIZE);
                cos_val <= $cos(pos_idx_reg * rope_freq(elem_idx % HEAD_SIZE));
                sin_val <= $sin(pos_idx_reg * rope_freq(elem_idx % HEAD_SIZE));
                if (elem_idx < KV_DIM) begin
                    rot_count <= 2;
                end else begin
                    rot_count <= 1;
                end
                state <= STATE_READ0_REQ;
            end

            STATE_READ0_REQ: begin
                busy <= 1'b1;
                act_rd_en <= 1'b1;
                act_rd_addr <= vec_base_addr(vec_sel) + elem_idx;
                state <= STATE_READ0_WAIT;
            end

            STATE_READ0_WAIT: begin
                busy <= 1'b1;
                state <= STATE_READ0_CAP;
            end

            STATE_READ0_CAP: begin
                busy <= 1'b1;
                v0 <= fp32_to_real(act_rd_data);
                state <= STATE_READ1_REQ;
            end

            STATE_READ1_REQ: begin
                busy <= 1'b1;
                act_rd_en <= 1'b1;
                act_rd_addr <= vec_base_addr(vec_sel) + elem_idx + 1;
                state <= STATE_READ1_WAIT;
            end

            STATE_READ1_WAIT: begin
                busy <= 1'b1;
                state <= STATE_READ1_CAP;
            end

            STATE_READ1_CAP: begin
                busy <= 1'b1;
                v1 <= fp32_to_real(act_rd_data);
                state <= STATE_WRITE0;
            end

            STATE_WRITE0: begin
                busy <= 1'b1;
                act_wr_en <= 1'b1;
                act_wr_addr <= vec_base_addr(vec_sel) + elem_idx;
                act_wr_data <= real_to_fp32_bits(fp32_round(v0 * cos_val - v1 * sin_val));
                state <= STATE_WRITE1;
            end

            STATE_WRITE1: begin
                busy <= 1'b1;
                act_wr_en <= 1'b1;
                act_wr_addr <= vec_base_addr(vec_sel) + elem_idx + 1;
                act_wr_data <= real_to_fp32_bits(fp32_round(v0 * sin_val + v1 * cos_val));
                state <= STATE_NEXT_VEC;
            end

            STATE_NEXT_VEC: begin
                busy <= 1'b1;
                if ((vec_sel + 1) < rot_count) begin
                    vec_sel <= vec_sel + 1;
                    state <= STATE_READ0_REQ;
                end else begin
                    vec_sel <= 0;
                    state <= STATE_NEXT_ELEM;
                end
            end

            STATE_NEXT_ELEM: begin
                busy <= 1'b1;
                if ((elem_idx + 2) >= DIM) begin
                    state <= STATE_DONE;
                end else begin
                    elem_idx <= elem_idx + 2;
                    state <= STATE_PREP;
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
