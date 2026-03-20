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

`include "real_fp32_helpers.vh"

localparam OP_ATTN  = 2'd1;
localparam OP_FFN   = 2'd2;
localparam OP_FINAL = 2'd3;

localparam STATE_IDLE        = 4'd0;
localparam STATE_SUM_REQ     = 4'd1;
localparam STATE_SUM_WAIT    = 4'd2;
localparam STATE_SUM_CAP     = 4'd3;
localparam STATE_PREP_WRITE  = 4'd4;
localparam STATE_WGT_REQ     = 4'd5;
localparam STATE_WGT_WAIT    = 4'd6;
localparam STATE_WGT_CAP     = 4'd7;
localparam STATE_ACT_REQ     = 4'd8;
localparam STATE_ACT_WAIT    = 4'd9;
localparam STATE_ACT_CAP     = 4'd10;
localparam STATE_WRITE_REQ   = 4'd11;
localparam STATE_WRITE_COMMIT= 4'd12;
localparam STATE_DONE        = 4'd13;

reg [3:0] state;
reg [1:0] op_code_reg;
reg [31:0] layer_idx_reg;
reg [31:0] idx_reg;
reg [31:0] weight_base_reg;
reg [31:0] output_base_reg;
real sum_sq;
real inv_norm;
real act_value;
real wgt_value;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        busy <= 1'b0;
        done <= 1'b0;
        act_rd_en <= 1'b0;
        act_rd_addr <= 32'd0;
        act_wr_en <= 1'b0;
        act_wr_addr <= 32'd0;
        act_wr_data <= 32'd0;
        wgt_rd_en <= 1'b0;
        wgt_rd_addr <= 32'd0;
        state <= STATE_IDLE;
        op_code_reg <= 2'd0;
        layer_idx_reg <= 32'd0;
        idx_reg <= 32'd0;
        weight_base_reg <= 32'd0;
        output_base_reg <= 32'd0;
        sum_sq <= 0.0;
        inv_norm <= 0.0;
        act_value <= 0.0;
        wgt_value <= 0.0;
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
                    sum_sq <= 0.0;
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
                act_value <= fp32_to_real(act_rd_data);
                sum_sq <= sum_sq + (fp32_to_real(act_rd_data) * fp32_to_real(act_rd_data));
                if (idx_reg == (DIM - 1)) begin
                    state <= STATE_PREP_WRITE;
                end else begin
                    idx_reg <= idx_reg + 32'd1;
                    state <= STATE_SUM_REQ;
                end
            end

            STATE_PREP_WRITE: begin
                busy <= 1'b1;
                sum_sq <= (sum_sq / DIM) + 1e-5;
                inv_norm <= 1.0 / $sqrt((sum_sq / DIM) + 1e-5);
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
                wgt_value <= fp32_to_real(wgt_rd_data);
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
                act_value <= fp32_to_real(act_rd_data);
                state <= STATE_WRITE_REQ;
            end

            STATE_WRITE_REQ: begin
                busy <= 1'b1;
                act_wr_en <= 1'b1;
                act_wr_addr <= output_base_reg + idx_reg;
                act_wr_data <= real_to_fp32_bits(fp32_round(wgt_value * (inv_norm * act_value)));
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
