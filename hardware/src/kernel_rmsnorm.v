module kernel_rmsnorm #(
    parameter DIM = 64,
    parameter CONTROL_MODE = 0
) (
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [31:0] layer_idx,
    output reg busy,
    output reg done,
    output reg act_rd_en,
    output reg [31:0] act_rd_addr,
    input wire [63:0] act_rd_data,
    output reg act_wr_en,
    output reg [31:0] act_wr_addr,
    output reg [63:0] act_wr_data,
    output reg wgt_rd_en,
    output reg [31:0] wgt_rd_addr,
    input wire [63:0] wgt_rd_data
);

`include "real_fp32_helpers.vh"

integer i;
real sum_sq;
real inv_norm;
real act_value;
real wgt_value;

task read_act;
    input integer addr;
    output real value;
    begin
        act_rd_en = 1'b1;
        act_rd_addr = addr;
        #0;
        value = $bitstoreal(act_rd_data);
        act_rd_en = 1'b0;
    end
endtask

task write_act;
    input integer addr;
    input real value;
    begin
        act_wr_en = 1'b1;
        act_wr_addr = addr;
        act_wr_data = $realtobits(value);
        #0;
        act_wr_en = 1'b0;
    end
endtask

task read_wgt;
    input integer addr;
    output real value;
    begin
        wgt_rd_en = 1'b1;
        wgt_rd_addr = addr;
        #0;
        value = $bitstoreal(wgt_rd_data);
        wgt_rd_en = 1'b0;
    end
endtask

task apply_attn;
    input integer task_layer_idx;
    integer base_idx;
    begin
        sum_sq = 0.0;
        for (i = 0; i < DIM; i = i + 1) begin
            read_act(i, act_value);
            sum_sq = sum_sq + (act_value * act_value);
        end
        sum_sq = (sum_sq / DIM) + 1e-5;
        inv_norm = 1.0 / $sqrt(sum_sq);
        base_idx = task_layer_idx * DIM;
        for (i = 0; i < DIM; i = i + 1) begin
            read_wgt(base_idx + i, wgt_value);
            read_act(i, act_value);
            write_act(
                DIM + i,
                fp32_round(wgt_value * (inv_norm * act_value))
            );
        end
    end
endtask

task apply_ffn;
    input integer task_layer_idx;
    integer base_idx;
    begin
        sum_sq = 0.0;
        for (i = 0; i < DIM; i = i + 1) begin
            read_act(i, act_value);
            sum_sq = sum_sq + (act_value * act_value);
        end
        sum_sq = (sum_sq / DIM) + 1e-5;
        inv_norm = 1.0 / $sqrt(sum_sq);
        base_idx = task_layer_idx * DIM;
        for (i = 0; i < DIM; i = i + 1) begin
            read_wgt(base_idx + i, wgt_value);
            read_act(i, act_value);
            write_act(
                DIM + i,
                fp32_round(wgt_value * (inv_norm * act_value))
            );
        end
    end
endtask

task apply_final;
    begin
        sum_sq = 0.0;
        for (i = 0; i < DIM; i = i + 1) begin
            read_act(i, act_value);
            sum_sq = sum_sq + (act_value * act_value);
        end
        sum_sq = (sum_sq / DIM) + 1e-5;
        inv_norm = 1.0 / $sqrt(sum_sq);
        for (i = 0; i < DIM; i = i + 1) begin
            read_wgt(i, wgt_value);
            read_act(i, act_value);
            write_act(
                i,
                fp32_round(wgt_value * (inv_norm * act_value))
            );
        end
    end
endtask

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        busy <= 1'b0;
        done <= 1'b0;
        act_rd_en <= 1'b0;
        act_rd_addr <= 32'd0;
        act_wr_en <= 1'b0;
        act_wr_addr <= 32'd0;
        act_wr_data <= 64'd0;
        wgt_rd_en <= 1'b0;
        wgt_rd_addr <= 32'd0;
    end else begin
        done <= 1'b0;
        act_rd_en <= 1'b0;
        act_wr_en <= 1'b0;
        wgt_rd_en <= 1'b0;
        if (start) begin
            busy <= 1'b1;
            if (CONTROL_MODE == 2) begin
                apply_final();
            end
            busy <= 1'b0;
            done <= 1'b1;
        end else begin
            busy <= 1'b0;
        end
    end
end

endmodule
