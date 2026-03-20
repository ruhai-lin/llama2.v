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

integer timestep;
real max_score;
real sum_exp;
real act_value;

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

task normalize_head;
    input integer head_idx;
    input integer pos_idx;
    integer att_base;
    begin
        att_base = ACT_ATT_BASE + head_idx * MAX_SEQ_LEN;
        read_act(att_base, max_score);
        for (timestep = 1; timestep <= pos_idx; timestep = timestep + 1) begin
            read_act(att_base + timestep, act_value);
            if (act_value > max_score) begin
                max_score = act_value;
            end
        end

        sum_exp = 0.0;
        for (timestep = 0; timestep <= pos_idx; timestep = timestep + 1) begin
            read_act(att_base + timestep, act_value);
            act_value = fp32_round($exp(act_value - max_score));
            write_act(att_base + timestep, act_value);
            sum_exp = sum_exp + act_value;
        end

        for (timestep = 0; timestep <= pos_idx; timestep = timestep + 1) begin
            read_act(att_base + timestep, act_value);
            write_act(att_base + timestep, fp32_round(act_value / sum_exp));
        end
    end
endtask

initial begin
    act_rd_en = 1'b0;
    act_rd_addr = 32'd0;
    act_wr_en = 1'b0;
    act_wr_addr = 32'd0;
    act_wr_data = 32'd0;
    busy = 1'b0;
    done = 1'b0;
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        act_rd_en <= 1'b0;
        act_rd_addr <= 32'd0;
        act_wr_en <= 1'b0;
        act_wr_addr <= 32'd0;
        act_wr_data <= 32'd0;
        busy <= 1'b0;
        done <= 1'b0;
    end else begin
        done <= 1'b0;
        if (start) begin
            busy <= 1'b1;
            normalize_head(head_idx, pos_idx);
            busy <= 1'b0;
            done <= 1'b1;
        end else begin
            busy <= 1'b0;
        end
    end
end

endmodule
