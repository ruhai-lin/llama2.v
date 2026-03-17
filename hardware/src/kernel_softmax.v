module kernel_softmax #(
    parameter N_HEADS = 8,
    parameter MAX_SEQ_LEN = 512,
    parameter ACT_ATT_BASE = 0
) (
    output reg act_rd_en,
    output reg [31:0] act_rd_addr,
    input wire [63:0] act_rd_data,
    output reg act_wr_en,
    output reg [31:0] act_wr_addr,
    output reg [63:0] act_wr_data
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

endmodule
