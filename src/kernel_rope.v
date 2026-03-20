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

integer elem_idx;
integer rot_count;
integer vec_sel;
integer head_dim;
real freq;
real angle;
real cos_val;
real sin_val;
real v0;
real v1;

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

task apply;
    input integer pos_idx;
    begin
        act_rd_en = 1'b0;
        act_wr_en = 1'b0;
        for (elem_idx = 0; elem_idx < DIM; elem_idx = elem_idx + 2) begin
            head_dim = elem_idx % HEAD_SIZE;
            freq = rope_freq(head_dim);
            angle = pos_idx * freq;
            cos_val = $cos(angle);
            sin_val = $sin(angle);
            if (elem_idx < KV_DIM) begin
                rot_count = 2;
            end else begin
                rot_count = 1;
            end
            for (vec_sel = 0; vec_sel < rot_count; vec_sel = vec_sel + 1) begin
                if (vec_sel == 0) begin
                    read_act(ACT_Q_BASE + elem_idx, v0);
                    read_act(ACT_Q_BASE + elem_idx + 1, v1);
                    write_act(ACT_Q_BASE + elem_idx, fp32_round(v0 * cos_val - v1 * sin_val));
                    write_act(ACT_Q_BASE + elem_idx + 1, fp32_round(v0 * sin_val + v1 * cos_val));
                end else begin
                    read_act(ACT_K_BASE + elem_idx, v0);
                    read_act(ACT_K_BASE + elem_idx + 1, v1);
                    write_act(ACT_K_BASE + elem_idx, fp32_round(v0 * cos_val - v1 * sin_val));
                    write_act(ACT_K_BASE + elem_idx + 1, fp32_round(v0 * sin_val + v1 * cos_val));
                end
            end
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
            apply(pos_idx);
            busy <= 1'b0;
            done <= 1'b1;
        end else begin
            busy <= 1'b0;
        end
    end
end

endmodule
