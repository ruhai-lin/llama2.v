module ffn #(
    parameter DIM = 64,
    parameter HIDDEN_DIM = 172,
    parameter N_HEADS = 8,
    parameter N_KV_HEADS = 4
) (
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [2:0] layer_idx,
    output reg busy,
    output reg done
);

`include "real_fp32_helpers.vh"

localparam VOCAB_SIZE = 512;
localparam N_LAYERS = 5;
localparam KV_DIM = (DIM * N_KV_HEADS) / N_HEADS;

`include "memory_map.vh"

wire rms_act_rd_en;
wire [31:0] rms_act_rd_addr;
wire [63:0] rms_act_rd_data;
wire rms_act_wr_en;
wire [31:0] rms_act_wr_addr;
wire [63:0] rms_act_wr_data;
wire rms_wgt_rd_en;
wire [31:0] rms_wgt_rd_addr;
wire [63:0] rms_wgt_rd_data;

wire matmul_act_rd_en;
wire [31:0] matmul_act_rd_addr;
wire [63:0] matmul_act_rd_data;
wire matmul_act_wr_en;
wire [31:0] matmul_act_wr_addr;
wire [63:0] matmul_act_wr_data;
wire matmul_wgt_rd_en;
wire [31:0] matmul_wgt_rd_addr;
wire [63:0] matmul_wgt_rd_data;

function [63:0] act_read_bits;
    input [31:0] addr;
    begin
        if (addr < DIM) begin
            act_read_bits = $realtobits(top_level_module.u_mem_activation.x[addr]);
        end else if (addr < `ACT_XB_BASE + DIM) begin
            act_read_bits = $realtobits(top_level_module.u_mem_activation.xb[addr - `ACT_XB_BASE]);
        end else if (addr < `ACT_XB2_BASE + DIM) begin
            act_read_bits = $realtobits(top_level_module.u_mem_activation.xb2[addr - `ACT_XB2_BASE]);
        end else if (addr < `ACT_HB_BASE + HIDDEN_DIM) begin
            act_read_bits = $realtobits(top_level_module.u_mem_activation.hb[addr - `ACT_HB_BASE]);
        end else begin
            act_read_bits = $realtobits(top_level_module.u_mem_activation.hb2[addr - `ACT_HB2_BASE]);
        end
    end
endfunction

function [63:0] wgt_read_bits;
    input [31:0] addr;
    begin
        if (addr < `WGT_W1_BASE) begin
            wgt_read_bits = $realtobits(top_level_module.u_mem_weights.rms_ffn_weight[addr - `WGT_RMS_FFN_BASE]);
        end else if (addr < `WGT_W2_BASE) begin
            wgt_read_bits = $realtobits(top_level_module.u_mem_weights.w1[addr - `WGT_W1_BASE]);
        end else if (addr < `WGT_W3_BASE) begin
            wgt_read_bits = $realtobits(top_level_module.u_mem_weights.w2[addr - `WGT_W2_BASE]);
        end else begin
            wgt_read_bits = $realtobits(top_level_module.u_mem_weights.w3[addr - `WGT_W3_BASE]);
        end
    end
endfunction

assign rms_act_rd_data = act_read_bits(rms_act_rd_addr);
assign matmul_act_rd_data = act_read_bits(matmul_act_rd_addr);
assign rms_wgt_rd_data = $realtobits(top_level_module.u_mem_weights.rms_ffn_weight[rms_wgt_rd_addr]);
assign matmul_wgt_rd_data = wgt_read_bits(matmul_wgt_rd_addr);

always @(*) begin
    if (rms_act_wr_en) begin
        top_level_module.u_mem_activation.xb[rms_act_wr_addr - `ACT_XB_BASE] = $bitstoreal(rms_act_wr_data);
    end
    if (matmul_act_wr_en) begin
        if (matmul_act_wr_addr >= `ACT_HB_BASE && matmul_act_wr_addr < `ACT_HB_BASE + HIDDEN_DIM) begin
            top_level_module.u_mem_activation.hb[matmul_act_wr_addr - `ACT_HB_BASE] = $bitstoreal(matmul_act_wr_data);
        end else if (matmul_act_wr_addr >= `ACT_HB2_BASE && matmul_act_wr_addr < `ACT_HB2_BASE + HIDDEN_DIM) begin
            top_level_module.u_mem_activation.hb2[matmul_act_wr_addr - `ACT_HB2_BASE] = $bitstoreal(matmul_act_wr_data);
        end else if (matmul_act_wr_addr >= `ACT_XB_BASE && matmul_act_wr_addr < `ACT_XB_BASE + DIM) begin
            top_level_module.u_mem_activation.xb[matmul_act_wr_addr - `ACT_XB_BASE] = $bitstoreal(matmul_act_wr_data);
        end else if (matmul_act_wr_addr < DIM) begin
            top_level_module.u_mem_activation.x[matmul_act_wr_addr] = $bitstoreal(matmul_act_wr_data);
        end
    end
end

kernel_rmsnorm #(
    .DIM(DIM)
) u_rmsnorm (
    .clk(clk),
    .rst_n(rst_n),
    .start(1'b0),
    .layer_idx(32'd0),
    .busy(),
    .done(),
    .act_rd_en(rms_act_rd_en),
    .act_rd_addr(rms_act_rd_addr),
    .act_rd_data(rms_act_rd_data),
    .act_wr_en(rms_act_wr_en),
    .act_wr_addr(rms_act_wr_addr),
    .act_wr_data(rms_act_wr_data),
    .wgt_rd_en(rms_wgt_rd_en),
    .wgt_rd_addr(rms_wgt_rd_addr),
    .wgt_rd_data(rms_wgt_rd_data)
);

kernel_matmul #(
    .DIM(DIM),
    .HIDDEN_DIM(HIDDEN_DIM),
    .N_HEADS(N_HEADS),
    .N_KV_HEADS(N_KV_HEADS)
) u_matmul (
    .clk(clk),
    .rst_n(rst_n),
    .start(1'b0),
    .layer_idx(32'd0),
    .busy(),
    .done(),
    .next_token(),
    .flat_logits(),
    .act_rd_en(matmul_act_rd_en),
    .act_rd_addr(matmul_act_rd_addr),
    .act_rd_data(matmul_act_rd_data),
    .act_wr_en(matmul_act_wr_en),
    .act_wr_addr(matmul_act_wr_addr),
    .act_wr_data(matmul_act_wr_data),
    .wgt_rd_en(matmul_wgt_rd_en),
    .wgt_rd_addr(matmul_wgt_rd_addr),
    .wgt_rd_data(matmul_wgt_rd_data)
);

integer i;
real val;
real hb_val;
real hb2_val;

task read_act_local;
    input integer addr;
    output real value;
    begin
        value = $bitstoreal(act_read_bits(addr));
    end
endtask

task write_act_local;
    input integer addr;
    input real value;
    begin
        if (addr >= `ACT_X_BASE && addr < `ACT_X_BASE + `ACT_X_SIZE) begin
            top_level_module.u_mem_activation.x[addr - `ACT_X_BASE] = value;
        end else if (addr >= `ACT_XB_BASE && addr < `ACT_XB_BASE + `ACT_XB_SIZE) begin
            top_level_module.u_mem_activation.xb[addr - `ACT_XB_BASE] = value;
        end else if (addr >= `ACT_HB_BASE && addr < `ACT_HB_BASE + `ACT_HB_SIZE) begin
            top_level_module.u_mem_activation.hb[addr - `ACT_HB_BASE] = value;
        end else if (addr >= `ACT_HB2_BASE && addr < `ACT_HB2_BASE + `ACT_HB2_SIZE) begin
            top_level_module.u_mem_activation.hb2[addr - `ACT_HB2_BASE] = value;
        end
    end
endtask

task run;
    input integer task_layer_idx;
    begin
        u_rmsnorm.apply_ffn(task_layer_idx);
        u_matmul.project_w1_w3(task_layer_idx);

        for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
            read_act_local(`ACT_HB_BASE + i, hb_val);
            read_act_local(`ACT_HB2_BASE + i, hb2_val);
            val = hb_val;
            val = val * (1.0 / (1.0 + $exp(-val)));
            write_act_local(`ACT_HB_BASE + i, fp32_round(val * hb2_val));
        end

        u_matmul.project_w2(task_layer_idx);
    end
endtask

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        busy <= 1'b0;
        done <= 1'b0;
    end else begin
        done <= 1'b0;
        if (start) begin
            busy <= 1'b1;
            run(layer_idx);
            busy <= 1'b0;
            done <= 1'b1;
        end else begin
            busy <= 1'b0;
        end
    end
end

endmodule
