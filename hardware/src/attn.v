module attn #(
    parameter DIM = 64,
    parameter N_HEADS = 8,
    parameter N_KV_HEADS = 4,
    parameter MAX_SEQ_LEN = 512
) (
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [2:0] layer_idx,
    input wire [9:0] pos_idx,
    output reg kv_rd_en,
    output reg [31:0] kv_rd_addr,
    input wire [63:0] kv_rd_data,
    output reg kv_wr_en,
    output reg [31:0] kv_wr_addr,
    output reg [63:0] kv_wr_data,
    output reg busy,
    output reg done
);

`include "real_fp32_helpers.vh"

localparam VOCAB_SIZE = 512;
localparam HIDDEN_DIM = 172;
localparam N_LAYERS = 5;
localparam HEAD_SIZE = DIM / N_HEADS;
localparam KV_DIM = (DIM * N_KV_HEADS) / N_HEADS;
localparam KV_MUL = N_HEADS / N_KV_HEADS;

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

wire rope_act_rd_en;
wire [31:0] rope_act_rd_addr;
wire [63:0] rope_act_rd_data;
wire rope_act_wr_en;
wire [31:0] rope_act_wr_addr;
wire [63:0] rope_act_wr_data;

wire softmax_act_rd_en;
wire [31:0] softmax_act_rd_addr;
wire [63:0] softmax_act_rd_data;
wire softmax_act_wr_en;
wire [31:0] softmax_act_wr_addr;
wire [63:0] softmax_act_wr_data;

real kv_value;

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
        end else if (addr < `ACT_HB2_BASE + HIDDEN_DIM) begin
            act_read_bits = $realtobits(top_level_module.u_mem_activation.hb2[addr - `ACT_HB2_BASE]);
        end else if (addr < `ACT_K_BASE) begin
            act_read_bits = $realtobits(top_level_module.u_mem_activation.q[addr - `ACT_Q_BASE]);
        end else if (addr < `ACT_V_BASE) begin
            act_read_bits = $realtobits(top_level_module.u_mem_activation.k_vec[addr - `ACT_K_BASE]);
        end else if (addr < `ACT_ATT_BASE) begin
            act_read_bits = $realtobits(top_level_module.u_mem_activation.v_vec[addr - `ACT_V_BASE]);
        end else begin
            act_read_bits = $realtobits(top_level_module.u_mem_activation.att[addr - `ACT_ATT_BASE]);
        end
    end
endfunction

function [63:0] wgt_read_bits;
    input [31:0] addr;
    begin
        if (addr < `WGT_WQ_BASE) begin
            wgt_read_bits = $realtobits(top_level_module.u_mem_weights.rms_att_weight[addr - `WGT_RMS_ATT_BASE]);
        end else if (addr < `WGT_WK_BASE) begin
            wgt_read_bits = $realtobits(top_level_module.u_mem_weights.wq[addr - `WGT_WQ_BASE]);
        end else if (addr < `WGT_WV_BASE) begin
            wgt_read_bits = $realtobits(top_level_module.u_mem_weights.wk[addr - `WGT_WK_BASE]);
        end else if (addr < `WGT_WO_BASE) begin
            wgt_read_bits = $realtobits(top_level_module.u_mem_weights.wv[addr - `WGT_WV_BASE]);
        end else begin
            wgt_read_bits = $realtobits(top_level_module.u_mem_weights.wo[addr - `WGT_WO_BASE]);
        end
    end
endfunction

assign rms_act_rd_data = act_read_bits(rms_act_rd_addr);
assign matmul_act_rd_data = act_read_bits(matmul_act_rd_addr);
assign rope_act_rd_data = act_read_bits(rope_act_rd_addr);
assign softmax_act_rd_data = act_read_bits(softmax_act_rd_addr);
assign rms_wgt_rd_data = $realtobits(top_level_module.u_mem_weights.rms_att_weight[rms_wgt_rd_addr]);
assign matmul_wgt_rd_data = wgt_read_bits(matmul_wgt_rd_addr);

always @(*) begin
    if (rms_act_wr_en) begin
        top_level_module.u_mem_activation.xb[rms_act_wr_addr - `ACT_XB_BASE] = $bitstoreal(rms_act_wr_data);
    end
    if (matmul_act_wr_en) begin
        if (matmul_act_wr_addr >= `ACT_Q_BASE && matmul_act_wr_addr < `ACT_K_BASE) begin
            top_level_module.u_mem_activation.q[matmul_act_wr_addr - `ACT_Q_BASE] = $bitstoreal(matmul_act_wr_data);
        end else if (matmul_act_wr_addr >= `ACT_K_BASE && matmul_act_wr_addr < `ACT_V_BASE) begin
            top_level_module.u_mem_activation.k_vec[matmul_act_wr_addr - `ACT_K_BASE] = $bitstoreal(matmul_act_wr_data);
        end else if (matmul_act_wr_addr >= `ACT_V_BASE && matmul_act_wr_addr < `ACT_ATT_BASE) begin
            top_level_module.u_mem_activation.v_vec[matmul_act_wr_addr - `ACT_V_BASE] = $bitstoreal(matmul_act_wr_data);
        end else if (matmul_act_wr_addr >= `ACT_XB2_BASE && matmul_act_wr_addr < `ACT_XB2_BASE + DIM) begin
            top_level_module.u_mem_activation.xb2[matmul_act_wr_addr - `ACT_XB2_BASE] = $bitstoreal(matmul_act_wr_data);
        end else if (matmul_act_wr_addr < DIM) begin
            top_level_module.u_mem_activation.x[matmul_act_wr_addr] = $bitstoreal(matmul_act_wr_data);
        end
    end
    if (rope_act_wr_en) begin
        if (rope_act_wr_addr >= `ACT_Q_BASE && rope_act_wr_addr < `ACT_K_BASE) begin
            top_level_module.u_mem_activation.q[rope_act_wr_addr - `ACT_Q_BASE] = $bitstoreal(rope_act_wr_data);
        end else if (rope_act_wr_addr >= `ACT_K_BASE && rope_act_wr_addr < `ACT_V_BASE) begin
            top_level_module.u_mem_activation.k_vec[rope_act_wr_addr - `ACT_K_BASE] = $bitstoreal(rope_act_wr_data);
        end
    end
    if (softmax_act_wr_en) begin
        top_level_module.u_mem_activation.att[softmax_act_wr_addr - `ACT_ATT_BASE] = $bitstoreal(softmax_act_wr_data);
    end
end

task read_kv;
    input integer addr;
    output real value;
    begin
        kv_rd_en = 1'b1;
        kv_rd_addr = addr;
        #0;
        value = $bitstoreal(kv_rd_data);
        kv_rd_en = 1'b0;
    end
endtask

task write_kv;
    input integer addr;
    input real value;
    begin
        kv_wr_en = 1'b1;
        kv_wr_addr = addr;
        kv_wr_data = $realtobits(value);
        #0;
        kv_wr_en = 1'b0;
    end
endtask

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
        end else if (addr >= `ACT_XB2_BASE && addr < `ACT_XB2_BASE + `ACT_XB2_SIZE) begin
            top_level_module.u_mem_activation.xb2[addr - `ACT_XB2_BASE] = value;
        end else if (addr >= `ACT_Q_BASE && addr < `ACT_Q_BASE + `ACT_Q_SIZE) begin
            top_level_module.u_mem_activation.q[addr - `ACT_Q_BASE] = value;
        end else if (addr >= `ACT_K_BASE && addr < `ACT_K_BASE + `ACT_K_SIZE) begin
            top_level_module.u_mem_activation.k_vec[addr - `ACT_K_BASE] = value;
        end else if (addr >= `ACT_V_BASE && addr < `ACT_V_BASE + `ACT_V_SIZE) begin
            top_level_module.u_mem_activation.v_vec[addr - `ACT_V_BASE] = value;
        end else if (addr >= `ACT_ATT_BASE && addr < `ACT_ATT_BASE + `ACT_ATT_SIZE) begin
            top_level_module.u_mem_activation.att[addr - `ACT_ATT_BASE] = value;
        end
    end
endtask

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
    .N_HEADS(N_HEADS),
    .N_KV_HEADS(N_KV_HEADS),
    .MAX_SEQ_LEN(MAX_SEQ_LEN)
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

kernel_rope #(
    .DIM(DIM),
    .N_HEADS(N_HEADS),
    .N_KV_HEADS(N_KV_HEADS),
    .ACT_Q_BASE(`ACT_Q_BASE),
    .ACT_K_BASE(`ACT_K_BASE)
) u_rope (
    .act_rd_en(rope_act_rd_en),
    .act_rd_addr(rope_act_rd_addr),
    .act_rd_data(rope_act_rd_data),
    .act_wr_en(rope_act_wr_en),
    .act_wr_addr(rope_act_wr_addr),
    .act_wr_data(rope_act_wr_data)
);

kernel_softmax #(
    .N_HEADS(N_HEADS),
    .MAX_SEQ_LEN(MAX_SEQ_LEN),
    .ACT_ATT_BASE(`ACT_ATT_BASE)
) u_softmax (
    .act_rd_en(softmax_act_rd_en),
    .act_rd_addr(softmax_act_rd_addr),
    .act_rd_data(softmax_act_rd_data),
    .act_wr_en(softmax_act_wr_en),
    .act_wr_addr(softmax_act_wr_addr),
    .act_wr_data(softmax_act_wr_data)
);

integer i;
integer head_idx;
integer timestep;
integer head_base;
integer kv_head;
integer cache_base;
integer att_base;
integer loff;
real score;
real inv_scale;
real act_value_a;
real act_value_b;

task run;
    input integer task_layer_idx;
    input integer task_pos_idx;
    begin
        u_rmsnorm.apply_attn(task_layer_idx);
        u_matmul.project_qkv(task_layer_idx);
        u_rope.apply(task_pos_idx);

        loff = (task_layer_idx * MAX_SEQ_LEN + task_pos_idx) * KV_DIM;
        for (i = 0; i < KV_DIM; i = i + 1) begin
            read_act_local(`ACT_K_BASE + i, kv_value);
            write_kv(`KV_KEY_BASE + loff + i, kv_value);
            read_act_local(`ACT_V_BASE + i, kv_value);
            write_kv(`KV_VALUE_BASE + loff + i, kv_value);
        end

        inv_scale = 1.0 / $sqrt(HEAD_SIZE);
        for (i = 0; i < DIM; i = i + 1) begin
            write_act_local(`ACT_XB_BASE + i, 0.0);
        end

        loff = task_layer_idx * MAX_SEQ_LEN * KV_DIM;
        for (head_idx = 0; head_idx < N_HEADS; head_idx = head_idx + 1) begin
            head_base = head_idx * HEAD_SIZE;
            att_base = head_idx * MAX_SEQ_LEN;
            kv_head = head_idx / KV_MUL;

            for (timestep = 0; timestep <= task_pos_idx; timestep = timestep + 1) begin
                cache_base = loff + timestep * KV_DIM + kv_head * HEAD_SIZE;
                score = 0.0;
                for (i = 0; i < HEAD_SIZE; i = i + 1) begin
                    read_act_local(`ACT_Q_BASE + head_base + i, act_value_a);
                    read_kv(`KV_KEY_BASE + cache_base + i, kv_value);
                    score = score + (act_value_a * kv_value);
                end
                write_act_local(`ACT_ATT_BASE + att_base + timestep, fp32_round(score * inv_scale));
            end

            u_softmax.normalize_head(head_idx, task_pos_idx);

            for (i = 0; i < HEAD_SIZE; i = i + 1) begin
                write_act_local(`ACT_XB_BASE + head_base + i, 0.0);
            end
            for (timestep = 0; timestep <= task_pos_idx; timestep = timestep + 1) begin
                cache_base = loff + timestep * KV_DIM + kv_head * HEAD_SIZE;
                for (i = 0; i < HEAD_SIZE; i = i + 1) begin
                    read_act_local(`ACT_XB_BASE + head_base + i, act_value_a);
                    read_act_local(`ACT_ATT_BASE + att_base + timestep, act_value_b);
                    read_kv(`KV_VALUE_BASE + cache_base + i, kv_value);
                    write_act_local(
                        `ACT_XB_BASE + head_base + i,
                        fp32_round(act_value_a + (act_value_b * kv_value))
                    );
                end
            end
        end

        u_matmul.project_attention_output(task_layer_idx);
    end
endtask

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        kv_rd_en <= 1'b0;
        kv_rd_addr <= 32'd0;
        kv_wr_en <= 1'b0;
        kv_wr_addr <= 32'd0;
        kv_wr_data <= 64'd0;
        busy <= 1'b0;
        done <= 1'b0;
    end else begin
        kv_rd_en <= 1'b0;
        kv_wr_en <= 1'b0;
        done <= 1'b0;
        if (start) begin
            busy <= 1'b1;
            run(layer_idx, pos_idx);
            busy <= 1'b0;
            done <= 1'b1;
        end else begin
            busy <= 1'b0;
        end
    end
end

endmodule
