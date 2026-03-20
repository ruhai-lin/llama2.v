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
    output reg act_rd_en,
    output reg [31:0] act_rd_addr,
    input wire [31:0] act_rd_data,
    output reg act_wr_en,
    output reg [31:0] act_wr_addr,
    output reg [31:0] act_wr_data,
    output reg wgt_rd_en,
    output reg [31:0] wgt_rd_addr,
    input wire [31:0] wgt_rd_data,
    output reg busy,
    output reg done
);

`include "real_fp32_helpers.vh"

localparam VOCAB_SIZE = 512;
localparam N_LAYERS = 5;
localparam KV_DIM = (DIM * N_KV_HEADS) / N_HEADS;

localparam STATE_IDLE      = 4'd0;
localparam STATE_RMS_START = 4'd1;
localparam STATE_RMS_WAIT  = 4'd2;
localparam STATE_W13_START = 4'd3;
localparam STATE_W13_WAIT  = 4'd4;
localparam STATE_SILU      = 4'd5;
localparam STATE_W2_START  = 4'd6;
localparam STATE_W2_WAIT   = 4'd7;
localparam STATE_DONE      = 4'd8;

localparam RMS_OP_FFN      = 2'd2;
localparam MATMUL_OP_W1_W3 = 3'd3;
localparam MATMUL_OP_W2    = 3'd4;

wire rms_act_rd_en;
wire [31:0] rms_act_rd_addr;
wire [31:0] rms_act_rd_data;
wire rms_act_wr_en;
wire [31:0] rms_act_wr_addr;
wire [31:0] rms_act_wr_data;
wire rms_wgt_rd_en;
wire [31:0] rms_wgt_rd_addr;
wire [31:0] rms_wgt_rd_data;
wire rms_busy;
wire rms_done;

wire matmul_act_rd_en;
wire [31:0] matmul_act_rd_addr;
wire [31:0] matmul_act_rd_data;
wire matmul_act_wr_en;
wire [31:0] matmul_act_wr_addr;
wire [31:0] matmul_act_wr_data;
wire matmul_wgt_rd_en;
wire [31:0] matmul_wgt_rd_addr;
wire [31:0] matmul_wgt_rd_data;
wire matmul_busy;
wire matmul_done;

reg local_act_rd_en;
reg [31:0] local_act_rd_addr;
reg local_act_wr_en;
reg [31:0] local_act_wr_addr;
reg [31:0] local_act_wr_data;
reg [3:0] state;

integer i;
real val;
real hb_val;
real hb2_val;

assign rms_act_rd_data = act_rd_data;
assign matmul_act_rd_data = act_rd_data;
assign rms_wgt_rd_data = wgt_rd_data;
assign matmul_wgt_rd_data = wgt_rd_data;

always @(*) begin
    act_rd_en = 1'b0;
    act_rd_addr = 32'd0;
    act_wr_en = 1'b0;
    act_wr_addr = 32'd0;
    act_wr_data = 32'd0;
    wgt_rd_en = 1'b0;
    wgt_rd_addr = 32'd0;

    if (local_act_rd_en) begin
        act_rd_en = 1'b1;
        act_rd_addr = local_act_rd_addr;
    end else if (matmul_act_rd_en) begin
        act_rd_en = 1'b1;
        act_rd_addr = matmul_act_rd_addr;
    end else if (rms_act_rd_en) begin
        act_rd_en = 1'b1;
        act_rd_addr = rms_act_rd_addr;
    end

    if (local_act_wr_en) begin
        act_wr_en = 1'b1;
        act_wr_addr = local_act_wr_addr;
        act_wr_data = local_act_wr_data;
    end else if (matmul_act_wr_en) begin
        act_wr_en = 1'b1;
        act_wr_addr = matmul_act_wr_addr;
        act_wr_data = matmul_act_wr_data;
    end else if (rms_act_wr_en) begin
        act_wr_en = 1'b1;
        act_wr_addr = rms_act_wr_addr;
        act_wr_data = rms_act_wr_data;
    end

    if (matmul_wgt_rd_en) begin
        wgt_rd_en = 1'b1;
        wgt_rd_addr = matmul_wgt_rd_addr;
    end else if (rms_wgt_rd_en) begin
        wgt_rd_en = 1'b1;
        wgt_rd_addr = `WGT_RMS_FFN_BASE + rms_wgt_rd_addr;
    end
end

task read_act_local;
    input integer addr;
    output real value;
    begin
        local_act_rd_addr = addr;
        local_act_rd_en = 1'b1;
        @(posedge clk);
        local_act_rd_en = 1'b0;
        @(negedge clk);
        value = fp32_to_real(act_rd_data);
    end
endtask

task write_act_local;
    input integer addr;
    input real value;
    begin
        local_act_wr_addr = addr;
        local_act_wr_data = real_to_fp32_bits(value);
        local_act_wr_en = 1'b1;
        @(posedge clk);
        local_act_wr_en = 1'b0;
    end
endtask

task apply_silu;
    begin
        for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
            read_act_local(`ACT_HB_BASE + i, hb_val);
            read_act_local(`ACT_HB2_BASE + i, hb2_val);
            val = hb_val;
            val = val * (1.0 / (1.0 + $exp(-val)));
            write_act_local(`ACT_HB_BASE + i, fp32_round(val * hb2_val));
        end
    end
endtask

kernel_rmsnorm #(
    .DIM(DIM)
) u_rmsnorm (
    .clk(clk),
    .rst_n(rst_n),
    .start(state == STATE_RMS_START),
    .layer_idx({29'd0, layer_idx}),
    .op_code(RMS_OP_FFN),
    .busy(rms_busy),
    .done(rms_done),
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
    .start((state == STATE_W13_START) || (state == STATE_W2_START)),
    .layer_idx({29'd0, layer_idx}),
    .op_code((state == STATE_W13_START) ? MATMUL_OP_W1_W3 : MATMUL_OP_W2),
    .busy(matmul_busy),
    .done(matmul_done),
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

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        local_act_rd_en <= 1'b0;
        local_act_rd_addr <= 32'd0;
        local_act_wr_en <= 1'b0;
        local_act_wr_addr <= 32'd0;
        local_act_wr_data <= 32'd0;
        state <= STATE_IDLE;
        busy <= 1'b0;
        done <= 1'b0;
    end else begin
        done <= 1'b0;
        case (state)
            STATE_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    busy <= 1'b1;
                    state <= STATE_RMS_START;
                end
            end

            STATE_RMS_START: begin
                busy <= 1'b1;
                state <= STATE_RMS_WAIT;
            end

            STATE_RMS_WAIT: begin
                busy <= 1'b1;
                if (rms_done) begin
                    state <= STATE_W13_START;
                end
            end

            STATE_W13_START: begin
                busy <= 1'b1;
                state <= STATE_W13_WAIT;
            end

            STATE_W13_WAIT: begin
                busy <= 1'b1;
                if (matmul_done) begin
                    state <= STATE_SILU;
                end
            end

            STATE_SILU: begin
                busy <= 1'b1;
                apply_silu();
                state <= STATE_W2_START;
            end

            STATE_W2_START: begin
                busy <= 1'b1;
                state <= STATE_W2_WAIT;
            end

            STATE_W2_WAIT: begin
                busy <= 1'b1;
                if (matmul_done) begin
                    state <= STATE_DONE;
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
