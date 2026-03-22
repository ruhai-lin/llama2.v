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

localparam VOCAB_SIZE = 512;
localparam N_LAYERS = 5;
localparam KV_DIM = (DIM * N_KV_HEADS) / N_HEADS;
`include "memory_map.vh"

localparam STATE_IDLE       = 5'd0;
localparam STATE_RMS_START  = 5'd1;
localparam STATE_RMS_WAIT   = 5'd2;
localparam STATE_W13_START  = 5'd3;
localparam STATE_W13_WAIT   = 5'd4;
localparam STATE_SILU_RD0   = 5'd5;
localparam STATE_SILU_RD0W  = 5'd6;
localparam STATE_SILU_RD0C  = 5'd7;
localparam STATE_SILU_RD1   = 5'd8;
localparam STATE_SILU_RD1W  = 5'd9;
localparam STATE_SILU_RD1C  = 5'd10;
localparam STATE_SILU_EXP   = 5'd11;
localparam STATE_SILU_DENOM = 5'd12;
localparam STATE_SILU_RECIP = 5'd13;
localparam STATE_SILU_SIGM  = 5'd14;
localparam STATE_SILU_XMUL  = 5'd15;
localparam STATE_SILU_FINAL = 5'd16;
localparam STATE_SILU_WR    = 5'd17;
localparam STATE_W2_START   = 5'd18;
localparam STATE_W2_WAIT    = 5'd19;
localparam STATE_DONE       = 5'd20;

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
reg [4:0] state;
integer silu_idx;
reg [31:0] hb_bits;
reg [31:0] hb2_bits;
reg [31:0] exp_input_bits;
reg [31:0] exp_bits;
reg [31:0] numerator_bits;
reg [31:0] denom_bits;
reg [31:0] recip_bits;
reg [31:0] sigmoid_bits;
reg [31:0] silu_bits;

reg [31:0] exp_lut [0:4095];
reg [31:0] reciprocal_lut [0:4095];
integer lut_fd;
reg [31:0] mul_a;
reg [31:0] mul_b;
reg [31:0] add_a;
reg [31:0] add_b;
wire [31:0] mul_y;
wire [31:0] add_y;

localparam [31:0] FP_ONE_BITS = 32'h3f800000;

assign rms_act_rd_data = act_rd_data;
assign matmul_act_rd_data = act_rd_data;
assign rms_wgt_rd_data = wgt_rd_data;
assign matmul_wgt_rd_data = wgt_rd_data;

function [11:0] lut_index_from_bits;
    input [31:0] bits;
    begin
        lut_index_from_bits = {bits[30:23], bits[22:19]};
    end
endfunction

function fp32_is_zero;
    input [31:0] bits;
    begin
        fp32_is_zero = (bits[30:0] == 31'd0);
    end
endfunction

function [11:0] exp_lut_index_nonpos;
    input [31:0] bits;
    begin
        if (fp32_is_zero(bits)) begin
            exp_lut_index_nonpos = 12'd0;
        end else if (bits[31] == 1'b0) begin
            exp_lut_index_nonpos = 12'd0;
        end else begin
            exp_lut_index_nonpos = lut_index_from_bits(bits);
        end
    end
endfunction

initial begin
    lut_fd = $fopen("src/LUTs/exp_lut.hex", "r");
    if (lut_fd != 0) begin
        $fclose(lut_fd);
        $readmemh("src/LUTs/exp_lut.hex", exp_lut);
    end else begin
        lut_fd = $fopen("../src/LUTs/exp_lut.hex", "r");
        if (lut_fd != 0) begin
            $fclose(lut_fd);
            $readmemh("../src/LUTs/exp_lut.hex", exp_lut);
        end else begin
            lut_fd = $fopen("../../src/LUTs/exp_lut.hex", "r");
            if (lut_fd != 0) begin
                $fclose(lut_fd);
                $readmemh("../../src/LUTs/exp_lut.hex", exp_lut);
            end else begin
                $display("ERROR: ffn could not locate exp_lut.hex");
                $finish;
            end
        end
    end

    lut_fd = $fopen("src/LUTs/reciprocal_lut.hex", "r");
    if (lut_fd != 0) begin
        $fclose(lut_fd);
        $readmemh("src/LUTs/reciprocal_lut.hex", reciprocal_lut);
    end else begin
        lut_fd = $fopen("../src/LUTs/reciprocal_lut.hex", "r");
        if (lut_fd != 0) begin
            $fclose(lut_fd);
            $readmemh("../src/LUTs/reciprocal_lut.hex", reciprocal_lut);
        end else begin
            lut_fd = $fopen("../../src/LUTs/reciprocal_lut.hex", "r");
            if (lut_fd != 0) begin
                $fclose(lut_fd);
                $readmemh("../../src/LUTs/reciprocal_lut.hex", reciprocal_lut);
            end else begin
                $display("ERROR: ffn could not locate reciprocal_lut.hex");
                $finish;
            end
        end
    end
end

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

    mul_a = 32'd0;
    mul_b = 32'd0;
    add_a = 32'd0;
    add_b = 32'd0;

    case (state)
        STATE_SILU_DENOM: begin
            add_a = FP_ONE_BITS;
            add_b = exp_bits;
        end
        STATE_SILU_SIGM: begin
            mul_a = numerator_bits;
            mul_b = recip_bits;
        end
        STATE_SILU_XMUL: begin
            mul_a = hb_bits;
            mul_b = sigmoid_bits;
        end
        STATE_SILU_FINAL: begin
            mul_a = silu_bits;
            mul_b = hb2_bits;
        end
        default: begin
        end
    endcase
end

kernel_mul u_local_mul (
    .a(mul_a),
    .b(mul_b),
    .y(mul_y)
);

kernel_add u_local_add (
    .a(add_a),
    .b(add_b),
    .y(add_y)
);

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
        silu_idx <= 0;
        hb_bits <= 32'd0;
        hb2_bits <= 32'd0;
        exp_input_bits <= 32'd0;
        exp_bits <= 32'd0;
        numerator_bits <= 32'd0;
        denom_bits <= 32'd0;
        recip_bits <= 32'd0;
        sigmoid_bits <= 32'd0;
        silu_bits <= 32'd0;
    end else begin
        done <= 1'b0;
        local_act_rd_en <= 1'b0;
        local_act_wr_en <= 1'b0;

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
                    silu_idx <= 0;
                    state <= STATE_SILU_RD0;
                end
            end

            STATE_SILU_RD0: begin
                busy <= 1'b1;
                local_act_rd_en <= 1'b1;
                local_act_rd_addr <= `ACT_HB_BASE + silu_idx;
                state <= STATE_SILU_RD0W;
            end

            STATE_SILU_RD0W: begin
                busy <= 1'b1;
                state <= STATE_SILU_RD0C;
            end

            STATE_SILU_RD0C: begin
                busy <= 1'b1;
                hb_bits <= act_rd_data;
                state <= STATE_SILU_RD1;
            end

            STATE_SILU_RD1: begin
                busy <= 1'b1;
                local_act_rd_en <= 1'b1;
                local_act_rd_addr <= `ACT_HB2_BASE + silu_idx;
                state <= STATE_SILU_RD1W;
            end

            STATE_SILU_RD1W: begin
                busy <= 1'b1;
                state <= STATE_SILU_RD1C;
            end

            STATE_SILU_RD1C: begin
                busy <= 1'b1;
                hb2_bits <= act_rd_data;
                if (hb_bits[31]) begin
                    exp_input_bits <= hb_bits;
                end else begin
                    exp_input_bits <= {~hb_bits[31], hb_bits[30:0]};
                end
                state <= STATE_SILU_EXP;
            end

            STATE_SILU_EXP: begin
                busy <= 1'b1;
                exp_bits <= exp_lut[exp_lut_index_nonpos(exp_input_bits)];
                state <= STATE_SILU_DENOM;
            end

            STATE_SILU_DENOM: begin
                busy <= 1'b1;
                denom_bits <= add_y;
                if (hb_bits[31]) begin
                    numerator_bits <= exp_bits;
                end else begin
                    numerator_bits <= FP_ONE_BITS;
                end
                state <= STATE_SILU_RECIP;
            end

            STATE_SILU_RECIP: begin
                busy <= 1'b1;
                recip_bits <= reciprocal_lut[lut_index_from_bits(denom_bits)];
                state <= STATE_SILU_SIGM;
            end

            STATE_SILU_SIGM: begin
                busy <= 1'b1;
                sigmoid_bits <= mul_y;
                state <= STATE_SILU_XMUL;
            end

            STATE_SILU_XMUL: begin
                busy <= 1'b1;
                silu_bits <= mul_y;
                state <= STATE_SILU_FINAL;
            end

            STATE_SILU_FINAL: begin
                busy <= 1'b1;
                local_act_wr_data <= mul_y;
                state <= STATE_SILU_WR;
            end

            STATE_SILU_WR: begin
                busy <= 1'b1;
                local_act_wr_en <= 1'b1;
                local_act_wr_addr <= `ACT_HB_BASE + silu_idx;
                if (silu_idx == (HIDDEN_DIM - 1)) begin
                    state <= STATE_W2_START;
                end else begin
                    silu_idx <= silu_idx + 1;
                    state <= STATE_SILU_RD0;
                end
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
