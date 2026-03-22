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

localparam HEAD_SIZE = DIM / N_HEADS;
localparam KV_DIM = (DIM * N_KV_HEADS) / N_HEADS;
localparam LUT_SEQ_LEN = 512;

localparam STATE_IDLE       = 4'd0;
localparam STATE_PREP       = 4'd1;
localparam STATE_READ0_REQ  = 4'd2;
localparam STATE_READ0_WAIT = 4'd3;
localparam STATE_READ0_CAP  = 4'd4;
localparam STATE_READ1_REQ  = 4'd5;
localparam STATE_READ1_WAIT = 4'd6;
localparam STATE_READ1_CAP  = 4'd7;
localparam STATE_CALC       = 4'd8;
localparam STATE_WRITE0     = 4'd9;
localparam STATE_WRITE1     = 4'd10;
localparam STATE_NEXT_VEC   = 4'd11;
localparam STATE_NEXT_ELEM  = 4'd12;
localparam STATE_DONE       = 4'd13;

reg [3:0] state;
reg [9:0] pos_idx_reg;
reg [31:0] elem_idx;
reg [31:0] vec_sel;
reg [31:0] rot_count;
reg [31:0] lut_idx;
reg [31:0] sin_bits;
reg [31:0] cos_bits;
reg [31:0] v0_bits;
reg [31:0] v1_bits;
reg [31:0] out0_bits;
reg [31:0] out1_bits;

reg [31:0] sin_lut [0:2047];
reg [31:0] cos_lut [0:2047];
integer lut_fd;

wire [31:0] neg_m1_y;
wire [31:0] m0_y;
wire [31:0] m1_y;
wire [31:0] m2_y;
wire [31:0] m3_y;
wire [31:0] out0_y;
wire [31:0] out1_y;

function [31:0] vec_base_addr;
    input [31:0] vec_idx;
    begin
        if (vec_idx == 32'd0) begin
            vec_base_addr = ACT_Q_BASE;
        end else begin
            vec_base_addr = ACT_K_BASE;
        end
    end
endfunction

function [31:0] rope_freq_sel;
    input [31:0] current_elem_idx;
    reg [31:0] head_dim;
    begin
        head_dim = current_elem_idx % HEAD_SIZE;
        case (head_dim)
            32'd0: rope_freq_sel = 32'd0;
            32'd2: rope_freq_sel = 32'd1;
            32'd4: rope_freq_sel = 32'd2;
            32'd6: rope_freq_sel = 32'd3;
            default: rope_freq_sel = 32'd0;
        endcase
    end
endfunction

initial begin
    lut_fd = $fopen("src/LUTs/rope_sin_lut.hex", "r");
    if (lut_fd != 0) begin
        $fclose(lut_fd);
        $readmemh("src/LUTs/rope_sin_lut.hex", sin_lut);
    end else begin
        lut_fd = $fopen("../src/LUTs/rope_sin_lut.hex", "r");
        if (lut_fd != 0) begin
            $fclose(lut_fd);
            $readmemh("../src/LUTs/rope_sin_lut.hex", sin_lut);
        end else begin
            lut_fd = $fopen("../../src/LUTs/rope_sin_lut.hex", "r");
            if (lut_fd != 0) begin
                $fclose(lut_fd);
                $readmemh("../../src/LUTs/rope_sin_lut.hex", sin_lut);
            end else begin
                $display("ERROR: kernel_rope could not locate rope_sin_lut.hex");
                $finish;
            end
        end
    end

    lut_fd = $fopen("src/LUTs/rope_cos_lut.hex", "r");
    if (lut_fd != 0) begin
        $fclose(lut_fd);
        $readmemh("src/LUTs/rope_cos_lut.hex", cos_lut);
    end else begin
        lut_fd = $fopen("../src/LUTs/rope_cos_lut.hex", "r");
        if (lut_fd != 0) begin
            $fclose(lut_fd);
            $readmemh("../src/LUTs/rope_cos_lut.hex", cos_lut);
        end else begin
            lut_fd = $fopen("../../src/LUTs/rope_cos_lut.hex", "r");
            if (lut_fd != 0) begin
                $fclose(lut_fd);
                $readmemh("../../src/LUTs/rope_cos_lut.hex", cos_lut);
            end else begin
                $display("ERROR: kernel_rope could not locate rope_cos_lut.hex");
                $finish;
            end
        end
    end
end

kernel_mul u_mul_v0_cos (
    .a(v0_bits),
    .b(cos_bits),
    .y(m0_y)
);

kernel_mul u_mul_v1_sin (
    .a(v1_bits),
    .b(sin_bits),
    .y(m1_y)
);

kernel_mul u_mul_v0_sin (
    .a(v0_bits),
    .b(sin_bits),
    .y(m2_y)
);

kernel_mul u_mul_v1_cos (
    .a(v1_bits),
    .b(cos_bits),
    .y(m3_y)
);

assign neg_m1_y = {~m1_y[31], m1_y[30:0]};

kernel_add u_add_out0 (
    .a(m0_y),
    .b(neg_m1_y),
    .y(out0_y)
);

kernel_add u_add_out1 (
    .a(m2_y),
    .b(m3_y),
    .y(out1_y)
);

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
        elem_idx <= 32'd0;
        vec_sel <= 32'd0;
        rot_count <= 32'd0;
        lut_idx <= 32'd0;
        sin_bits <= 32'd0;
        cos_bits <= 32'd0;
        v0_bits <= 32'd0;
        v1_bits <= 32'd0;
        out0_bits <= 32'd0;
        out1_bits <= 32'd0;
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
                    elem_idx <= 32'd0;
                    vec_sel <= 32'd0;
                    state <= STATE_PREP;
                end
            end

            STATE_PREP: begin
                busy <= 1'b1;
                lut_idx <= rope_freq_sel(elem_idx) * LUT_SEQ_LEN + {22'd0, pos_idx_reg};
                sin_bits <= sin_lut[rope_freq_sel(elem_idx) * LUT_SEQ_LEN + {22'd0, pos_idx_reg}];
                cos_bits <= cos_lut[rope_freq_sel(elem_idx) * LUT_SEQ_LEN + {22'd0, pos_idx_reg}];
                if (elem_idx < KV_DIM) begin
                    rot_count <= 32'd2;
                end else begin
                    rot_count <= 32'd1;
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
                v0_bits <= act_rd_data;
                state <= STATE_READ1_REQ;
            end

            STATE_READ1_REQ: begin
                busy <= 1'b1;
                act_rd_en <= 1'b1;
                act_rd_addr <= vec_base_addr(vec_sel) + elem_idx + 32'd1;
                state <= STATE_READ1_WAIT;
            end

            STATE_READ1_WAIT: begin
                busy <= 1'b1;
                state <= STATE_READ1_CAP;
            end

            STATE_READ1_CAP: begin
                busy <= 1'b1;
                v1_bits <= act_rd_data;
                state <= STATE_CALC;
            end

            STATE_CALC: begin
                busy <= 1'b1;
                out0_bits <= out0_y;
                out1_bits <= out1_y;
                state <= STATE_WRITE0;
            end

            STATE_WRITE0: begin
                busy <= 1'b1;
                act_wr_en <= 1'b1;
                act_wr_addr <= vec_base_addr(vec_sel) + elem_idx;
                act_wr_data <= out0_bits;
                state <= STATE_WRITE1;
            end

            STATE_WRITE1: begin
                busy <= 1'b1;
                act_wr_en <= 1'b1;
                act_wr_addr <= vec_base_addr(vec_sel) + elem_idx + 32'd1;
                act_wr_data <= out1_bits;
                state <= STATE_NEXT_VEC;
            end

            STATE_NEXT_VEC: begin
                busy <= 1'b1;
                if ((vec_sel + 32'd1) < rot_count) begin
                    vec_sel <= vec_sel + 32'd1;
                    state <= STATE_READ0_REQ;
                end else begin
                    vec_sel <= 32'd0;
                    state <= STATE_NEXT_ELEM;
                end
            end

            STATE_NEXT_ELEM: begin
                busy <= 1'b1;
                if ((elem_idx + 32'd2) >= DIM) begin
                    state <= STATE_DONE;
                end else begin
                    elem_idx <= elem_idx + 32'd2;
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
