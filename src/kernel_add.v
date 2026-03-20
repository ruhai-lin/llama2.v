module kernel_add (
    input wire [31:0] a,
    input wire [31:0] b,
    output reg [31:0] y
);

reg sign_a;
reg sign_b;
reg sign_big;
reg sign_small;
reg [7:0] exp_a;
reg [7:0] exp_b;
reg [7:0] exp_big;
reg [7:0] exp_small;
reg [23:0] mant_a;
reg [23:0] mant_b;
reg [23:0] mant_big_src;
reg [23:0] mant_small_src;
reg [26:0] mant_big;
reg [26:0] mant_small;
reg [26:0] mant_small_shifted;
reg [27:0] mant_sum;
reg [26:0] mant_norm;
reg [24:0] mant_round_ext;
reg guard_bit;
reg round_bit;
reg sticky_bit;
integer exp_diff;
integer exp_work;
integer shift_idx;
reg sign_out;

always @(*) begin
    sign_a = a[31];
    sign_b = b[31];
    exp_a = a[30:23];
    exp_b = b[30:23];
    mant_a = (exp_a == 8'd0) ? {1'b0, a[22:0]} : {1'b1, a[22:0]};
    mant_b = (exp_b == 8'd0) ? {1'b0, b[22:0]} : {1'b1, b[22:0]};
    sign_big = 1'b0;
    sign_small = 1'b0;
    exp_big = 8'd0;
    exp_small = 8'd0;
    mant_big_src = 24'd0;
    mant_small_src = 24'd0;
    mant_big = 27'd0;
    mant_small = 27'd0;
    mant_small_shifted = 27'd0;
    mant_sum = 28'd0;
    mant_norm = 27'd0;
    mant_round_ext = 25'd0;
    guard_bit = 1'b0;
    round_bit = 1'b0;
    sticky_bit = 1'b0;
    exp_diff = 0;
    exp_work = 0;
    shift_idx = 0;
    sign_out = 1'b0;
    y = 32'd0;

    if ((exp_a == 8'd0) && (a[22:0] == 23'd0)) begin
        y = b;
    end else if ((exp_b == 8'd0) && (b[22:0] == 23'd0)) begin
        y = a;
    end else begin
        if ((exp_a > exp_b) || ((exp_a == exp_b) && (mant_a >= mant_b))) begin
            exp_big = exp_a;
            exp_small = exp_b;
            mant_big_src = mant_a;
            mant_small_src = mant_b;
            sign_big = sign_a;
            sign_small = sign_b;
        end else begin
            exp_big = exp_b;
            exp_small = exp_a;
            mant_big_src = mant_b;
            mant_small_src = mant_a;
            sign_big = sign_b;
            sign_small = sign_a;
        end

        mant_big = {mant_big_src, 3'b000};
        mant_small = {mant_small_src, 3'b000};
        exp_diff = {24'd0, exp_big} - {24'd0, exp_small};
        exp_work = (exp_big == 8'd0) ? 32'd1 : {24'd0, exp_big};
        sign_out = sign_big;

        if (exp_diff >= 27) begin
            mant_small_shifted = (mant_small != 27'd0) ? 27'd1 : 27'd0;
        end else begin
            mant_small_shifted = mant_small >> exp_diff;
            if ((exp_diff > 0) && ((mant_small & ((27'd1 << exp_diff) - 1)) != 0)) begin
                mant_small_shifted[0] = 1'b1;
            end
        end

        if (sign_big == sign_small) begin
            mant_sum = {1'b0, mant_big} + {1'b0, mant_small_shifted};
            if (mant_sum[27]) begin
                mant_norm = mant_sum[27:1];
                if (mant_sum[0]) begin
                    mant_norm[0] = 1'b1;
                end
                exp_work = exp_work + 1;
            end else begin
                mant_norm = mant_sum[26:0];
            end
        end else begin
            mant_sum = {1'b0, mant_big} - {1'b0, mant_small_shifted};
            if (mant_sum[26:0] == 27'd0) begin
                y = 32'd0;
                exp_work = 0;
            end else begin
                mant_norm = mant_sum[26:0];
                for (shift_idx = 0; shift_idx < 26; shift_idx = shift_idx + 1) begin
                    if ((mant_norm[26] == 1'b0) && (exp_work > 1)) begin
                        mant_norm = mant_norm << 1;
                        exp_work = exp_work - 1;
                    end
                end
            end
        end

        if (exp_work > 0) begin
            mant_round_ext = {1'b0, mant_norm[26:3]};
            guard_bit = mant_norm[2];
            round_bit = mant_norm[1];
            sticky_bit = mant_norm[0];

            if (guard_bit && (round_bit || sticky_bit || mant_round_ext[0])) begin
                mant_round_ext = mant_round_ext + 25'd1;
            end

            if (mant_round_ext[24]) begin
                mant_round_ext = mant_round_ext >> 1;
                exp_work = exp_work + 1;
            end

            if (exp_work >= 255) begin
                y = {sign_out, 8'hFF, 23'd0};
            end else if (exp_work <= 0) begin
                y = 32'd0;
            end else begin
                y = {sign_out, exp_work[7:0], mant_round_ext[22:0]};
            end
        end
    end
end

endmodule
