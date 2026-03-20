module kernel_mul (
    input wire [31:0] a,
    input wire [31:0] b,
    output reg [31:0] y
);

reg sign_y;
reg [7:0] exp_a;
reg [7:0] exp_b;
reg [23:0] mant_a;
reg [23:0] mant_b;
reg [47:0] product;
reg [26:0] mant_ext;
reg [24:0] mant_round_ext;
reg guard_bit;
reg round_bit;
reg sticky_bit;
integer exp_work;

always @(*) begin
    sign_y = a[31] ^ b[31];
    exp_a = a[30:23];
    exp_b = b[30:23];
    mant_a = (exp_a == 8'd0) ? {1'b0, a[22:0]} : {1'b1, a[22:0]};
    mant_b = (exp_b == 8'd0) ? {1'b0, b[22:0]} : {1'b1, b[22:0]};
    product = 48'd0;
    mant_ext = 27'd0;
    mant_round_ext = 25'd0;
    guard_bit = 1'b0;
    round_bit = 1'b0;
    sticky_bit = 1'b0;
    exp_work = 0;
    y = 32'd0;

    if (((exp_a == 8'd0) && (a[22:0] == 23'd0)) ||
        ((exp_b == 8'd0) && (b[22:0] == 23'd0))) begin
        y = 32'd0;
    end else begin
        product = mant_a * mant_b;
        exp_work = ((exp_a == 8'd0) ? 32'd1 : {24'd0, exp_a})
                 + ((exp_b == 8'd0) ? 32'd1 : {24'd0, exp_b})
                 - 127;

        if (product[47]) begin
            mant_ext = product[47:21];
            exp_work = exp_work + 1;
        end else begin
            mant_ext = product[46:20];
        end

        mant_round_ext = {1'b0, mant_ext[26:3]};
        guard_bit = mant_ext[2];
        round_bit = mant_ext[1];
        sticky_bit = mant_ext[0];

        if (guard_bit && (round_bit || sticky_bit || mant_round_ext[0])) begin
            mant_round_ext = mant_round_ext + 25'd1;
        end

        if (mant_round_ext[24]) begin
            mant_round_ext = mant_round_ext >> 1;
            exp_work = exp_work + 1;
        end

        if (exp_work >= 255) begin
            y = {sign_y, 8'hFF, 23'd0};
        end else if (exp_work <= 0) begin
            y = 32'd0;
        end else begin
            y = {sign_y, exp_work[7:0], mant_round_ext[22:0]};
        end
    end
end

endmodule
