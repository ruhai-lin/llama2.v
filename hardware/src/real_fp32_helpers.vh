function real pow2_real;
    input integer exp;
    integer k;
    real value;
    begin
        value = 1.0;
        if (exp >= 0) begin
            for (k = 0; k < exp; k = k + 1) begin
                value = value * 2.0;
            end
        end else begin
            for (k = 0; k < -exp; k = k + 1) begin
                value = value / 2.0;
            end
        end
        pow2_real = value;
    end
endfunction

function real fp32_to_real;
    input [31:0] bits;
    integer exponent;
    integer bit_idx;
    real frac;
    real scale;
    begin
        exponent = bits[30:23];
        if ((exponent == 0) && (bits[22:0] == 0)) begin
            fp32_to_real = 0.0;
        end else begin
            if (exponent == 0) begin
                frac = 0.0;
                for (bit_idx = 0; bit_idx < 23; bit_idx = bit_idx + 1) begin
                    if (bits[22-bit_idx]) begin
                        frac = frac + pow2_real(-(bit_idx + 1));
                    end
                end
                scale = pow2_real(-126);
            end else begin
                frac = 1.0;
                for (bit_idx = 0; bit_idx < 23; bit_idx = bit_idx + 1) begin
                    if (bits[22-bit_idx]) begin
                        frac = frac + pow2_real(-(bit_idx + 1));
                    end
                end
                scale = pow2_real(exponent - 127);
            end

            if (bits[31]) begin
                fp32_to_real = -frac * scale;
            end else begin
                fp32_to_real = frac * scale;
            end
        end
    end
endfunction

function [31:0] real_to_fp32_bits;
    input real value;
    integer sign_bit;
    integer exponent;
    integer biased_exponent;
    integer mantissa;
    real abs_value;
    real norm_value;
    real mantissa_real;
    begin
        if (value == 0.0) begin
            real_to_fp32_bits = 32'd0;
        end else begin
            if (value < 0.0) begin
                sign_bit = 1;
                abs_value = -value;
            end else begin
                sign_bit = 0;
                abs_value = value;
            end

            exponent = 0;
            norm_value = abs_value;
            while (norm_value >= 2.0) begin
                norm_value = norm_value / 2.0;
                exponent = exponent + 1;
            end
            while (norm_value < 1.0) begin
                norm_value = norm_value * 2.0;
                exponent = exponent - 1;
            end

            biased_exponent = exponent + 127;
            if (biased_exponent <= 0) begin
                real_to_fp32_bits = 32'd0;
            end else if (biased_exponent >= 255) begin
                real_to_fp32_bits = {sign_bit[0], 8'hFE, 23'h7FFFFF};
            end else begin
                mantissa_real = (norm_value - 1.0) * 8388608.0;
                mantissa = mantissa_real + 0.5;
                if (mantissa >= 8388608) begin
                    mantissa = 0;
                    biased_exponent = biased_exponent + 1;
                end
                if (biased_exponent >= 255) begin
                    real_to_fp32_bits = {sign_bit[0], 8'hFE, 23'h7FFFFF};
                end else begin
                    real_to_fp32_bits = {sign_bit[0], biased_exponent[7:0], mantissa[22:0]};
                end
            end
        end
    end
endfunction

function real fp32_round;
    input real value;
    begin
        fp32_round = fp32_to_real(real_to_fp32_bits(value));
    end
endfunction
