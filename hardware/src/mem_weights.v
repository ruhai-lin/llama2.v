module mem_weights #(
    parameter VOCAB_SIZE = 512,
    parameter DIM = 64,
    parameter HIDDEN_DIM = 172,
    parameter N_LAYERS = 5,
    parameter N_HEADS = 8,
    parameter N_KV_HEADS = 4
) ();

localparam HEAD_SIZE = DIM / N_HEADS;
localparam KV_DIM = (DIM * N_KV_HEADS) / N_HEADS;

reg [31:0] token_embedding_bits [0:VOCAB_SIZE*DIM-1];
reg [31:0] rms_att_bits [0:N_LAYERS*DIM-1];
reg [31:0] wq_bits [0:N_LAYERS*DIM*DIM-1];
reg [31:0] wk_bits [0:N_LAYERS*DIM*KV_DIM-1];
reg [31:0] wv_bits [0:N_LAYERS*DIM*KV_DIM-1];
reg [31:0] wo_bits [0:N_LAYERS*DIM*DIM-1];
reg [31:0] rms_ffn_bits [0:N_LAYERS*DIM-1];
reg [31:0] w1_bits [0:N_LAYERS*DIM*HIDDEN_DIM-1];
reg [31:0] w2_bits [0:N_LAYERS*HIDDEN_DIM*DIM-1];
reg [31:0] w3_bits [0:N_LAYERS*DIM*HIDDEN_DIM-1];
reg [31:0] rms_final_bits [0:DIM-1];

real token_embedding [0:VOCAB_SIZE*DIM-1];
real rms_att_weight [0:N_LAYERS*DIM-1];
real wq [0:N_LAYERS*DIM*DIM-1];
real wk [0:N_LAYERS*DIM*KV_DIM-1];
real wv [0:N_LAYERS*DIM*KV_DIM-1];
real wo [0:N_LAYERS*DIM*DIM-1];
real rms_ffn_weight [0:N_LAYERS*DIM-1];
real w1 [0:N_LAYERS*DIM*HIDDEN_DIM-1];
real w2 [0:N_LAYERS*HIDDEN_DIM*DIM-1];
real w3 [0:N_LAYERS*DIM*HIDDEN_DIM-1];
real rms_final_weight [0:DIM-1];

reg synced;
integer idx;

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

task sync_from_bits;
    begin
        for (idx = 0; idx < VOCAB_SIZE * DIM; idx = idx + 1) begin
            token_embedding[idx] = fp32_to_real(token_embedding_bits[idx]);
        end
        for (idx = 0; idx < N_LAYERS * DIM; idx = idx + 1) begin
            rms_att_weight[idx] = fp32_to_real(rms_att_bits[idx]);
            rms_ffn_weight[idx] = fp32_to_real(rms_ffn_bits[idx]);
        end
        for (idx = 0; idx < N_LAYERS * DIM * DIM; idx = idx + 1) begin
            wq[idx] = fp32_to_real(wq_bits[idx]);
            wo[idx] = fp32_to_real(wo_bits[idx]);
        end
        for (idx = 0; idx < N_LAYERS * DIM * KV_DIM; idx = idx + 1) begin
            wk[idx] = fp32_to_real(wk_bits[idx]);
            wv[idx] = fp32_to_real(wv_bits[idx]);
        end
        for (idx = 0; idx < N_LAYERS * DIM * HIDDEN_DIM; idx = idx + 1) begin
            w1[idx] = fp32_to_real(w1_bits[idx]);
            w3[idx] = fp32_to_real(w3_bits[idx]);
        end
        for (idx = 0; idx < N_LAYERS * HIDDEN_DIM * DIM; idx = idx + 1) begin
            w2[idx] = fp32_to_real(w2_bits[idx]);
        end
        for (idx = 0; idx < DIM; idx = idx + 1) begin
            rms_final_weight[idx] = fp32_to_real(rms_final_bits[idx]);
        end
        synced = 1'b1;
    end
endtask

initial begin
    synced = 1'b0;
end

endmodule
