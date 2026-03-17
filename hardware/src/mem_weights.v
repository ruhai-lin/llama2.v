module mem_weights #(
    parameter VOCAB_SIZE = 512,
    parameter DIM = 64,
    parameter HIDDEN_DIM = 172,
    parameter N_LAYERS = 5,
    parameter N_HEADS = 8,
    parameter N_KV_HEADS = 4
) (
    input wire clk,
    input wire rst_n,
    input wire sync_start,
    output reg busy,
    output reg done
);

`include "real_fp32_helpers.vh"

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

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        busy <= 1'b0;
        done <= 1'b0;
        synced <= 1'b0;
    end else begin
        done <= 1'b0;
        if (sync_start && !synced) begin
            busy <= 1'b1;
            sync_from_bits();
            busy <= 1'b0;
            done <= 1'b1;
        end else begin
            busy <= 1'b0;
        end
    end
end

endmodule
