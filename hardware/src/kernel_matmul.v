module kernel_matmul #(
    parameter VOCAB_SIZE = 512,
    parameter TOKEN_W = 9,
    parameter LOGIT_W = 32,
    parameter LOGITS_W = VOCAB_SIZE * LOGIT_W,
    parameter DIM = 64,
    parameter HIDDEN_DIM = 172,
    parameter N_LAYERS = 5,
    parameter N_HEADS = 8,
    parameter N_KV_HEADS = 4
) ();

localparam KV_DIM = (DIM * N_KV_HEADS) / N_HEADS;

integer i;
integer j;
integer base_idx;
real acc;
real max_logit;
integer best_token;

task project_qkv;
    input integer layer_idx;
    integer base_q;
    integer base_k;
    integer base_v;
    begin
        base_q = layer_idx * DIM * DIM;
        base_k = layer_idx * DIM * KV_DIM;
        base_v = layer_idx * DIM * KV_DIM;

        for (i = 0; i < DIM; i = i + 1) begin
            acc = 0.0;
            for (j = 0; j < DIM; j = j + 1) begin
                acc = acc + (top_level_module.u_block.u_mem_weights.wq[base_q + i * DIM + j]
                    * top_level_module.u_block.u_mem_activation.xb[j]);
            end
            top_level_module.u_block.u_mem_activation.q[i] =
                top_level_module.u_block.u_mem_weights.fp32_round(acc);
        end

        for (i = 0; i < KV_DIM; i = i + 1) begin
            acc = 0.0;
            for (j = 0; j < DIM; j = j + 1) begin
                acc = acc + (top_level_module.u_block.u_mem_weights.wk[base_k + i * DIM + j]
                    * top_level_module.u_block.u_mem_activation.xb[j]);
            end
            top_level_module.u_block.u_mem_activation.k_vec[i] =
                top_level_module.u_block.u_mem_weights.fp32_round(acc);
        end

        for (i = 0; i < KV_DIM; i = i + 1) begin
            acc = 0.0;
            for (j = 0; j < DIM; j = j + 1) begin
                acc = acc + (top_level_module.u_block.u_mem_weights.wv[base_v + i * DIM + j]
                    * top_level_module.u_block.u_mem_activation.xb[j]);
            end
            top_level_module.u_block.u_mem_activation.v_vec[i] =
                top_level_module.u_block.u_mem_weights.fp32_round(acc);
        end
    end
endtask

task project_attention_output;
    input integer layer_idx;
    integer base_o;
    begin
        base_o = layer_idx * DIM * DIM;
        for (i = 0; i < DIM; i = i + 1) begin
            acc = 0.0;
            for (j = 0; j < DIM; j = j + 1) begin
                acc = acc + (top_level_module.u_block.u_mem_weights.wo[base_o + i * DIM + j]
                    * top_level_module.u_block.u_mem_activation.xb[j]);
            end
            top_level_module.u_block.u_mem_activation.xb2[i] =
                top_level_module.u_block.u_mem_weights.fp32_round(acc);
        end

        for (i = 0; i < DIM; i = i + 1) begin
            top_level_module.u_block.u_mem_activation.x[i] =
                top_level_module.u_block.u_mem_weights.fp32_round(
                    top_level_module.u_block.u_mem_activation.x[i]
                    + top_level_module.u_block.u_mem_activation.xb2[i]
                );
        end
    end
endtask

task project_w1_w3;
    input integer layer_idx;
    integer base_w1;
    integer base_w3;
    begin
        base_w1 = layer_idx * DIM * HIDDEN_DIM;
        base_w3 = layer_idx * DIM * HIDDEN_DIM;
        for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
            acc = 0.0;
            for (j = 0; j < DIM; j = j + 1) begin
                acc = acc + (top_level_module.u_block.u_mem_weights.w1[base_w1 + i * DIM + j]
                    * top_level_module.u_block.u_mem_activation.xb[j]);
            end
            top_level_module.u_block.u_mem_activation.hb[i] =
                top_level_module.u_block.u_mem_weights.fp32_round(acc);

            acc = 0.0;
            for (j = 0; j < DIM; j = j + 1) begin
                acc = acc + (top_level_module.u_block.u_mem_weights.w3[base_w3 + i * DIM + j]
                    * top_level_module.u_block.u_mem_activation.xb[j]);
            end
            top_level_module.u_block.u_mem_activation.hb2[i] =
                top_level_module.u_block.u_mem_weights.fp32_round(acc);
        end
    end
endtask

task project_w2;
    input integer layer_idx;
    integer base_w2;
    begin
        base_w2 = layer_idx * HIDDEN_DIM * DIM;
        for (i = 0; i < DIM; i = i + 1) begin
            acc = 0.0;
            for (j = 0; j < HIDDEN_DIM; j = j + 1) begin
                acc = acc + (top_level_module.u_block.u_mem_weights.w2[base_w2 + i * HIDDEN_DIM + j]
                    * top_level_module.u_block.u_mem_activation.hb[j]);
            end
            top_level_module.u_block.u_mem_activation.xb[i] =
                top_level_module.u_block.u_mem_weights.fp32_round(acc);
        end

        for (i = 0; i < DIM; i = i + 1) begin
            top_level_module.u_block.u_mem_activation.x[i] =
                top_level_module.u_block.u_mem_weights.fp32_round(
                    top_level_module.u_block.u_mem_activation.x[i]
                    + top_level_module.u_block.u_mem_activation.xb[i]
                );
        end
    end
endtask

task classify;
    output [TOKEN_W-1:0] next_token;
    output [LOGITS_W-1:0] flat_logits;
    begin
        best_token = 0;
        max_logit = 0.0;
        for (i = 0; i < VOCAB_SIZE; i = i + 1) begin
            acc = 0.0;
            base_idx = i * DIM;
            for (j = 0; j < DIM; j = j + 1) begin
                acc = acc + (top_level_module.u_block.u_mem_weights.token_embedding[base_idx + j]
                    * top_level_module.u_block.u_mem_activation.x[j]);
            end
            top_level_module.u_block.u_mem_activation.logits_real[i] =
                top_level_module.u_block.u_mem_weights.fp32_round(acc);
            flat_logits[i * LOGIT_W +: LOGIT_W] =
                top_level_module.u_block.u_mem_weights.real_to_fp32_bits(
                    top_level_module.u_block.u_mem_activation.logits_real[i]
                );
            if ((i == 0) || (top_level_module.u_block.u_mem_activation.logits_real[i] > max_logit)) begin
                max_logit = top_level_module.u_block.u_mem_activation.logits_real[i];
                best_token = i;
            end
        end
        next_token = best_token[TOKEN_W-1:0];
    end
endtask

endmodule
