module transformer_block #(
    parameter VOCAB_SIZE = 512,
    parameter TOKEN_W = 9,
    parameter LOGIT_W = 32,
    parameter LOGITS_W = VOCAB_SIZE * LOGIT_W,
    parameter DIM = 64,
    parameter HIDDEN_DIM = 172,
    parameter N_LAYERS = 5,
    parameter N_HEADS = 8,
    parameter N_KV_HEADS = 4,
    parameter MAX_SEQ_LEN = 512
) ();

mem_weights #(
    .VOCAB_SIZE(VOCAB_SIZE),
    .DIM(DIM),
    .HIDDEN_DIM(HIDDEN_DIM),
    .N_LAYERS(N_LAYERS),
    .N_HEADS(N_HEADS),
    .N_KV_HEADS(N_KV_HEADS)
) u_mem_weights ();

mem_activation #(
    .VOCAB_SIZE(VOCAB_SIZE),
    .DIM(DIM),
    .HIDDEN_DIM(HIDDEN_DIM),
    .N_HEADS(N_HEADS),
    .MAX_SEQ_LEN(MAX_SEQ_LEN),
    .N_KV_HEADS(N_KV_HEADS)
) u_mem_activation ();

mem_kv_cache #(
    .DIM(DIM),
    .N_LAYERS(N_LAYERS),
    .N_HEADS(N_HEADS),
    .N_KV_HEADS(N_KV_HEADS),
    .MAX_SEQ_LEN(MAX_SEQ_LEN)
) u_mem_kv_cache ();

attn #(
    .DIM(DIM),
    .N_HEADS(N_HEADS),
    .N_KV_HEADS(N_KV_HEADS),
    .MAX_SEQ_LEN(MAX_SEQ_LEN)
) u_attn ();

ffn #(
    .DIM(DIM),
    .HIDDEN_DIM(HIDDEN_DIM),
    .N_HEADS(N_HEADS),
    .N_KV_HEADS(N_KV_HEADS)
) u_ffn ();

kernel_rmsnorm #(
    .DIM(DIM)
) u_final_rms ();

kernel_matmul #(
    .VOCAB_SIZE(VOCAB_SIZE),
    .TOKEN_W(TOKEN_W),
    .LOGIT_W(LOGIT_W),
    .LOGITS_W(LOGITS_W),
    .DIM(DIM),
    .HIDDEN_DIM(HIDDEN_DIM),
    .N_LAYERS(N_LAYERS),
    .N_HEADS(N_HEADS),
    .N_KV_HEADS(N_KV_HEADS)
) u_cls_matmul ();

integer i;
reg initialized;

task sync_weights;
    begin
        if (!u_mem_weights.synced) begin
            u_mem_weights.sync_from_bits();
        end
        initialized = 1'b1;
    end
endtask

task zero_state;
    begin
        u_mem_activation.zero_state();
        u_mem_kv_cache.zero_cache();
    end
endtask

task load_embedding;
    input integer token_id;
    integer base_idx;
    begin
        base_idx = token_id * DIM;
        for (i = 0; i < DIM; i = i + 1) begin
            u_mem_activation.x[i] = u_mem_weights.token_embedding[base_idx + i];
        end
    end
endtask

task run_layer;
    input integer layer_idx;
    input integer pos_idx;
    begin
        u_attn.run(layer_idx, pos_idx);
        u_ffn.run(layer_idx);
    end
endtask

task final_rmsnorm;
    begin
        u_final_rms.apply_final();
    end
endtask

task classify_and_argmax;
    output [TOKEN_W-1:0] next_token;
    output [LOGITS_W-1:0] flat_logits;
    begin
        u_cls_matmul.classify(next_token, flat_logits);
    end
endtask

initial begin
    initialized = 1'b0;
end

endmodule
