module attn #(
    parameter DIM = 64,
    parameter N_HEADS = 8,
    parameter N_KV_HEADS = 4,
    parameter MAX_SEQ_LEN = 512
) ();

localparam HEAD_SIZE = DIM / N_HEADS;
localparam KV_DIM = (DIM * N_KV_HEADS) / N_HEADS;
localparam KV_MUL = N_HEADS / N_KV_HEADS;

kernel_rmsnorm #(
    .DIM(DIM)
) u_rmsnorm ();

kernel_matmul #(
    .DIM(DIM),
    .N_HEADS(N_HEADS),
    .N_KV_HEADS(N_KV_HEADS)
) u_matmul ();

kernel_rope #(
    .DIM(DIM),
    .N_HEADS(N_HEADS),
    .N_KV_HEADS(N_KV_HEADS)
) u_rope ();

kernel_softmax #(
    .N_HEADS(N_HEADS),
    .MAX_SEQ_LEN(MAX_SEQ_LEN)
) u_softmax ();

integer i;
integer head_idx;
integer timestep;
integer head_base;
integer kv_head;
integer cache_base;
integer att_base;
integer loff;
real score;
real inv_scale;

task run;
    input integer layer_idx;
    input integer pos_idx;
    begin
        u_rmsnorm.apply_attn(layer_idx);
        u_matmul.project_qkv(layer_idx);
        u_rope.apply(pos_idx);

        loff = (layer_idx * MAX_SEQ_LEN + pos_idx) * KV_DIM;
        for (i = 0; i < KV_DIM; i = i + 1) begin
            top_level_module.u_block.u_mem_kv_cache.key_cache[loff + i] =
                top_level_module.u_block.u_mem_activation.k_vec[i];
            top_level_module.u_block.u_mem_kv_cache.value_cache[loff + i] =
                top_level_module.u_block.u_mem_activation.v_vec[i];
        end

        inv_scale = 1.0 / $sqrt(HEAD_SIZE);
        for (i = 0; i < DIM; i = i + 1) begin
            top_level_module.u_block.u_mem_activation.xb[i] = 0.0;
        end

        loff = layer_idx * MAX_SEQ_LEN * KV_DIM;
        for (head_idx = 0; head_idx < N_HEADS; head_idx = head_idx + 1) begin
            head_base = head_idx * HEAD_SIZE;
            att_base = head_idx * MAX_SEQ_LEN;
            kv_head = head_idx / KV_MUL;

            for (timestep = 0; timestep <= pos_idx; timestep = timestep + 1) begin
                cache_base = loff + timestep * KV_DIM + kv_head * HEAD_SIZE;
                score = 0.0;
                for (i = 0; i < HEAD_SIZE; i = i + 1) begin
                    score = score + (
                        top_level_module.u_block.u_mem_activation.q[head_base + i]
                        * top_level_module.u_block.u_mem_kv_cache.key_cache[cache_base + i]
                    );
                end
                top_level_module.u_block.u_mem_activation.att[att_base + timestep] =
                    top_level_module.u_block.u_mem_weights.fp32_round(score * inv_scale);
            end

            u_softmax.normalize_head(head_idx, pos_idx);

            for (i = 0; i < HEAD_SIZE; i = i + 1) begin
                top_level_module.u_block.u_mem_activation.xb[head_base + i] = 0.0;
            end
            for (timestep = 0; timestep <= pos_idx; timestep = timestep + 1) begin
                cache_base = loff + timestep * KV_DIM + kv_head * HEAD_SIZE;
                for (i = 0; i < HEAD_SIZE; i = i + 1) begin
                    top_level_module.u_block.u_mem_activation.xb[head_base + i] =
                        top_level_module.u_block.u_mem_weights.fp32_round(
                            top_level_module.u_block.u_mem_activation.xb[head_base + i]
                            + (
                                top_level_module.u_block.u_mem_activation.att[att_base + timestep]
                                * top_level_module.u_block.u_mem_kv_cache.value_cache[cache_base + i]
                            )
                        );
                end
            end
        end

        u_matmul.project_attention_output(layer_idx);
    end
endtask

endmodule
