module ffn #(
    parameter DIM = 64,
    parameter HIDDEN_DIM = 172,
    parameter N_HEADS = 8,
    parameter N_KV_HEADS = 4
) ();

kernel_rmsnorm #(
    .DIM(DIM)
) u_rmsnorm ();

kernel_matmul #(
    .DIM(DIM),
    .HIDDEN_DIM(HIDDEN_DIM),
    .N_HEADS(N_HEADS),
    .N_KV_HEADS(N_KV_HEADS)
) u_matmul ();

integer i;
real val;

task run;
    input integer layer_idx;
    begin
        u_rmsnorm.apply_ffn(layer_idx);
        u_matmul.project_w1_w3(layer_idx);

        for (i = 0; i < HIDDEN_DIM; i = i + 1) begin
            val = top_level_module.u_block.u_mem_activation.hb[i];
            val = val * (1.0 / (1.0 + $exp(-val)));
            top_level_module.u_block.u_mem_activation.hb[i] =
                top_level_module.u_block.u_mem_weights.fp32_round(
                    val * top_level_module.u_block.u_mem_activation.hb2[i]
                );
        end

        u_matmul.project_w2(layer_idx);
    end
endtask

endmodule
