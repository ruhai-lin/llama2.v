module kernel_rmsnorm #(
    parameter DIM = 64
) ();

integer i;
real sum_sq;
real inv_norm;

task apply_attn;
    input integer layer_idx;
    integer base_idx;
    begin
        sum_sq = 0.0;
        for (i = 0; i < DIM; i = i + 1) begin
            sum_sq = sum_sq + (top_level_module.u_block.u_mem_activation.x[i] * top_level_module.u_block.u_mem_activation.x[i]);
        end
        sum_sq = (sum_sq / DIM) + 1e-5;
        inv_norm = 1.0 / $sqrt(sum_sq);
        base_idx = layer_idx * DIM;
        for (i = 0; i < DIM; i = i + 1) begin
            top_level_module.u_block.u_mem_activation.xb[i] =
                top_level_module.u_block.u_mem_weights.fp32_round(
                    top_level_module.u_block.u_mem_weights.rms_att_weight[base_idx + i]
                    * (inv_norm * top_level_module.u_block.u_mem_activation.x[i])
                );
        end
    end
endtask

task apply_ffn;
    input integer layer_idx;
    integer base_idx;
    begin
        sum_sq = 0.0;
        for (i = 0; i < DIM; i = i + 1) begin
            sum_sq = sum_sq + (top_level_module.u_block.u_mem_activation.x[i] * top_level_module.u_block.u_mem_activation.x[i]);
        end
        sum_sq = (sum_sq / DIM) + 1e-5;
        inv_norm = 1.0 / $sqrt(sum_sq);
        base_idx = layer_idx * DIM;
        for (i = 0; i < DIM; i = i + 1) begin
            top_level_module.u_block.u_mem_activation.xb[i] =
                top_level_module.u_block.u_mem_weights.fp32_round(
                    top_level_module.u_block.u_mem_weights.rms_ffn_weight[base_idx + i]
                    * (inv_norm * top_level_module.u_block.u_mem_activation.x[i])
                );
        end
    end
endtask

task apply_final;
    begin
        sum_sq = 0.0;
        for (i = 0; i < DIM; i = i + 1) begin
            sum_sq = sum_sq + (top_level_module.u_block.u_mem_activation.x[i] * top_level_module.u_block.u_mem_activation.x[i]);
        end
        sum_sq = (sum_sq / DIM) + 1e-5;
        inv_norm = 1.0 / $sqrt(sum_sq);
        for (i = 0; i < DIM; i = i + 1) begin
            top_level_module.u_block.u_mem_activation.x[i] =
                top_level_module.u_block.u_mem_weights.fp32_round(
                    top_level_module.u_block.u_mem_weights.rms_final_weight[i]
                    * (inv_norm * top_level_module.u_block.u_mem_activation.x[i])
                );
        end
    end
endtask

endmodule
