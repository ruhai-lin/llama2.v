module kernel_softmax #(
    parameter N_HEADS = 8,
    parameter MAX_SEQ_LEN = 512
) ();

integer timestep;
real max_score;
real sum_exp;

task normalize_head;
    input integer head_idx;
    input integer pos_idx;
    integer att_base;
    begin
        att_base = head_idx * MAX_SEQ_LEN;
        max_score = top_level_module.u_block.u_mem_activation.att[att_base];
        for (timestep = 1; timestep <= pos_idx; timestep = timestep + 1) begin
            if (top_level_module.u_block.u_mem_activation.att[att_base + timestep] > max_score) begin
                max_score = top_level_module.u_block.u_mem_activation.att[att_base + timestep];
            end
        end

        sum_exp = 0.0;
        for (timestep = 0; timestep <= pos_idx; timestep = timestep + 1) begin
            top_level_module.u_block.u_mem_activation.att[att_base + timestep] =
                top_level_module.u_block.u_mem_weights.fp32_round(
                    $exp(top_level_module.u_block.u_mem_activation.att[att_base + timestep] - max_score)
                );
            sum_exp = sum_exp + top_level_module.u_block.u_mem_activation.att[att_base + timestep];
        end

        for (timestep = 0; timestep <= pos_idx; timestep = timestep + 1) begin
            top_level_module.u_block.u_mem_activation.att[att_base + timestep] =
                top_level_module.u_block.u_mem_weights.fp32_round(
                    top_level_module.u_block.u_mem_activation.att[att_base + timestep] / sum_exp
                );
        end
    end
endtask

endmodule
