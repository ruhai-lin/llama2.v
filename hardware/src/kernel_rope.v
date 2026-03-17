module kernel_rope #(
    parameter DIM = 64,
    parameter N_HEADS = 8,
    parameter N_KV_HEADS = 4
) ();

localparam HEAD_SIZE = DIM / N_HEADS;
localparam KV_DIM = (DIM * N_KV_HEADS) / N_HEADS;

integer elem_idx;
integer rot_count;
integer vec_sel;
integer head_dim;
real freq;
real angle;
real cos_val;
real sin_val;
real v0;
real v1;

function real rope_freq;
    input integer current_head_dim;
    begin
        case (current_head_dim)
            0: rope_freq = 1.0;
            2: rope_freq = 0.1;
            4: rope_freq = 0.01;
            6: rope_freq = 0.001;
            default: rope_freq = 1.0;
        endcase
    end
endfunction

task apply;
    input integer pos_idx;
    begin
        for (elem_idx = 0; elem_idx < DIM; elem_idx = elem_idx + 2) begin
            head_dim = elem_idx % HEAD_SIZE;
            freq = rope_freq(head_dim);
            angle = pos_idx * freq;
            cos_val = $cos(angle);
            sin_val = $sin(angle);
            if (elem_idx < KV_DIM) begin
                rot_count = 2;
            end else begin
                rot_count = 1;
            end
            for (vec_sel = 0; vec_sel < rot_count; vec_sel = vec_sel + 1) begin
                if (vec_sel == 0) begin
                    v0 = top_level_module.u_block.u_mem_activation.q[elem_idx];
                    v1 = top_level_module.u_block.u_mem_activation.q[elem_idx + 1];
                    top_level_module.u_block.u_mem_activation.q[elem_idx] =
                        top_level_module.u_block.u_mem_weights.fp32_round(v0 * cos_val - v1 * sin_val);
                    top_level_module.u_block.u_mem_activation.q[elem_idx + 1] =
                        top_level_module.u_block.u_mem_weights.fp32_round(v0 * sin_val + v1 * cos_val);
                end else begin
                    v0 = top_level_module.u_block.u_mem_activation.k_vec[elem_idx];
                    v1 = top_level_module.u_block.u_mem_activation.k_vec[elem_idx + 1];
                    top_level_module.u_block.u_mem_activation.k_vec[elem_idx] =
                        top_level_module.u_block.u_mem_weights.fp32_round(v0 * cos_val - v1 * sin_val);
                    top_level_module.u_block.u_mem_activation.k_vec[elem_idx + 1] =
                        top_level_module.u_block.u_mem_weights.fp32_round(v0 * sin_val + v1 * cos_val);
                end
            end
        end
    end
endtask

endmodule
