module mem_activation #(
    parameter VOCAB_SIZE = 512,
    parameter DIM = 64,
    parameter HIDDEN_DIM = 172,
    parameter N_HEADS = 8,
    parameter MAX_SEQ_LEN = 512,
    parameter N_KV_HEADS = 4
) ();

localparam KV_DIM = (DIM * N_KV_HEADS) / N_HEADS;

real x [0:DIM-1];
real xb [0:DIM-1];
real xb2 [0:DIM-1];
real hb [0:HIDDEN_DIM-1];
real hb2 [0:HIDDEN_DIM-1];
real q [0:DIM-1];
real k_vec [0:KV_DIM-1];
real v_vec [0:KV_DIM-1];
real att [0:N_HEADS*MAX_SEQ_LEN-1];
real logits_real [0:VOCAB_SIZE-1];

integer idx;

task zero_state;
    begin
        for (idx = 0; idx < DIM; idx = idx + 1) begin
            x[idx] = 0.0;
            xb[idx] = 0.0;
            xb2[idx] = 0.0;
            q[idx] = 0.0;
        end
        for (idx = 0; idx < HIDDEN_DIM; idx = idx + 1) begin
            hb[idx] = 0.0;
            hb2[idx] = 0.0;
        end
        for (idx = 0; idx < KV_DIM; idx = idx + 1) begin
            k_vec[idx] = 0.0;
            v_vec[idx] = 0.0;
        end
        for (idx = 0; idx < N_HEADS * MAX_SEQ_LEN; idx = idx + 1) begin
            att[idx] = 0.0;
        end
        for (idx = 0; idx < VOCAB_SIZE; idx = idx + 1) begin
            logits_real[idx] = 0.0;
        end
    end
endtask

endmodule
