module mem_kv_cache #(
    parameter DIM = 64,
    parameter N_LAYERS = 5,
    parameter N_HEADS = 8,
    parameter N_KV_HEADS = 4,
    parameter MAX_SEQ_LEN = 512
) ();

localparam KV_DIM = (DIM * N_KV_HEADS) / N_HEADS;

real key_cache [0:N_LAYERS*MAX_SEQ_LEN*KV_DIM-1];
real value_cache [0:N_LAYERS*MAX_SEQ_LEN*KV_DIM-1];

integer idx;

task zero_cache;
    begin
        for (idx = 0; idx < N_LAYERS * MAX_SEQ_LEN * KV_DIM; idx = idx + 1) begin
            key_cache[idx] = 0.0;
            value_cache[idx] = 0.0;
        end
    end
endtask

endmodule
