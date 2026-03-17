module top_level_module #(
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
) (
    input wire clk,
    input wire rst_n,
    input wire in_valid,
    input wire [TOKEN_W-1:0] in_token_id,
    input wire is_prompt_token,
    output wire in_ready,
    output wire out_valid,
    output reg [TOKEN_W-1:0] next_token_id,
    output wire logits_valid,
    output reg [LOGITS_W-1:0] logits,
    output wire busy
);

localparam FSM_IDLE  = 3'd0;
localparam FSM_EMBED = 3'd1;
localparam FSM_BLOCK = 3'd2;
localparam FSM_FINAL = 3'd3;
localparam FSM_CLS   = 3'd4;
localparam FSM_DONE  = 3'd5;

reg [TOKEN_W-1:0] current_token_id;
reg [9:0] seq_pos;
reg [9:0] current_pos;
reg weights_initialized;

wire [2:0] fsm_state;
wire [2:0] fsm_layer_idx;
wire fsm_busy;
wire fsm_out_valid;
wire fsm_in_ready;
wire start_step;

assign start_step = in_valid & fsm_in_ready;
assign in_ready = fsm_in_ready;
assign busy = fsm_busy;
assign out_valid = fsm_out_valid;
assign logits_valid = fsm_out_valid;

fsm #(
    .N_LAYERS(N_LAYERS)
) u_fsm (
    .clk(clk),
    .rst_n(rst_n),
    .start(start_step),
    .state(fsm_state),
    .layer_idx(fsm_layer_idx),
    .in_ready(fsm_in_ready),
    .busy(fsm_busy),
    .out_valid(fsm_out_valid)
);

transformer_block #(
    .VOCAB_SIZE(VOCAB_SIZE),
    .TOKEN_W(TOKEN_W),
    .LOGIT_W(LOGIT_W),
    .LOGITS_W(LOGITS_W),
    .DIM(DIM),
    .HIDDEN_DIM(HIDDEN_DIM),
    .N_LAYERS(N_LAYERS),
    .N_HEADS(N_HEADS),
    .N_KV_HEADS(N_KV_HEADS),
    .MAX_SEQ_LEN(MAX_SEQ_LEN)
) u_block ();

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        current_token_id <= {TOKEN_W{1'b0}};
        seq_pos <= 10'd0;
        current_pos <= 10'd0;
        weights_initialized <= 1'b0;
        next_token_id <= {TOKEN_W{1'b0}};
        logits <= {LOGITS_W{1'b0}};
        u_block.zero_state();
    end else begin
        if (start_step) begin
            if (!weights_initialized) begin
                u_block.sync_weights();
                u_block.zero_state();
                weights_initialized <= 1'b1;
            end
            current_token_id <= in_token_id;
            current_pos <= seq_pos;
        end

        case (fsm_state)
            FSM_EMBED: begin
                u_block.load_embedding(current_token_id);
            end

            FSM_BLOCK: begin
                u_block.run_layer(fsm_layer_idx, current_pos);
            end

            FSM_FINAL: begin
                u_block.final_rmsnorm();
            end

            FSM_CLS: begin
                u_block.classify_and_argmax(next_token_id, logits);
            end

            FSM_DONE: begin
                seq_pos <= seq_pos + 10'd1;
            end

            default: begin
            end
        endcase
    end
end

endmodule
