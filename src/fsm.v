module fsm #(
    parameter N_LAYERS = 5
) (
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire embed_done,
    input wire block_done,
    input wire final_rms_done,
    input wire cls_done,
    output reg [3:0] state,
    output reg [2:0] layer_idx,
    output wire in_ready,
    output wire busy,
    output wire out_valid
);

localparam FSM_IDLE            = 4'd0;
localparam FSM_EMBED           = 4'd1;
localparam FSM_BLOCK_START     = 4'd2;
localparam FSM_BLOCK_WAIT      = 4'd3;
localparam FSM_FINAL_RMS_START = 4'd4;
localparam FSM_FINAL_RMS_WAIT  = 4'd5;
localparam FSM_CLS_START       = 4'd6;
localparam FSM_CLS_WAIT        = 4'd7;
localparam FSM_DONE            = 4'd8;

assign in_ready = (state == FSM_IDLE);
assign busy = (state != FSM_IDLE);
assign out_valid = (state == FSM_DONE);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= FSM_IDLE;
        layer_idx <= 3'd0;
    end else begin
        case (state)
            FSM_IDLE: begin
                layer_idx <= 3'd0;
                if (start) begin
                    state <= FSM_EMBED;
                end
            end

            FSM_EMBED: begin
                if (embed_done) begin
                    state <= FSM_BLOCK_START;
                    layer_idx <= 3'd0;
                end
            end

            FSM_BLOCK_START: begin
                state <= FSM_BLOCK_WAIT;
            end

            FSM_BLOCK_WAIT: begin
                if (block_done) begin
                    if (layer_idx == N_LAYERS - 1) begin
                        state <= FSM_FINAL_RMS_START;
                        layer_idx <= 3'd0;
                    end else begin
                        layer_idx <= layer_idx + 3'd1;
                        state <= FSM_BLOCK_START;
                    end
                end
            end

            FSM_FINAL_RMS_START: begin
                state <= FSM_FINAL_RMS_WAIT;
            end

            FSM_FINAL_RMS_WAIT: begin
                if (final_rms_done) begin
                    state <= FSM_CLS_START;
                end
            end

            FSM_CLS_START: begin
                state <= FSM_CLS_WAIT;
            end

            FSM_CLS_WAIT: begin
                if (cls_done) begin
                    state <= FSM_DONE;
                end
            end

            FSM_DONE: begin
                state <= FSM_IDLE;
            end

            default: begin
                state <= FSM_IDLE;
                layer_idx <= 3'd0;
            end
        endcase
    end
end

endmodule
