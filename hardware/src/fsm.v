module fsm #(
    parameter N_LAYERS = 5
) (
    input wire clk,
    input wire rst_n,
    input wire start,
    output reg [2:0] state,
    output reg [2:0] layer_idx,
    output wire in_ready,
    output wire busy,
    output wire out_valid
);

localparam FSM_IDLE  = 3'd0;
localparam FSM_EMBED = 3'd1;
localparam FSM_BLOCK = 3'd2;
localparam FSM_FINAL = 3'd3;
localparam FSM_CLS   = 3'd4;
localparam FSM_DONE  = 3'd5;

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
                state <= FSM_BLOCK;
                layer_idx <= 3'd0;
            end

            FSM_BLOCK: begin
                if (layer_idx == N_LAYERS - 1) begin
                    state <= FSM_FINAL;
                    layer_idx <= 3'd0;
                end else begin
                    layer_idx <= layer_idx + 3'd1;
                end
            end

            FSM_FINAL: begin
                state <= FSM_CLS;
            end

            FSM_CLS: begin
                state <= FSM_DONE;
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
