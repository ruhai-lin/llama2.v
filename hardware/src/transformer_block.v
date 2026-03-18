module transformer_block #(
    parameter DIM = 64,
    parameter HIDDEN_DIM = 172,
    parameter N_HEADS = 8,
    parameter N_KV_HEADS = 4,
    parameter MAX_SEQ_LEN = 512
) (
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [2:0] layer_idx,
    input wire [9:0] pos_idx,
    input wire [31:0] block_input_base_addr,
    input wire [31:0] block_output_base_addr,
    output reg act_rd_en,
    output reg [31:0] act_rd_addr,
    input wire [31:0] act_rd_data,
    output reg act_wr_en,
    output reg [31:0] act_wr_addr,
    output reg [31:0] act_wr_data,
    output reg wgt_rd_en,
    output reg [31:0] wgt_rd_addr,
    input wire [31:0] wgt_rd_data,
    output reg kv_rd_en,
    output reg [31:0] kv_rd_addr,
    input wire [31:0] kv_rd_data,
    output reg kv_wr_en,
    output reg [31:0] kv_wr_addr,
    output reg [31:0] kv_wr_data,
    output wire busy,
    output reg done
);

localparam STATE_IDLE       = 3'd0;
localparam STATE_ATTN_START = 3'd1;
localparam STATE_ATTN_WAIT  = 3'd2;
localparam STATE_FFN_START  = 3'd3;
localparam STATE_FFN_WAIT   = 3'd4;
localparam STATE_DONE       = 3'd5;
localparam VOCAB_SIZE = 512;
localparam N_LAYERS = 5;

reg [2:0] state;

wire attn_start;
wire attn_busy;
wire attn_done;
wire attn_act_rd_en;
wire [31:0] attn_act_rd_addr;
wire attn_act_wr_en;
wire [31:0] attn_act_wr_addr;
wire [31:0] attn_act_wr_data;
wire attn_wgt_rd_en;
wire [31:0] attn_wgt_rd_addr;
wire [31:0] attn_wgt_rd_data;
wire attn_kv_rd_en;
wire [31:0] attn_kv_rd_addr;
wire [31:0] attn_kv_rd_data;
wire attn_kv_wr_en;
wire [31:0] attn_kv_wr_addr;
wire [31:0] attn_kv_wr_data;
wire ffn_start;
wire ffn_busy;
wire ffn_done;
wire ffn_act_rd_en;
wire [31:0] ffn_act_rd_addr;
wire ffn_act_wr_en;
wire [31:0] ffn_act_wr_addr;
wire [31:0] ffn_act_wr_data;
wire ffn_wgt_rd_en;
wire [31:0] ffn_wgt_rd_addr;
wire [31:0] ffn_wgt_rd_data;

assign attn_kv_rd_data = kv_rd_data;
assign attn_wgt_rd_data = wgt_rd_data;
assign ffn_wgt_rd_data = wgt_rd_data;

assign busy = (state != STATE_IDLE);
assign attn_start = (state == STATE_ATTN_START);
assign ffn_start = (state == STATE_FFN_START);

always @(*) begin
    act_rd_en = 1'b0;
    act_rd_addr = 32'd0;
    act_wr_en = 1'b0;
    act_wr_addr = 32'd0;
    act_wr_data = 32'd0;
    wgt_rd_en = 1'b0;
    wgt_rd_addr = 32'd0;
    kv_rd_en = 1'b0;
    kv_rd_addr = 32'd0;
    kv_wr_en = 1'b0;
    kv_wr_addr = 32'd0;
    kv_wr_data = 32'd0;

    case (state)
        STATE_ATTN_START,
        STATE_ATTN_WAIT: begin
            act_rd_en = attn_act_rd_en;
            act_rd_addr = attn_act_rd_addr;
            act_wr_en = attn_act_wr_en;
            act_wr_addr = attn_act_wr_addr;
            act_wr_data = attn_act_wr_data;
            wgt_rd_en = attn_wgt_rd_en;
            wgt_rd_addr = attn_wgt_rd_addr;
            kv_rd_en = attn_kv_rd_en;
            kv_rd_addr = attn_kv_rd_addr;
            kv_wr_en = attn_kv_wr_en;
            kv_wr_addr = attn_kv_wr_addr;
            kv_wr_data = attn_kv_wr_data;
        end

        STATE_FFN_START,
        STATE_FFN_WAIT: begin
            act_rd_en = ffn_act_rd_en;
            act_rd_addr = ffn_act_rd_addr;
            act_wr_en = ffn_act_wr_en;
            act_wr_addr = ffn_act_wr_addr;
            act_wr_data = ffn_act_wr_data;
            wgt_rd_en = ffn_wgt_rd_en;
            wgt_rd_addr = ffn_wgt_rd_addr;
        end

        default: begin
        end
    endcase
end

attn #(
    .DIM(DIM),
    .N_HEADS(N_HEADS),
    .N_KV_HEADS(N_KV_HEADS),
    .MAX_SEQ_LEN(MAX_SEQ_LEN)
) u_attn (
    .clk(clk),
    .rst_n(rst_n),
    .start(attn_start),
    .layer_idx(layer_idx),
    .pos_idx(pos_idx),
    .act_rd_en(attn_act_rd_en),
    .act_rd_addr(attn_act_rd_addr),
    .act_rd_data(act_rd_data),
    .act_wr_en(attn_act_wr_en),
    .act_wr_addr(attn_act_wr_addr),
    .act_wr_data(attn_act_wr_data),
    .wgt_rd_en(attn_wgt_rd_en),
    .wgt_rd_addr(attn_wgt_rd_addr),
    .wgt_rd_data(attn_wgt_rd_data),
    .kv_rd_en(attn_kv_rd_en),
    .kv_rd_addr(attn_kv_rd_addr),
    .kv_rd_data(attn_kv_rd_data),
    .kv_wr_en(attn_kv_wr_en),
    .kv_wr_addr(attn_kv_wr_addr),
    .kv_wr_data(attn_kv_wr_data),
    .busy(attn_busy),
    .done(attn_done)
);

ffn #(
    .DIM(DIM),
    .HIDDEN_DIM(HIDDEN_DIM),
    .N_HEADS(N_HEADS),
    .N_KV_HEADS(N_KV_HEADS)
) u_ffn (
    .clk(clk),
    .rst_n(rst_n),
    .start(ffn_start),
    .layer_idx(layer_idx),
    .act_rd_en(ffn_act_rd_en),
    .act_rd_addr(ffn_act_rd_addr),
    .act_rd_data(act_rd_data),
    .act_wr_en(ffn_act_wr_en),
    .act_wr_addr(ffn_act_wr_addr),
    .act_wr_data(ffn_act_wr_data),
    .wgt_rd_en(ffn_wgt_rd_en),
    .wgt_rd_addr(ffn_wgt_rd_addr),
    .wgt_rd_data(ffn_wgt_rd_data),
    .busy(ffn_busy),
    .done(ffn_done)
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= STATE_IDLE;
        done <= 1'b0;
    end else begin
        done <= 1'b0;
        case (state)
            STATE_IDLE: begin
                if (start) begin
                    state <= STATE_ATTN_START;
                end
            end

            STATE_ATTN_START: begin
                state <= STATE_ATTN_WAIT;
            end

            STATE_ATTN_WAIT: begin
                if (attn_done) begin
                    state <= STATE_FFN_START;
                end
            end

            STATE_FFN_START: begin
                state <= STATE_FFN_WAIT;
            end

            STATE_FFN_WAIT: begin
                if (ffn_done) begin
                    state <= STATE_DONE;
                end
            end

            STATE_DONE: begin
                done <= 1'b1;
                state <= STATE_IDLE;
            end

            default: begin
                state <= STATE_IDLE;
            end
        endcase
    end
end

endmodule
