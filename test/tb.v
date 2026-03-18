module tb;

reg clk;
reg rst_n;
reg in_valid;
reg [8:0] in_token_id;
reg is_prompt_token;
wire in_ready;
wire out_valid;
wire [8:0] next_token_id;
wire logits_valid;
wire [(512*32)-1:0] logits;
wire busy;

reg [8:0] prompt_tokens [0:4];
reg [8:0] feedback_token;
integer idx;

top_level_module dut (
    .clk(clk),
    .rst_n(rst_n),
    .in_valid(in_valid),
    .in_token_id(in_token_id),
    .is_prompt_token(is_prompt_token),
    .in_ready(in_ready),
    .out_valid(out_valid),
    .next_token_id(next_token_id),
    .logits_valid(logits_valid),
    .logits(logits),
    .busy(busy)
);

task drive_token;
    input [8:0] token_id;
    input prompt_flag;
    begin
        @(posedge clk);
        while (!in_ready) begin
            @(posedge clk);
        end
        in_valid <= 1'b1;
        in_token_id <= token_id;
        is_prompt_token <= prompt_flag;
        @(posedge clk);
        in_valid <= 1'b0;
        while (!out_valid) begin
            @(posedge clk);
        end
    end
endtask

always #5 clk = ~clk;

initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    in_valid = 1'b0;
    in_token_id = 9'd0;
    is_prompt_token = 1'b0;

    prompt_tokens[0] = 9'd1;
    prompt_tokens[1] = 9'd403;
    prompt_tokens[2] = 9'd407;
    prompt_tokens[3] = 9'd261;
    prompt_tokens[4] = 9'd378;

    repeat (3) @(posedge clk);
    rst_n = 1'b1;

    for (idx = 0; idx < 5; idx = idx + 1) begin
        drive_token(prompt_tokens[idx], 1'b1);
        $display("prompt step=%0d next_token_id=%0d", idx, next_token_id);
    end

    feedback_token = next_token_id;
    for (idx = 0; idx < 10; idx = idx + 1) begin
        drive_token(feedback_token, 1'b0);
        $display("gen step=%0d next_token_id=%0d", idx, feedback_token);
        feedback_token = next_token_id;
    end

    $finish;
end

endmodule
