import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from test import MODEL_PATH, PROMPT, TOKENIZER_PATH, Tokenizer, drive_token, load_model_weights, read_model_checkpoint


PROMPT = os.getenv("PROMPT", PROMPT)
GEN_STEPS = int(os.getenv("GEN_STEPS", "10"))


@cocotb.test()
async def generate_free_run(dut):
    config, weights = read_model_checkpoint(MODEL_PATH)
    tokenizer = Tokenizer(TOKENIZER_PATH, config["vocab_size"])
    prompt_tokens = tokenizer.encode(PROMPT, bos=True, eos=False)

    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.in_token_id.value = 0
    dut.is_prompt_token.value = 0

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1

    cocotb.log.info("Loading model weights into DUT")
    await load_model_weights(dut, weights)

    cocotb.log.info("Prompt: %s", PROMPT)
    cocotb.log.info("Prompt tokens: %s", prompt_tokens)
    cocotb.log.info("Generation steps: %d", GEN_STEPS)

    next_token = None
    for token in prompt_tokens:
        next_token = await drive_token(dut, token, 1)

    generated_tokens = []
    for _ in range(GEN_STEPS):
        generated_tokens.append(next_token)
        next_token = await drive_token(dut, next_token, 0)

    prev_token = prompt_tokens[-1] if prompt_tokens else 1
    generated_text = tokenizer.decode_tokens(generated_tokens, prev_token)
    combined_text = PROMPT + generated_text
    safe_text = combined_text.encode("ascii", errors="ignore").decode("ascii")

    cocotb.log.info("Generated tokens: %s", generated_tokens)
    cocotb.log.info("Generated continuation: %s", generated_text)
    cocotb.log.info("Generated text: %s", safe_text)
