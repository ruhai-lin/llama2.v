import bisect
import os
import struct

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "model", "stories260K.bin"))
TOKENIZER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "model", "tok512.bin"))
PROMPT = "Once upon a time"

class Tokenizer:
    def __init__(self, path, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = []
        self.vocab_scores = []
        self.sorted_vocab = None
        self.byte_pieces = [bytes([i]).decode("latin1") for i in range(256)]

        with open(path, "rb") as f:
            self.max_token_length = struct.unpack("i", f.read(4))[0]
            for _ in range(vocab_size):
                self.vocab_scores.append(struct.unpack("f", f.read(4))[0])
                token_len = struct.unpack("i", f.read(4))[0]
                self.vocab.append(f.read(token_len).decode("utf-8", errors="ignore"))

        self.sorted_vocab = sorted((token, idx) for idx, token in enumerate(self.vocab))
        self.sorted_keys = [token for token, _ in self.sorted_vocab]

    def str_lookup(self, piece):
        idx = bisect.bisect_left(self.sorted_keys, piece)
        if idx < len(self.sorted_keys) and self.sorted_keys[idx] == piece:
            return self.sorted_vocab[idx][1]
        return -1

    def encode(self, text, bos=True, eos=False):
        tokens = []
        if bos:
            tokens.append(1)
        if text:
            tokens.append(self.str_lookup(" "))

        raw = text.encode("utf-8")
        i = 0
        while i < len(raw):
            buf = bytearray()
            while True:
                current = raw[i]
                if current & 0xC0 != 0x80:
                    buf = bytearray()
                buf.append(current)
                i += 1
                if not (i < len(raw) and (raw[i] & 0xC0) == 0x80 and len(buf) < 4):
                    break

            piece = buf.decode("utf-8", errors="ignore")
            token_id = self.str_lookup(piece)
            if token_id != -1:
                tokens.append(token_id)
            else:
                for value in buf:
                    tokens.append(value + 3)

        while True:
            best_score = float("-inf")
            best_token_id = None
            best_index = None
            for idx in range(len(tokens) - 1):
                merged = self.vocab[tokens[idx]] + self.vocab[tokens[idx + 1]]
                merged_id = self.str_lookup(merged)
                if merged_id != -1 and self.vocab_scores[merged_id] > best_score:
                    best_score = self.vocab_scores[merged_id]
                    best_token_id = merged_id
                    best_index = idx
            if best_index is None:
                break
            tokens = tokens[:best_index] + [best_token_id] + tokens[best_index + 2 :]

        if eos:
            tokens.append(2)
        return tokens

    def decode_piece(self, prev_token, token):
        piece = self.vocab[token]
        if prev_token == 1 and piece.startswith(" "):
            piece = piece[1:]
        if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
            byte_value = int(piece[3:5], 16)
            return self.byte_pieces[byte_value]
        return piece

    def decode_tokens(self, tokens, prev_token):
        out = ""
        current_prev = prev_token
        for token in tokens:
            out += self.decode_piece(current_prev, token)
            current_prev = token
        return out


def read_model_checkpoint(path):
    with open(path, "rb") as f:
        header = struct.unpack("7i", f.read(28))
        payload = f.read()

    config = {
        "dim": header[0],
        "hidden_dim": header[1],
        "n_layers": header[2],
        "n_heads": header[3],
        "n_kv_heads": header[4],
        "vocab_size_raw": header[5],
        "vocab_size": abs(header[5]),
        "seq_len": header[6],
        "shared_weights": header[5] > 0,
    }

    words = list(struct.unpack("<{}I".format(len(payload) // 4), payload))
    dim = config["dim"]
    hidden_dim = config["hidden_dim"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    n_kv_heads = config["n_kv_heads"]
    vocab_size = config["vocab_size"]
    seq_len = config["seq_len"]
    head_size = dim // n_heads
    kv_dim = (dim * n_kv_heads) // n_heads

    ptr = 0

    def take(size):
        nonlocal ptr
        out = words[ptr : ptr + size]
        ptr += size
        return out

    weights = {
        "token_embedding": take(vocab_size * dim),
        "rms_att": take(n_layers * dim),
        "wq": take(n_layers * dim * dim),
        "wk": take(n_layers * dim * kv_dim),
        "wv": take(n_layers * dim * kv_dim),
        "wo": take(n_layers * dim * dim),
        "rms_ffn": take(n_layers * dim),
        "w1": take(n_layers * dim * hidden_dim),
        "w2": take(n_layers * hidden_dim * dim),
        "w3": take(n_layers * dim * hidden_dim),
        "rms_final": take(dim),
    }

    ptr += seq_len * head_size // 2
    ptr += seq_len * head_size // 2

    if not config["shared_weights"]:
        weights["wcls"] = take(vocab_size * dim)
    else:
        weights["wcls"] = weights["token_embedding"]

    return config, weights


async def write_memory_array(memory_handle, values, chunk_size=2048):
    for idx, value in enumerate(values):
        memory_handle[idx].value = value
        if idx and (idx % chunk_size) == 0:
            await Timer(1, units="ps")


async def load_model_weights(dut, weights):
    mem = dut.u_mem_weights
    flat_weights = (
        weights["token_embedding"]
        + weights["rms_att"]
        + weights["wq"]
        + weights["wk"]
        + weights["wv"]
        + weights["wo"]
        + weights["rms_ffn"]
        + weights["w1"]
        + weights["w2"]
        + weights["w3"]
        + weights["rms_final"]
    )
    await write_memory_array(mem.mem, flat_weights)
    mem.synced.value = 0
    await Timer(1, units="ps")


async def drive_token(dut, token_id, is_prompt_token):
    while True:
        await RisingEdge(dut.clk)
        if int(dut.in_ready.value) == 1:
            dut.in_valid.value = 1
            dut.in_token_id.value = token_id
            dut.is_prompt_token.value = is_prompt_token
            break

    await RisingEdge(dut.clk)
    dut.in_valid.value = 0

    while int(dut.out_valid.value) == 0:
        await RisingEdge(dut.clk)

    return int(dut.next_token_id.value)


@cocotb.test()
async def test_stage1_smoke(dut):
    config, weights = read_model_checkpoint(MODEL_PATH)
    assert config["dim"] == 64
    assert config["hidden_dim"] == 172
    assert config["n_layers"] == 5
    assert config["n_heads"] == 8
    assert config["n_kv_heads"] == 4
    assert config["vocab_size"] == 512
    assert config["seq_len"] == 512
    assert config["shared_weights"]

    tokenizer = Tokenizer(TOKENIZER_PATH, config["vocab_size"])
    prompt_tokens = tokenizer.encode(PROMPT, bos=True, eos=False)
    assert prompt_tokens == [1, 403, 407, 261, 378]

    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.in_token_id.value = 0
    dut.is_prompt_token.value = 0

    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1

    cocotb.log.info("Loading model weights into DUT")
    await load_model_weights(dut, weights)

    next_token = None
    for token in prompt_tokens:
        next_token = await drive_token(dut, token, 1)

    generated_tokens = []
    for _ in range(3):
        generated_tokens.append(next_token)
        next_token = await drive_token(dut, next_token, 0)

    generated_text = tokenizer.decode_tokens(generated_tokens, prompt_tokens[-1])
    combined_text = PROMPT + generated_text
    safe_text = combined_text.encode("ascii", errors="ignore").decode("ascii")

    cocotb.log.info("Generated tokens: %s", generated_tokens)
    cocotb.log.info("Generated text: %s", safe_text)

    assert len(set(generated_tokens)) >= 3
    assert sum(ch.isalpha() for ch in safe_text) >= 8
