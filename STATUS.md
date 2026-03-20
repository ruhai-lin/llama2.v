top_level_module (src/top_level_module.v)  [CTRL/BITS + 局部 REAL]
│
├── [A] 全局控制 / 状态机选择  [CTRL/BITS]
│   │
│   ├── u_fsm : fsm.v
│   │   ├── FSM_IDLE
│   │   ├── FSM_EMBED
│   │   ├── FSM_BLOCK_START / FSM_BLOCK_WAIT
│   │   ├── FSM_FINAL_RMS_START / FSM_FINAL_RMS_WAIT
│   │   ├── FSM_CLS_START / FSM_CLS_WAIT
│   │   └── FSM_DONE
│   │
│   ├── 顶层根据 fsm_state 复用/切换 memory 端口  [CTRL/BITS]
│   │   ├── FSM_EMBED        -> local embedding 读写权重/激活
│   │   ├── FSM_BLOCK_*      -> u_block 占用 act/wgt/kv 端口
│   │   ├── FSM_FINAL_RMS_*  -> u_final_rms 占用 act/wgt 端口
│   │   └── FSM_CLS_*        -> u_cls_matmul 占用 act/wgt 端口
│   │
│   └── 当前这部分没有 real 数学计算
│
├── [B] 三个存储模块
│   │
│   ├── u_mem_weights : mem_weights.v  [CTRL/BITS]
│   │   ├── 本质：reg [31:0] mem [0:DEPTH-1]
│   │   ├── 存的是 FP32 bit pattern
│   │   ├── 读接口：wgt_rd_addr -> wgt_rd_data
│   │   └── 当前模块内部没有 real 运算
│   │
│   ├── u_mem_activation : mem_activation.v  [CTRL/BITS]
│   │   ├── 本质：reg [31:0] mem [0:DEPTH-1]
│   │   ├── 存的是中间 activation 的 FP32 bit pattern
│   │   ├── 带 tag/epoch 机制
│   │   └── 当前模块内部没有 real 运算
│   │
│   └── u_mem_kv_cache : mem_kv_cache.v  [CTRL/BITS]
│       ├── 本质：reg [31:0] mem [0:DEPTH-1]
│       ├── 存的是 KV cache 的 FP32 bit pattern
│       └── 当前模块内部没有 real 运算
│
├── [C] embedding 阶段  [MIXED / REAL]
│   │
│   ├── 触发条件：fsm_state == FSM_EMBED && weights_initialized
│   │
│   ├── 动作：
│   │   ├── embed_base_idx = current_token_id * DIM
│   │   ├── for embed_idx = 0..DIM-1
│   │   │   ├── read_wgt_local(WGT_TOKEN_EMBED_BASE + embed_base_idx + embed_idx, embed_value)
│   │   │   └── write_act_local(ACT_X_BASE + embed_idx, embed_value)
│   │   └── embed_done_reg <= 1
│   │
│   ├── 这里的真实情况：
│   │   ├── 读出来的权重 mem_wgt_rd_data 先经过 fp32_to_real(...)  [REAL]
│   │   ├── embed_value 是 real 变量  [REAL]
│   │   ├── 写回 activation 时再经过 real_to_fp32_bits(...)  [REAL]
│   │   └── 所以 embedding 现在不是纯 bit-level copy，而是 REAL 包了一层
│   │
│   └── 结论：embedding 当前仍然是 [REAL]
│
├── [D] u_block : transformer_block.v  [CTRL/BITS]
│   │
│   ├── 作用：每层先跑 attn，再跑 ffn
│   │
│   ├── 内部状态：
│   │   ├── STATE_IDLE
│   │   ├── STATE_ATTN_START / STATE_ATTN_WAIT
│   │   ├── STATE_FFN_START  / STATE_FFN_WAIT
│   │   └── STATE_DONE
│   │
│   ├── 端口仲裁：
│   │   ├── attn 阶段 -> act/wgt/kv 端口给 u_attn
│   │   └── ffn  阶段 -> act/wgt     端口给 u_ffn
│   │
│   └── transformer_block 自身没有 real 数学计算，主要是 [CTRL/BITS]
│
├── [E] Self-Attention : u_attn (src/attn.v)  [大量 REAL]
│   │
│   ├── [E.0] 总体结构
│   │   ├── 实例化：
│   │   │   ├── u_rmsnorm : kernel_rmsnorm
│   │   │   ├── u_matmul  : kernel_matmul
│   │   │   ├── u_rope    : kernel_rope
│   │   │   └── u_softmax : kernel_softmax
│   │   ├── 当前已经改成显式 FSM/controller
│   │   ├── 通过 start / busy / done 握手顺序驱动这些 kernel
│   │   └── 不再直接跨层级调用 task
│   │
│   ├── [E.1] rms_att 预归一化  [REAL]
│   │   ├── 触发：STATE_RMS_START -> 等待 u_rmsnorm.done
│   │   ├── 配置：u_rmsnorm.op_code = ATTN
│   │   ├── 输入：ACT_X_BASE
│   │   ├── 权重：WGT_RMS_ATT_BASE + layer_idx * DIM
│   │   ├── 输出：写到 ACT_XB_BASE ... ACT_XB_BASE+DIM-1
│   │   └── kernel_rmsnorm 内部是：
│   │       ├── read_act -> fp32_to_real  [REAL]
│   │       ├── sum_sq += act_value * act_value  [REAL]
│   │       ├── inv_norm = 1.0 / $sqrt(sum_sq)   [REAL]
│   │       └── write_act(real_to_fp32_bits(...)) [REAL]
│   │
│   ├── [E.2] q/k/v projection  [REAL via kernel_matmul]
│   │   ├── 触发：STATE_QKV_START -> 等待 u_matmul.done
│   │   ├── 配置：u_matmul.op_code = QKV
│   │   ├── 内部顺序：
│   │   │   ├── load_x_buffer(ACT_XB_BASE, DIM)
│   │   │   ├── matvec_tiled_from_xbuf(base_q, ACT_Q_BASE, DIM, DIM)
│   │   │   ├── matvec_tiled_from_xbuf(base_k, ACT_K_BASE, KV_DIM, DIM)
│   │   │   └── matvec_tiled_from_xbuf(base_v, ACT_V_BASE, KV_DIM, DIM)
│   │   ├── kernel_matmul 当前内部实际情况：
│   │   │   ├── x_buf / w_buf / acc_buf 是 reg [31:0] 存 bit pattern  [BITS]
│   │   │   ├── compute_tile 中：
│   │   │   │   ├── fp32_to_real(w_buf[..])
│   │   │   │   ├── fp32_to_real(x_buf[..])
│   │   │   │   ├── acc_real += w * x
│   │   │   │   └── real_to_fp32_bits(fp32_round(acc_real))
│   │   │   └── 所以真正 MAC 仍然是 REAL
│   │   └── 结论：Q/K/V 目前全都还是 [REAL]
│   │
│   ├── [E.3] RoPE 位置编码  [REAL]
│   │   ├── 触发：STATE_ROPE_START -> 等待 u_rope.done
│   │   ├── 操作对象：ACT_Q_BASE / ACT_K_BASE
│   │   ├── kernel_rope 内部：
│   │   │   ├── read_act -> fp32_to_real  [REAL]
│   │   │   ├── angle = pos_idx * freq    [REAL]
│   │   │   ├── cos_val = $cos(angle)     [REAL]
│   │   │   ├── sin_val = $sin(angle)     [REAL]
│   │   │   ├── v0*cos - v1*sin           [REAL]
│   │   │   └── 写回 real_to_fp32_bits(...) [REAL]
│   │   └── 结论：RoPE 当前完全是 [REAL]
│   │
│   ├── [E.4] KV cache 写入  [MIXED / REAL]
│   │   ├── loff = (layer_idx * MAX_SEQ_LEN + pos_idx) * KV_DIM
│   │   ├── for i = 0..KV_DIM-1
│   │   │   ├── read_act_local(ACT_K_BASE + i, kv_value)
│   │   │   ├── write_kv(KV_KEY_BASE + loff + i, kv_value)
│   │   │   ├── read_act_local(ACT_V_BASE + i, kv_value)
│   │   │   └── write_kv(KV_VALUE_BASE + loff + i, kv_value)
│   │   ├── 读 act / 写 kv 两边都要经过 real <-> bits 转换
│   │   └── 所以 KV 写入路径当前也是 [REAL]
│   │
│   ├── [E.5] attention score = Q · K^T / sqrt(d)  [REAL]
│   │   ├── inv_scale = 1.0 / $sqrt(HEAD_SIZE)  [REAL]
│   │   ├── for each head_idx
│   │   │   ├── for timestep = 0..pos_idx
│   │   │   │   ├── score = 0.0
│   │   │   │   ├── for i = 0..HEAD_SIZE-1
│   │   │   │   │   ├── read_act_local(ACT_Q_BASE + ...)
│   │   │   │   │   ├── read_kv(KV_KEY_BASE + ...)
│   │   │   │   │   └── score += act_value_a * kv_value
│   │   │   │   └── write_act_local(ACT_ATT_BASE + ..., fp32_round(score * inv_scale))
│   │   └── 这整段是显式 REAL
│   │
│   ├── [E.6] softmax 归一化  [REAL]
│   │   ├── 触发：STATE_SOFTMAX_START -> 等待 u_softmax.done
│   │   ├── kernel_softmax 内部：
│   │   │   ├── 遍历找 max_score          [REAL]
│   │   │   ├── $exp(act_value-max_score) [REAL]
│   │   │   ├── sum_exp 累加              [REAL]
│   │   │   └── act_value / sum_exp       [REAL]
│   │   └── 结论：softmax 当前完全是 [REAL]
│   │
│   ├── [E.7] attention merge = weight * V 的加权和  [REAL]
│   │   ├── 先把 ACT_XB_BASE 清零
│   │   ├── for timestep = 0..pos_idx
│   │   │   ├── for i = 0..HEAD_SIZE-1
│   │   │   │   ├── read_act_local(ACT_XB_BASE + ...)
│   │   │   │   ├── read_act_local(ACT_ATT_BASE + ...)
│   │   │   │   ├── read_kv(KV_VALUE_BASE + ...)
│   │   │   │   └── write_act_local( ..., fp32_round(act_value_a + act_value_b * kv_value) )
│   │   └── 这整段也是显式 REAL
│   │
│   ├── [E.8] o_proj 输出投影  [REAL via kernel_matmul]
│   │   ├── 触发：STATE_ATTN_OUT_START -> 等待 u_matmul.done
│   │   ├── 配置：u_matmul.op_code = ATTN_OUT
│   │   ├── 内部：
│   │   │   ├── load_x_buffer(ACT_XB_BASE, DIM)
│   │   │   ├── matvec_tiled_from_xbuf(base_o, ACT_XB2_BASE, DIM, DIM)
│   │   │   └── residual: ACT_X = ACT_X + ACT_XB2
│   │   ├── 其中：
│   │   │   ├── matvec 部分来自 kernel_matmul -> [REAL]
│   │   │   └── residual add:
│   │   │       ├── read_act(ACT_X + i, act_value)   [REAL]
│   │   │       ├── read_act(ACT_XB2 + i, xb_val)    [REAL]
│   │   │       └── write_act(ACT_X + i, fp32_round(act_value + xb_val)) [REAL]
│   │   └── 所以 o_proj + res1 当前都还是 [REAL]
│   │
│   └── [E.9] 小结
│       ├── attn 的“控制外壳”是 bits/控制
│       └── 但核心数学几乎整段都是 [REAL]
│
├── [F] FFN : u_ffn (src/ffn.v)  [大量 REAL]
│   │
│   ├── [F.1] rms_ffn 预归一化  [REAL]
│   │   ├── 触发：STATE_RMS_START -> 等待 u_rmsnorm.done
│   │   ├── 配置：u_rmsnorm.op_code = FFN
│   │   ├── 输入：ACT_X_BASE
│   │   ├── 输出：ACT_XB_BASE
│   │   └── 内部实现与 attn 前的 rmsnorm 一样 -> [REAL]
│   │
│   ├── [F.2] w1 / w3 投影  [REAL via kernel_matmul]
│   │   ├── 触发：STATE_W13_START -> 等待 u_matmul.done
│   │   ├── 配置：u_matmul.op_code = W1_W3
│   │   ├── 输出：
│   │   │   ├── ACT_HB_BASE  = W1 * XB
│   │   │   └── ACT_HB2_BASE = W3 * XB
│   │   └── MAC 内核来自 kernel_matmul -> [REAL]
│   │
│   ├── [F.3] SwiGLU / SiLU 激活  [REAL]
│   │   ├── for i = 0..HIDDEN_DIM-1
│   │   │   ├── read_act_local(ACT_HB_BASE + i, hb_val)
│   │   │   ├── read_act_local(ACT_HB2_BASE + i, hb2_val)
│   │   │   ├── val = hb_val
│   │   │   ├── val = val * (1.0 / (1.0 + $exp(-val)))   // SiLU
│   │   │   └── write_act_local(ACT_HB_BASE + i, fp32_round(val * hb2_val))
│   │   └── 这整段明确是 [REAL]
│   │
│   ├── [F.4] w2 下投影  [REAL via kernel_matmul]
│   │   ├── 触发：STATE_W2_START -> 等待 u_matmul.done
│   │   ├── 配置：u_matmul.op_code = W2
│   │   ├── 内部：
│   │   │   ├── load_x_buffer(ACT_HB_BASE, HIDDEN_DIM)
│   │   │   ├── matvec_tiled_from_xbuf(base_w2, ACT_XB_BASE, DIM, HIDDEN_DIM)
│   │   │   └── residual: ACT_X = ACT_X + ACT_XB
│   │   ├── matvec 部分来自 kernel_matmul -> [REAL]
│   │   └── residual add 也走 read_act/write_act(real) -> [REAL]
│   │
│   └── [F.5] 小结
│       └── ffn 核心数学当前也是几乎全 [REAL]
│
├── [G] final_rms : u_final_rms (kernel_rmsnorm, CONTROL_MODE=2)  [REAL]
│   │
│   ├── 触发：FSM_FINAL_RMS_START
│   ├── 调用的是 kernel_rmsnorm 的 start/done 正常接口
│   ├── op_code = FINAL
│   ├── 处理：
│   │   ├── 读最后一层的 ACT_X
│   │   ├── 读 WGT_RMS_FINAL_BASE
│   │   ├── sum_sq / sqrt / scale
│   │   └── 写回 ACT_X
│   └── 但内部数学仍然完全是 [REAL]
│
├── [H] cls_matmul : u_cls_matmul (kernel_matmul, CONTROL_MODE=4)  [REAL]
│   │
│   ├── 触发：FSM_CLS_START
│   ├── 调用方式：start/done 握手
│   ├── op_code = CLASSIFY
│   ├── 内部：
│   │   ├── load_x_buffer(ACT_X_BASE, DIM)
│   │   ├── for each vocab tile
│   │   │   ├── load_w_tile(WGT_TOKEN_EMBED_BASE, ...)
│   │   │   ├── compute_tile(tile_rows, DIM)
│   │   │   ├── 写 ACT_LOGITS_BASE
│   │   │   └── 比较 max_logit / best_token
│   │   └── 输出：
│   │       ├── flat_logits
│   │       └── next_token
│   │
│   ├── 注意：
│   │   ├── MAC 计算在 compute_tile 中 -> [REAL]
│   │   ├── max_logit 是 real 变量 -> [REAL]
│   │   └── argmax 比较本身也建立在 real act_value 上 -> [REAL]
│   │
│   └── 结论：分类 head 当前也是 [REAL]
│
└── [I] 输出阶段  [CTRL/BITS]
    │
    ├── if (cls_done)
    │   ├── next_token_id <= cls_next_token_id
    │   └── logits       <= cls_logits
    │
    ├── if (fsm_state == FSM_DONE)
    │   └── seq_pos <= seq_pos + 1
    │
    └── 这一层只是寄存器更新 / 输出打拍，没有 real 数学计算


============================================================
当前代码里“还在用 REAL”的模块总表
============================================================

[1] top_level_module.v
    └── embedding 读权重/写 activation 的本地 task 还是 real

[2] kernel_matmul.v
    ├── compute_tile 的 MAC 是 real acc_real
    ├── project_attention_output 里的 residual add 是 real
    ├── project_w2 里的 residual add 是 real
    └── classify 的 max_logit / act_value 比较是 real

[3] kernel_rmsnorm.v
    ├── sum_sq
    ├── inv_norm = 1.0 / $sqrt(...)
    └── act/wgt 都先转 real 再算

[4] kernel_softmax.v
    ├── max_score
    ├── sum_exp
    ├── $exp(...)
    └── 除法 act_value / sum_exp

[5] kernel_rope.v
    ├── $cos / $sin
    └── 旋转组合全是 real

[6] attn.v
    ├── backbone 已经改成 FSM + start/busy/done
    ├── KV 写入辅助 task 是 real
    ├── attention score Q·K^T 是 real
    ├── inv_scale = 1.0 / $sqrt(HEAD_SIZE)
    └── attn merge = weight * V + accumulate 也是 real

[7] ffn.v
    ├── backbone 已经改成 FSM + start/busy/done
    └── SiLU / SwiGLU 激活那段是 real + $exp


============================================================
如果你“先替 kernel_matmul”，会立刻影响哪些路径
============================================================

kernel_matmul.v 替换一次
│
├── attn 通过 op_code=QKV / ATTN_OUT 启动 u_matmul
├── ffn 通过 op_code=W1_W3 / W2 启动 u_matmul
└── top_level_module 通过 op_code=CLASSIFY 启动 u_cls_matmul

也就是说：
你一旦替掉 kernel_matmul，
Q/K/V、O投影、FFN上下投影、最终分类头 会一起换掉。

但以下这些地方仍然不会自动变成可综合数字实现：
│
├── kernel_rmsnorm
├── kernel_softmax
├── kernel_rope
├── attn 里手写的 score / merge / kv 辅助
├── ffn 里的 SiLU
└── top_level_module 里的 embedding 本地读写
