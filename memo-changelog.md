# Speaches Realtime API 服务端修改备忘录

统一记录每次服务端修改的内容、原因和预期效果，供外部测试方参考。

---

## 2026-07-20 第 6 轮：修复模型加载耗时导致客户端超时断开 + 转录任务被取消

### 根因

1. **模型加载 20s 导致客户端超时断开**：第一个分块转录时需要加载 faster-whisper 模型到 GPU，耗时约 20 秒。在此期间没有发送任何事件给客户端，OpenAI Realtime API 客户端默认等待约 5 秒后断开 WebSocket。
2. **WS 断开导致 handler 任务被取消**：客户端断开后，`asyncio.TaskGroup` 取消内部所有任务，`commit_and_transcribe()` 中 `await transcriber.task` 抛出 `CancelledError`，即使转录已完成也不会发 completed 事件。

### 修复

1. **模型预加载（`realtime_ws.py` + `input_audio_buffer.py`）**：session 创建后立即在后台用 1 秒静音音频调用转录 executor，让 faster-whisper 预加载到内存。之后第一个真实分块转录时模型已就绪。
   - 新增 `preload_transcription_model()` 函数
   - `realtime_ws.py` 中 session 创建后 `asyncio.create_task(asyncio.to_thread(preload_transcription_model, ...))`

2. **`asyncio.shield()` 防止任务取消（`input_audio_buffer_event_router.py` + `input_audio_buffer.py`）**：
   - `commit_and_transcribe()` 中 `await asyncio.shield(transcriber.task)` 保护 handler 任务
   - `_handler()` 中 `await asyncio.shield(loop.run_in_executor(...))` 保护分块转录
   - 即使客户端断开、TaskGroup 被取消，转录任务仍会继续执行完成

3. **心跳 delta（`input_audio_buffer.py`）**：`_handler()` 开始立即发送一条空 delta 事件 `{"delta": ""}`，让客户端知道服务端已开始处理转录，避免因无事件而超时断开。

4. **`prefix_padding_ms` 回归修复（`session_event_router.py` + `types/realtime.py`）**：第 6 轮部署时不小心带入了之前未提交的修改，导致 `prefix_padding_ms` 被从 exclude 集移除且移除了 warning 检查。恢复为：日志 warning 级别提示（不向客户端发送 error）+ `turn_detection.prefix_padding_ms` 从 session update 中排除（静默忽略，回退到默认值 300）。

### 变更文件

- `src/speaches/realtime/input_audio_buffer.py` — `preload_transcription_model()` 函数；`_handler()` 加心跳 delta + `asyncio.shield()`
- `src/speaches/realtime/input_audio_buffer_event_router.py` — `await asyncio.shield(transcriber.task)`
- `src/speaches/routers/realtime_ws.py` — session 创建后后台预加载模型
- `src/speaches/realtime/session_event_router.py` — 恢复 prefix_padding_ms 的 warning（非 error）+ 恢复 exclude
- `src/speaches/types/realtime.py` — `TurnDetection.prefix_padding_ms` 恢复为 `int | NotGiven = NOT_GIVEN`

### 注意事项

- 预加载会增加约 20s 的 session 创建时间（后台任务，不阻塞客户端）
- 如果客户端在预加载完成前就发送音频，转录仍会等待模型就绪（正常排队）
- 心跳 delta 的 `delta` 为空字符串，客户端可以忽略或选择展示"正在识别..."状态
- `prefix_padding_ms` 客户端可以任意发送，服务端静默忽略（日志 warning 级别记录），不会返回 error 事件

---

## 2026-07-20 第 5 轮：修复 Bug 7 分块转录全失败

### 根因

`_transcribe_raw_sync()` 构造 `TranscriptionRequest` 时传了 `speech_segments=None` 和 `vad_options=None`，但 Pydantic 模型要求这两个字段必填，所有 63 个分块全部验证失败，返回空字符串。结果 `delta` 和 `completed` 的 `transcript` 都是 `""`。

### 修复

`src/speaches/realtime/input_audio_buffer.py` — `_transcribe_raw_sync()` 中改为：

```python
speech_segments=[],                  # 空列表 = 转写全部音频，不分段
vad_options=DEFAULT_VAD_OPTIONS,     # 默认 VAD 参数（不会被用到，因为 speech_segments 为空）
```

`merge_segments([])` 返回 `[]`，faster-whisper 接收空 `clip_timestamps` 会转写整段输入音频，行为正确。

### 重叠从 1.5s 改为 0.5s

应测试方建议缩小重叠，减少块数量（40s 音频从 ~27 块降到 ~14 块），降低重复文字量。

---

## 2026-07-20 第 4 轮：Bug 7 delta 事件实现（分块转录）

### 实现方案

将 VAD 检测到的整段语音音频，按 **3 秒一个块（含 0.5 秒重叠）** 切分，逐块独立转录。每完成一块就通过 `conversation.item.input_audio_transcription.delta` 事件推送给客户端，最后发 `completed` 事件。

- **块大小 3 秒**：faster-whisper 转录约 1-2 秒完成，延迟可接受
- **重叠 0.5 秒**：覆盖跨边界音节（典型音节 200-400ms）
- **无前缀去重**：当前版本重叠部分的文字会重复输出在各 delta 中（如 "hello world" 和 "world this is" 中的 "world"）。后续可考虑前缀匹配去重优化

### 关于 `prefix_padding_ms` 与重叠去重

对方提到的思路分析：

- **`prefix_padding_ms`** 是 VAD 阶段的概念：检测到语音起点时，在起点**之前**多保留若干毫秒音频，避免切掉语音开头
- **重叠**是分块阶段的概念：固定窗口滑动时，每个窗口在**之后**多包含若干毫秒音频，避免跨边界文字丢失
- **前缀匹配去重**：后一块转录完成后，将重叠部分的文字与前一块末尾做最长前缀匹配，去掉重复部分再输出 delta。这是一个独立的优化，与 `prefix_padding_ms` 无关

当前版本暂不实现前缀去重，0.5s 重叠已足够小，重复量有限。

### 事件序列

```
input_audio_buffer.speech_started    ← 开始说话
input_audio_buffer.speech_stopped    ← VAD 检测沉默
input_audio_buffer.committed         ← 提交音频
                                     ← 开始分块转录
conversation.item.input_audio_transcription.delta  (delta="第一段")
conversation.item.input_audio_transcription.delta  (delta="第二段")
...                                  ← 每 1-2 秒出一个块的结果
conversation.item.input_audio_transcription.completed  (transcript="完整文字")
conversation.item.added              (transcript="完整文字")
conversation.item.done               (transcript="完整文字")
```

### 变更文件

- `src/speaches/types/realtime.py` — 新增 `ConversationItemInputAudioTranscriptionDeltaEvent` 类型，注册到 `ConversationServerEvent` 联合类型和 `SERVER_EVENT_TYPES` 集合
- `src/speaches/realtime/input_audio_buffer.py` — 新增 `_transcribe_raw_sync()`（无 VAD 的直接转录函数），`_handler()` 改为分块转录 + 每块发 delta 事件

### 预期效果

1. 录音结束后 **1-2 秒内** 客户端就能收到第一段 delta 事件（之前要等整段 40 秒音频全部转完，耗时 10-20 秒）
2. 后续每 1-2 秒持续收到新的 delta，用户体验接近"边说边出字"
3. `conversation.item.added` / `done` 仍带完整 transcript（Bug 6 保留）
4. 3 秒块大小 + 1.5 秒重叠：短句整句出现，重叠避免切词导致前半句丢失

### 已知局限

- 分块转录是 speech_stopped 后才开始的（不是录音期间实时出字）。要真正的"边说边出字"需要重构成 VAD append 阶段的流式转录，复杂度高，后续评估是否值得做。
- 每块的转录独立运行，不跨块利用上下文，对超长语音可能有轻微精度损失。
- 块转录跳过 VAD 预处理（直接传 raw chunk），因为 `data_w_vad_applied` 已经过滤过。

---

## 2026-07-20 第 3 轮：Bug 6 修复 + Bug 8 验证通过

### Bug 6：转录结果无法到达客户端 — ✅ 已修复

**根因**：之前 `_handler()` 先创建 conversation item（transcript=null）再异步转录。等转录完成时，客户端可能已经断开 WebSocket，completed 事件发不出去。

**修复**：颠倒流程——先等转录完成，再创建 conversation item，`conversation.item.added` / `done` 发出时就带实际文字。

**变更**：`src/speaches/realtime/input_audio_buffer.py` — `_handler()` 方法重构

### Bug 8：session.update VAD 参数不生效 — ✅ 经测试方验证已修

### Bug 5：Unexpected streaming response — ✅ 已随 Bug 3 修复

Bug 3 已将 realtime 转录改为直接调用 executor（`_transcribe_sync`），不再经过 HTTP 路由 `transcription_response_to_http_response`。那行 `logger.error` 对 realtime 流程无影响。

---

## 2026-07-20 第 2 轮：Bug 4/6 修复 + prefix_padding_ms 修正

### Bug 6：转录结果无法到达客户端 — ✅ 已修复

（与第 3 轮相同，已合并到第 3 轮记录）

### Bug 4：prefix_padding_ms 错误 — ✅ 已修复

**根因**：`TurnDetection.prefix_padding_ms` 无默认值，客户端必须显式传入，否则 400 错误。且 `session.update` 中此字段被静默丢弃（exclude）。

**修复**：
- 默认值设为 `300`（和 OpenAI 一致），客户端不再需要显式携带
- 客户端通过 `session.update` 发送的 `prefix_padding_ms` 会实际生效
- 取值范围：任意非负整数（毫秒），建议 0~1000

**变更**：`src/speaches/types/realtime.py` — `TurnDetection.prefix_padding_ms: int = 300`；`src/speaches/realtime/session_event_router.py` — 移除 `prefix_padding_ms` 的 exclude 和 error check

### Bug 5：Unexpected streaming response — ✅ 已修复

（已合并到第 3 轮记录）

## 2026-07-20 第 1 轮：Bug 1-4 基础修复

### Bug 1：`file.seek(0)` — ✅

`input_audio_buffer.py:154` 写入音频后缺少 `file.seek(0)`，导致后续读取为空。

### Bug 2：`data_w_vad_applied` assert — ✅

`input_audio_buffer.py:81-88` 中 `assert audio_end_ms` 在 VAD 未检测到结束时报错，改为 `audio_end = len(self.data)` 兜底。

### Bug 3：`transcription_client.create()` 卡死 — ✅

ASGITransport 路径死锁问题。`_handler()` 绕过 HTTP/ASGI，直接调用 `executor.handle_transcription_request()`。

### Bug 4：`prefix_padding_ms` spurious error — ✅

（已合并到第 2 轮记录）

---

## 2026-07-21 第 7 轮：信号量 + CPU 优化 + Handler 取消修复 + VAD 强制切段

### 本轮修改

#### 1. 信号量限制并发 Transcription（`input_audio_buffer.py`）

新增 `_MAX_CONCURRENT_TRANSCRIPTIONS = 2` + `asyncio.Semaphore(2)`，限制同时运行的 handler 不超过 2 个，避免 CPU 过载。

#### 2. `prefix_padding_ms` 回归修复（`session_event_router.py` + `types/realtime.py`）

第 6 轮部署时误将 `prefix_padding_ms` 改为报 error，恢复为：warning 日志提示 + 从 session.update 中排除（静默忽略）。

#### 3. CPU 线程优化（`compose.yaml`）

针对 N100 4 核处理器配置：
- `WHISPER__CPU_THREADS=2` — 每推理使用 2 核
- `WHISPER__NUM_WORKERS=2` — 2 个推理并行
- `OMP_NUM_THREADS=2` — BLAS 库也遵循
- 效果：chunk 处理时间从 80-100s 降到 **7-10s**

#### 4. Handler 取消链路修复（`input_audio_buffer.py` + `input_audio_buffer_event_router.py`）

**根因**：两层 `asyncio.shield` 阻止了 CancelledError 传播：
- 外层：`commit_and_transcribe()` 中 `await asyncio.shield(transcriber.task)`
- 内层：`_handler()` 中 `await asyncio.shield(loop.run_in_executor(...))`
- 加上 `except CancelledError` 在 while 循环里 catch 后继续循环

导致 WebSocket 断连后 handler 不释放信号量，后续 handler 排队等 104s。

**修复**：
- 去掉外层 shield：`await transcriber.task`（不再 shield）
- 去掉内层 shield：`loop.run_in_executor(...)`（不再 shield）
- 去掉 `except CancelledError`：让 CancelledError 正常传播，handler 退出、释放信号量

#### 5. 连续说话强制切段（`input_audio_buffer_event_router.py`）

**根因**：`vad_detection_flow` 只看 3s 音频窗口判断语音起止，没有累计时长检查。连续说话不停时 VAD 从不触发 `speech_stopped`，音频无限累积。第一个 handler 可能拿 30-40s 音频，处理半天，堵住后续 handler。

**修复**：新增 `MAX_SPEECH_DURATION_MS = 30000`，在 `vad_detection_flow` 中检查 `duration_ms - audio_start_ms > 30s` 时强制触发 `speech_stopped`，切段提交。

#### 6. WebSocket 断连方向日志（`message_manager.py`）

新增 `[CLOSE-DIRECTION]` 标记日志，明确记录：
- Receiver 先退出（客户端/代理主动发 Close 帧）
- Sender 先退出（服务端发送时连接已断）

### 变更文件

| 文件 | 改动 |
|------|------|
| `src/speaches/realtime/input_audio_buffer.py` | 新增 `_MAX_CONCURRENT_TRANSCRIPTIONS=2` 信号量；去掉内层 `asyncio.shield`；去掉 `except CancelledError` |
| `src/speaches/realtime/input_audio_buffer_event_router.py` | 去掉外层 `asyncio.shield(transcriber.task)`；新增 `MAX_SPEECH_DURATION_MS=30000` 强制切段 |
| `src/speaches/realtime/message_manager.py` | 新增 `[CLOSE-DIRECTION]` 日志 |
| `src/speaches/realtime/session_event_router.py` | prefix_padding_ms 回归修复 |
| `src/speaches/types/realtime.py` | TurnDetection.prefix_padding_ms 恢复为 `int \| NotGiven = NOT_GIVEN` |
| `compose.yaml` | 新增 `WHISPER__CPU_THREADS=2`、`WHISPER__NUM_WORKERS=2`、`OMP_NUM_THREADS=2` |

### 待验证

- [ ] 实时转录文字能持续跟上（不再卡在第一个 handler）
- [ ] 长录音不会断连
- [ ] 客户端能看到持续的 delta 事件输出

---

## 2026-07-21 第 8 轮：空闲断开 + 并发模型访问 + 块步长 bug 修复与性能调优

### 本轮修改

#### 1. 空闲断开修复 — keep-alive 心跳（`input_audio_buffer.py`）

**根因**：chunk 转录耗时 12-16s，期间无任何事件发送给客户端，OpenAI Realtime SDK 约 5s 后断开 WebSocket。

**修复**：`_handler()` 内新增 `_keep_alive()` 协程，每 2s 发送一条空 `ConversationItemInputAudioTranscriptionDeltaEvent(delta="")`，防止连接空闲超时。

#### 2. 并发 ONNX 模型访问崩溃修复（`input_audio_buffer.py`）

**根因**：模型预加载（`preload_transcription_model`）与第一个 handler 的 `_transcribe_raw_sync` 同时调用 `faster-whisper`，竞争同一个 ONNX 会话导致崩溃。

**修复**：
- `_MAX_CONCURRENT_TRANSCRIPTIONS = 2 → 1`，保证同一时刻只有一个 handler 持有 ONNX 会话
- keep-alive 移至 `async with _transcription_semaphore:` 块内，避免空转阻塞信号量

#### 3. 模型预加载方式变更（`input_audio_buffer.py` + `routers/realtime_ws.py` + `lifespan.py` 新增）

**问题**：之前的 `preload_transcription_model` 用 1s 静音音频调用完整转录来"预热"模型，耗时 10-20s 且与第一个 handler 竞争 ONNX 会话。

**修复**：
- `preload_transcription_model()` 改为只加载模型到内存（`model_manager.load_model()` + `with model_wrapper: pass`），不做推理
- 删除 `realtime_ws.py` 中的每连接预加载
- 新增 `lifespan.py` 服务器启动预加载（通过 `PRELOAD_MODELS` 环境变量）

#### 4. 块转录步长 bug 修复（`input_audio_buffer.py`）

**根因**：`chunk_start += overlap` 中 `overlap = 8000`（0.5s），变量名暗示是重叠量但实际用作步长，导致 3s 块的滑窗步长只有 0.5s。13.6s 音频生成 26 个块，处理耗时约 390s。

**修复**：
- `CHUNK_DURATION_SAMPLES = 16000 * 3 → 72000`（3s → 4.5s 块）
- `hop_size = chunk_size - 16000 // 2`（步长 = 块大小 - 0.5s 重叠 = 4s）
- 13.6s 音频从 26 块降到 4 块，耗时从 ~390s 降到 ~90s

#### 5. chunk 超时放宽（`input_audio_buffer.py`）

`wait_for` 超时从 60s 放宽到 `300.0`，避免 CPU 慢速时因长块超时返回空串。

#### 6. 转录文本日志（`input_audio_buffer.py`）

`logger.info(f"Transcription done in {elapsed:.2f}s, transcript='{transcript}'")` — 记录完整转录文本而非仅长度。

### 变更文件

| 文件 | 改动 |
|------|------|
| `src/speaches/realtime/input_audio_buffer.py` | 新增 `_keep_alive()`；信号量改为 1；预加载函数改为纯加载；块步长 bug 修复；chunk 超时 60→300s；4.5s 块；完整文本日志 |
| `src/speaches/routers/realtime_ws.py` | 删除每连接模型预加载 |
| `src/speaches/lifespan.py` | 新增（服务器启动时通过 PRELOAD_MODELS 预加载模型） |
| `src/speaches/executors/whisper.py` | `_transcribe_raw_sync` → `handle_non_streaming_transcription_request`（支持 preload 兼容） |
| `compose.yaml` | `PRELOAD_MODELS` 环境变量 |

### 已知问题

- **CPU 推理慢**：`Systran/faster-whisper-small` 模型在 N100 CPU 上约 5x 实时（4.5s 音频需 ~20s 推理）。可换 `tiny`/`base` 模型优化，或启用 GPU。
- **samples_dropped**：客户端 WebSocket 发送线程消费 ring buffer 跟不上麦克风硬件写入，属客户端侧优化项，不影响功能。
- **块顺序依赖**：当前串行处理（信号量=1），一次只能处理一个 utterance 的块。多段语音排队时长 = sum(各段块数 × 推理耗时)。
