# 开发进度记录（CLI + TOML）

本文件用于记录“VideoCaptioner 在 Linux 服务器无 GUI 场景下使用 CLI 运行”的实现进度与关键决策，便于后续继续迭代与排查。

## 已完成（当前工作区）

### 1) 纯命令行入口（Headless CLI）

- 新增：`app/cli.py`
- 特点：
  - 不依赖 GUI/Qt（避免 `main.py`/`app/thread/*` 的 PyQt 信号与窗口）
  - 使用 TOML 配置（Python 3.11+ 自带 `tomllib`）
  - 支持子命令：
    - `transcribe`：只转录
    - `subtitle`：断句/优化/翻译
    - `synthesize`：ffmpeg 合成字幕视频
    - `full`：全流程（转录 → 字幕处理 → 合成）
  - 支持 `--profile`（同一份 config 内切换环境配置）
  - 默认输出结构：`paths.work_dir/<输入stem>/...`，并提供 `--run-dir` 覆盖
  - 批量输入支持：
    - 重复 `--input`
    - `--glob` 通配符
    - `--input-dir`（可选 `--recursive`）
    - `--continue-on-error` 失败不中断（可选）
  - 转录分块参数可配置（`[transcribe]`）：
    - `chunk_length_sec` / `chunk_overlap_sec` / `chunk_concurrency`
    - FasterWhisper(cuda) 默认更保守（建议 `chunk_concurrency=1`）
  - 支持 `--no-cache`：本次运行禁用 ASR/翻译/LLM 缓存（便于调参后强制重跑）

### 2) TOML 配置示例

- 新增：`config.example.toml`
- 包含：
  - `[paths]`：工作目录
  - `[transcribe]` + `[transcribe.faster_whisper]`：CUDA FasterWhisper 配置
  - `[subtitle]`：断句/优化/翻译配置（含字幕样式）
  - `[llm]`：OpenAI 兼容 API 配置（用于断句/优化/LLM 翻译）
  - `[synthesis]`：合成配置
  - `[profiles.<name>]`：示例 profile 写法

### 3) FasterWhisper 可执行文件查找增强（服务器关键）

- 修改：`app/core/asr/faster_whisper.py`
- 目的：
  - 支持在配置中直接指定 `faster-whisper-xxl` 的绝对路径（不必依赖 PATH）
  - 兼容 “cpu 使用 faster-whisper 时不支持 vad_method” 的行为（自动清空）

### 4) translate 包懒加载

- 修改：`app/core/translate/__init__.py`
- 目的：
  - 避免 CLI 仅做 `--help` 或轻量导入时触发大量依赖导入链
  - 减少在缺依赖环境下的“无关失败”

### 5) 文档更新

- 修改：`README.md`
- 新增 CLI 章节，说明如何在 Linux/服务器环境运行与 FasterWhisper(cuda) 注意事项。

## 使用方式（快速验证）

1) 复制并修改配置：
- `config.example.toml`

2) 运行：
- 全流程：`python3 -m app.cli full --config config.example.toml --input /path/video.mp4`
- 仅转录：`python3 -m app.cli transcribe --config config.example.toml --input /path/video.mp4`
- 仅字幕处理：`python3 -m app.cli subtitle --config config.example.toml --input /path/raw.srt --video /path/video.mp4`
- 仅合成：`python3 -m app.cli synthesize --config config.example.toml --video /path/video.mp4 --subtitle /path/subtitle.ass`
- 批量（示例）：
  - `python3 -m app.cli transcribe --config config.example.toml --input-dir /data/in --recursive --continue-on-error`
  - `python3 -m app.cli full --config config.example.toml --glob '/data/in/*.mp4' --output-dir /data/out --continue-on-error`

## 运行依赖与注意事项

- Python：建议 `>=3.11`（需要 `tomllib` 读取 TOML）
- 系统工具：需要 `ffmpeg`
- CUDA FasterWhisper：
  - 需要 `faster-whisper-xxl` 可执行文件可用
  - 可选两种方式：
    1) 放入 `PATH`
    2) 在 `config.example.toml` 里设置 `[transcribe.faster_whisper] program="/abs/path/faster-whisper-xxl"`
- LLM（断句/优化/LLM 翻译时需要）：
  - 配置 `[llm] base_url / api_key / model`
  - CLI 会设置环境变量：`OPENAI_BASE_URL`、`OPENAI_API_KEY`

## 下一步（建议 TODO）

- 批处理：
  - 支持输入目录/通配符（多视频批量跑）
  - 输出目录策略与失败重试策略
- 多音轨支持：
  - 将 `audio_track_index` 暴露为 CLI 参数或配置项
- 更强的配置校验与 `config check` 子命令：
  - 检查 `ffmpeg`、`faster-whisper-xxl`、模型目录、LLM 连通性等
- Docker/容器化（可选）：
  - 在现有 CLI 基础上提供一个最小 Dockerfile/运行示例（带 GPU 的版本需要 nvidia runtime）
