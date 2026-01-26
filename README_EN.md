# ComfyUI HAIGC Qwen3TTS

ComfyUI custom nodes integrating Qwen3-TTS for voice design, voice clone, and custom voice generation.

## Author

- WeChat: HAIGC1994

## Original Open-Source Project

[https://github.com/QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)

## Features

- 🎤 **Voice Design**: Generate custom voices from text prompts
- 🎭 **Voice Clone**: Clone a voice from reference audio and output a role preset
- 🎨 **Custom Voice**: Use preset speakers or custom prompts
- 🧩 **Role Presets**: Save, load, and batch input role presets
- 🗣️ **Multi-speaker Dialogue**: Role mapping and auto .pt loading
- 🧰 **Prompt Splitter**: Six prompt inputs with six outputs
- 🌍 **Multi-language**: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
- ⚡ **GPU/CPU**: CUDA acceleration and CPU mode
- 🎯 **Precision**: FP16 and FP32

## Installation

### 1. Dependencies

```bash
pip install torch torchaudio transformers librosa soundfile accelerate
```

### 2. Model Download

Download models from:
[https://huggingface.co/collections/Qwen/qwen3-tts](https://huggingface.co/collections/Qwen/qwen3-tts)

Local model path:
`ComfyUI\models\qwen-tts`

```
ComfyUI/models/qwen-tts/{model_folder_name}/
```

Supported models:
- `Qwen3-TTS-12Hz-0.6B-Base`
- `Qwen3-TTS-12Hz-0.6B-CustomVoice`
- `Qwen3-TTS-12Hz-1.7B-Base`
- `Qwen3-TTS-12Hz-1.7B-CustomVoice`
- `Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- `Qwen3-TTS-Tokenizer-12Hz`

Folder naming rule:
- Use the model name suffix, e.g. `Qwen3-TTS-12Hz-1.7B-VoiceDesign`

Example:
```
ComfyUI/
└── models/
    └── qwen-tts/
        ├── Qwen3-TTS-12Hz-1.7B-VoiceDesign/
        ├── Qwen3-TTS-12Hz-1.7B-CustomVoice/
        └── Qwen3-TTS-12Hz-1.7B-Base/
```

## Nodes

### 1. Qwen3 TTS Model Loader

Load a Qwen3-TTS model.

Inputs:
- Model Name
- Device: cuda / cpu / auto
- Precision: fp16 / fp32

Output:
- Model

Notes:
- Models must exist under `ComfyUI/models/qwen-tts/`
- Online downloads are disabled

### 2. Qwen3 TTS Voice Design

Generate speech from a prompt-based voice description.

Inputs:
- Model
- Text
- Prompt
- Language (optional)
- Auto unload model (optional)
- Max new tokens (optional)
- `随机种子` (Seed) (optional)
- `生成后控制` (Post-generate control) (optional): randomize / fixed / increment / decrement

### 3. Qwen3 TTS Voice Clone

Clone a voice from reference audio.

Inputs:
- Model
- Reference audio
- Text
- Reference text (optional)
- Language (optional)
- Auto unload model (optional)
- Max new tokens (optional)
- `随机种子` (Seed) (optional)
- `生成后控制` (Post-generate control) (optional): randomize / fixed / increment / decrement

Outputs:
- Audio
- Role Preset

### 4. Qwen3 TTS Custom Voice

Use preset speakers or custom prompts.

Inputs:
- Model
- Text
- Preset speaker
- Language (optional)
- Prompt (optional, overrides preset prompt)
- Auto unload model (optional)
- Max new tokens (optional)
- `随机种子` (Seed) (optional)

### 5. Qwen3 TTS Role Preset Save

Save a role preset to a .pt file named by role name.

Inputs:
- Role Preset
- Role Name
- Save Directory (optional)

Outputs:
- File Path
- Role Preset (includes role name)

### 6. Qwen3 TTS Role Preset Select

Load a .pt role preset and extract role name (from file content or file name).

Inputs:
- Preset File
- Manual File Name (optional)

Outputs:
- Role Preset (includes role name)

### 7. Qwen3 TTS Role Preset Input

Batch 6 role presets at once. Chain nodes for more.

Inputs:
- Existing Role Presets (optional)
- Role Preset 1-6

Outputs:
- Role Presets

### 8. Qwen3 TTS Dialogue Synthesis

Generate multi-speaker dialogue from lines like “Role: Text”.

Inputs:
- Model
- Dialogue Text
- Role Presets (optional)
- Role Mapping (optional): one per line, `role=file`
- Language (optional)
- `Enable Pause Control` (optional): Enable/disable pause handling (default: True)
- Auto unload model (optional)
- Max new tokens (optional)
- `随机种子` (Seed) (optional)

**Pause Control:**
- Insert `=Ns` in the text to add silence (N is seconds).
- Example: `Role: Hello, =1s I am Role. =2.5s Nice to meet you.`
- Note: Pause tag must be `=` followed by number and `s`.

Output:
- Audio

### 9. Qwen3 TTS Prompts

Six prompt inputs with six outputs.

Inputs:
- Prompt 1-6

Outputs:
- Prompt 1-6

### 10. Qwen3 TTS Voice Description

Construct voice description prompts with infinite chaining support.

Inputs:
- `Base Voice`: Select age and gender (e.g., Loli, Shota, Young Lady, Uncle)
- `Texture 1-3`: Select voice texture (e.g., Sweet, Husky, Magnetic, Lazy)
- `Previous Prompt` (optional): Chain from previous node
- `Custom Description` (optional): Manual text input

Outputs:
- `Prompt`: Combined prompt string

## Notes

1. Use the correct model type for each node.
2. Enable auto-unload to reduce VRAM usage if needed.
3. Audio outputs can be saved with ComfyUI audio save nodes.

## Troubleshooting

### Model Not Found

- Confirm the model is under `ComfyUI/models/qwen-tts/`
- Check the folder name matches the model suffix

### Unsupported Feature

- Voice Design needs VoiceDesign models
- Voice Clone needs Base models
- Custom Voice needs CustomVoice models

## License

See the original project license.

## Changelog

### v1.6.0
- Optimized parameter names to use Chinese (e.g., `随机种子` for Seed)
- Added `生成后控制` (Post-generate control) to Voice Clone and Voice Design nodes (supports Chinese options)
- Optimized Voice Clone parameter order
- Fixed some runtime errors

### v1.5.0
- Added silence/pause insertion support in Dialogue Synthesis (syntax: `=2s`)
- Added pause control toggle

### v1.4.0
- Added "Qwen3 TTS Voice Description" node for visual prompt construction
- Includes rich options for age, gender, and texture

### v1.3.0
- Added Dialogue Synthesis and Role Preset Input nodes
- Role Preset Save uses role name as filename and outputs role name info
- Voice Clone outputs role presets
- Added six-prompt node

### v1.2.0
- Added seed and post-generate control options
- Improved custom prompt priority

### v1.1.0
- Renamed internal package to `_qwen_tts_haigc`
- Fixed transformers 4.57.1 compatibility
- Disabled online model download, local only
- Improved model path handling

### v1.0.0
- Initial release
- Voice design, voice clone, and custom voice

## Links

- Workflow demo: [https://www.runninghub.cn/post/2014536001888198657/inviteCode=rh-v1127](https://www.runninghub.cn/post/2014536001888198657/inviteCode=rh-v1127)
- Recommended ComfyUI cloud: [https://www.runninghub.cn/user-center/1887871050510716930/webapp?inviteCode=rh-v1127](https://www.runninghub.cn/user-center/1887871050510716930/webapp?inviteCode=rh-v1127)
- Invite code: rh-v1127 (1000 RH coins)
- Resource download: [https://pan.quark.cn/s/a56c5a6ec9c2](https://pan.quark.cn/s/a56c5a6ec9c2)
