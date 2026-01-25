import os
import sys
import json
import torch
import random
import numpy as np
import folder_paths
import gc
import shutil
import librosa

# 添加本地 qwen_tts 目录到 Python 路径（参考 ComfyUI-Qwen-TTS 的方式）
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 尝试导入 qwen_tts（参考 ComfyUI-Qwen-TTS 的导入方式）
try:
    from ._qwen_tts_haigc import Qwen3TTSModel
    from ._qwen_tts_haigc.inference.qwen3_tts_model import VoiceClonePromptItem
except (ImportError, ValueError):
    try:
        from _qwen_tts_haigc import Qwen3TTSModel
        from _qwen_tts_haigc.inference.qwen3_tts_model import VoiceClonePromptItem
    except ImportError as e:
        import traceback
        print(f"❌ Failed to import Qwen3-TTS: {e}")
        traceback.print_exc()
        print("Please ensure the qwen_tts package is present in the plugin directory.")
        raise

# Helper to convert ComfyUI audio to numpy
def comfy_audio_to_numpy(audio_dict):
    if audio_dict is None:
        return None
    waveform = audio_dict['waveform'] # [batch, channels, samples]
    sample_rate = audio_dict['sample_rate']
    
    # Take the first item in batch and mix down to mono if needed, or keep as is?
    # Qwen might expect mono.
    # waveform is tensor.
    
    # Use the first batch
    wav = waveform[0]
    
    # If stereo, mix to mono?
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
        
    wav_np = wav.squeeze().cpu().numpy()
    return (wav_np, sample_rate)

LANGUAGE_MAP = {
    "自动": "auto",
    "中文": "Chinese",
    "英文": "English",
    "日文": "Japanese",
    "韩文": "Korean",
    "德文": "German",
    "法文": "French",
    "俄文": "Russian",
    "葡萄牙文": "Portuguese",
    "西班牙文": "Spanish",
    "意大利文": "Italian",
}
LANGUAGE_OPTIONS = list(LANGUAGE_MAP.keys())

def apply_seed(seed_value):
    seed_value = int(seed_value) & 0xFFFFFFFF
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    return seed_value

def normalize_text_input(text, batch_mode):
    if isinstance(text, list):
        return text if len(text) > 1 else (text[0] if len(text) == 1 else "")
    if text is None:
        return ""
    raw = str(text)
    stripped = raw.strip()
    if batch_mode:
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    parsed_clean = [str(x).strip() for x in parsed if str(x).strip()]
                    return parsed_clean if len(parsed_clean) > 1 else (parsed_clean[0] if len(parsed_clean) == 1 else "")
            except Exception:
                pass
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        return lines if len(lines) > 1 else (lines[0] if len(lines) == 1 else "")
    return stripped

def _to_mono_wav(wav):
    if not isinstance(wav, np.ndarray):
        return None
    if wav.ndim == 1:
        return wav
    if wav.ndim == 2:
        if wav.shape[0] <= 8 and wav.shape[0] < wav.shape[1]:
            return wav.mean(axis=0)
        if wav.shape[1] <= 8 and wav.shape[1] < wav.shape[0]:
            return wav.mean(axis=1)
        if wav.shape[0] >= wav.shape[1]:
            return wav.mean(axis=1)
        return wav.mean(axis=0)
    return wav.reshape(-1)

def _audio_duration_seconds(wav, sr):
    mono = _to_mono_wav(wav)
    if mono is None:
        return 0.0
    if sr is None or sr <= 0:
        return 0.0
    return float(len(mono)) / float(sr)

def _effective_speech_duration(wav, sr, top_db=30):
    mono = _to_mono_wav(wav)
    if mono is None:
        return 0.0
    if sr is None or sr <= 0:
        return 0.0
    if mono.size == 0:
        return 0.0
    mono = mono.astype(np.float32)
    intervals = librosa.effects.split(mono, top_db=top_db)
    if intervals is None or len(intervals) == 0:
        return 0.0
    total = 0
    for start, end in intervals:
        total += (end - start)
    return float(total) / float(sr)

def _supports_voice_clone_prompt(model):
    if model is None or not hasattr(model, "model"):
        return False
    return getattr(model.model, "tts_model_type", None) == "base"

def _prompt_item_get(item, key, default=None):
    if hasattr(item, key):
        return getattr(item, key)
    if isinstance(item, dict):
        return item.get(key, default)
    return default

def _to_prompt_dict(prompt):
    if isinstance(prompt, dict):
        return prompt
    if isinstance(prompt, list):
        ref_code = []
        ref_spk = []
        xvec = []
        icl = []
        for it in prompt:
            ref_code.append(_prompt_item_get(it, "ref_code", None))
            ref_spk.append(_prompt_item_get(it, "ref_spk_embedding", None))
            xvec_val = bool(_prompt_item_get(it, "x_vector_only_mode", False))
            icl_val = _prompt_item_get(it, "icl_mode", None)
            if icl_val is None:
                icl_val = not xvec_val
            xvec.append(bool(xvec_val))
            icl.append(bool(icl_val))
        return {
            "ref_code": ref_code,
            "ref_spk_embedding": ref_spk,
            "x_vector_only_mode": xvec,
            "icl_mode": icl,
        }
    raise ValueError("Invalid voice clone prompt format.")

def _normalize_prompt_dict(prompt_dict):
    if not isinstance(prompt_dict, dict):
        raise ValueError("Invalid voice clone prompt format.")
    ref_code_list = prompt_dict.get("ref_code", None)
    ref_spk_list = prompt_dict.get("ref_spk_embedding", None)
    xvec_list = prompt_dict.get("x_vector_only_mode", None)
    icl_list = prompt_dict.get("icl_mode", None)
    if ref_spk_list is None:
        raise ValueError("Missing ref_spk_embedding in voice prompt.")
    if isinstance(ref_spk_list, list) and len(ref_spk_list) == 0:
        raise ValueError("Voice prompt is empty. Please regenerate the role preset.")
    if not isinstance(ref_spk_list, list):
        ref_spk_list = [ref_spk_list]
    if ref_code_list is None:
        ref_code_list = [None] * len(ref_spk_list)
    if not isinstance(ref_code_list, list):
        ref_code_list = [ref_code_list]
    if xvec_list is None:
        xvec_list = [False] * len(ref_spk_list)
    if not isinstance(xvec_list, list):
        xvec_list = [xvec_list]
    if icl_list is None:
        icl_list = [not bool(x) for x in xvec_list]
    if not isinstance(icl_list, list):
        icl_list = [icl_list]

    norm_ref_code = []
    for v in ref_code_list:
        if v is None:
            norm_ref_code.append(None)
        elif torch.is_tensor(v):
            norm_ref_code.append(v)
        else:
            norm_ref_code.append(torch.tensor(v))

    norm_ref_spk = []
    for v in ref_spk_list:
        if v is None:
            raise ValueError("Missing ref_spk_embedding in voice prompt.")
        if torch.is_tensor(v):
            norm_ref_spk.append(v)
        else:
            norm_ref_spk.append(torch.tensor(v))

    return {
        "ref_code": norm_ref_code,
        "ref_spk_embedding": norm_ref_spk,
        "x_vector_only_mode": [bool(x) for x in xvec_list],
        "icl_mode": [bool(x) for x in icl_list],
    }

def _extract_role_name_from_payload(payload, fallback_name=""):
    if isinstance(payload, dict):
        for key in ("role", "role_name", "角色名", "name"):
            val = payload.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        if "voice_clone_prompt" in payload and isinstance(payload["voice_clone_prompt"], dict):
            for key in ("role", "role_name", "角色名", "name"):
                val = payload["voice_clone_prompt"].get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
    if isinstance(payload, list) and len(payload) > 0:
        for key in ("role", "role_name", "name"):
            val = _prompt_item_get(payload[0], key, None)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return (fallback_name or "").strip()

def _build_prompt_from_speaker_encoder(model, ref_audio_np_tuple):
    if model is None or not hasattr(model, "model"):
        raise ValueError("Model is not available for speaker embedding extraction.")
    if not hasattr(model.model, "speaker_encoder") or model.model.speaker_encoder is None:
        raise ValueError("Current model does not support speaker embedding extraction.")
    if not isinstance(ref_audio_np_tuple, tuple) or len(ref_audio_np_tuple) != 2:
        raise ValueError("Invalid reference audio format for speaker embedding extraction.")
    wav, sr = ref_audio_np_tuple
    if not isinstance(wav, np.ndarray):
        raise ValueError("Reference audio must be a numpy array.")
    wav = _to_mono_wav(wav)
    if wav is None or wav.size == 0:
        raise ValueError("Reference audio is empty.")
    wav = wav.astype(np.float32)
    target_sr = int(getattr(model.model, "speaker_encoder_sample_rate", 24000))
    if sr != target_sr:
        wav = librosa.resample(y=wav, orig_sr=int(sr), target_sr=target_sr).astype(np.float32)
    min_samples = int(target_sr * 0.5)
    if wav.shape[0] < min_samples:
        raise ValueError(f"Reference audio is too short after resampling ({wav.shape[0] / target_sr:.2f}s).")
    spk_emb = model.model.extract_speaker_embedding(audio=wav, sr=target_sr)
    if spk_emb is None or (torch.is_tensor(spk_emb) and spk_emb.numel() == 0):
        raise ValueError("Speaker embedding extraction returned empty result.")
    return [VoiceClonePromptItem(
        ref_code=None,
        ref_spk_embedding=spk_emb,
        x_vector_only_mode=True,
        icl_mode=False,
        ref_text=None,
    )]

def _resolve_prompt_path(path):
    raw = (path or "").strip()
    if not raw:
        raise ValueError("【Qwen3TTS Error】文件路径不能为空。")
    base_path, ext = os.path.splitext(raw)
    if ext == "":
        raw = f"{raw}.pt"
    if os.path.isabs(raw):
        if not os.path.exists(raw):
            raise FileNotFoundError(f"文件不存在: {raw}")
        return raw
    candidates = []
    if hasattr(folder_paths, "get_output_directory"):
        candidates.append(folder_paths.get_output_directory())
    else:
        candidates.append(getattr(folder_paths, "output_directory", None))
    candidates.append(getattr(folder_paths, "input_directory", None))
    candidates.append(os.path.join(current_dir, "output"))
    candidates.append(current_dir)
    candidates = [c for c in candidates if c]
    for base in candidates:
        candidate_path = os.path.join(base, raw)
        if os.path.exists(candidate_path):
            return candidate_path
    
    # Try recursive search if file not found directly
    has_sep = os.path.sep in raw or (os.path.altsep is not None and os.path.altsep in raw)
    if not has_sep:
        target_filename = os.path.basename(raw)
        for base in candidates:
            for root, _, files in os.walk(base):
                if target_filename in files:
                    return os.path.join(root, target_filename)
                    
    # Also check ComfyUI root directory as a fallback
    # comfy_root = os.path.dirname(folder_paths.base_path) if hasattr(folder_paths, "base_path") else os.path.abspath(os.path.join(current_dir, "../../.."))
    # candidate_path = os.path.join(comfy_root, raw)
    # if os.path.exists(candidate_path):
    #    return candidate_path
    
    raise FileNotFoundError(f"文件不存在: {raw} (搜索路径: {candidates})")

def get_voice_prompt_files():
    candidates = set()
    search_paths = []
    
    if hasattr(folder_paths, "get_output_directory"):
        out_dir = folder_paths.get_output_directory()
    else:
        out_dir = getattr(folder_paths, "output_directory", None)
        
    if out_dir:
        search_paths.append(os.path.join(out_dir, "qwen_tts_presets"))
        search_paths.append(out_dir)
    
    inp_dir = getattr(folder_paths, "input_directory", None)
    if inp_dir:
        search_paths.append(os.path.join(inp_dir, "qwen_tts_presets"))
        search_paths.append(inp_dir)
    
    # search_paths.append(os.path.join(current_dir, "output"))
    # search_paths.append(current_dir)
    
    # Add ComfyUI root directory as well - REMOVED to avoid scanning all models
    # comfy_root = os.path.dirname(folder_paths.base_path) if hasattr(folder_paths, "base_path") else os.path.abspath(os.path.join(current_dir, "../../.."))
    # search_paths.append(comfy_root)

    for path in search_paths:
        if not path or not os.path.exists(path):
            continue
        try:
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(".pt"):
                        # Only add the filename, assuming _resolve_prompt_path handles the rest
                        # If multiple files have same name, this dropdown just shows the name.
                        # The resolution logic prioritizes output dir, then input, etc.
                        candidates.add(file)
        except Exception:
            continue
            
    if not candidates:
        return ["None"]
    
    # Sort candidates: Prioritize human-readable names over long hash-like filenames
    def sort_key(name):
        # 1. Heuristic: if name is long (>20 chars) and looks like hex, it's low priority (likely a generated hash)
        is_hex_hash = False
        base = os.path.splitext(name)[0]
        if len(base) > 20 and all(c in "0123456789abcdefABCDEF" for c in base):
            is_hex_hash = True
            
        # 2. Prioritize "known" presets if any (optional, but sorting by length helps)
        # 3. Sort by length (shorter = more likely human named)
        # 4. Alphabetical
        return (is_hex_hash, len(name), name.lower())

    return sorted(list(candidates), key=sort_key)

class Qwen3TTSModelLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "模型名称": ([
                    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                ],),
                "运行设备": (["cuda", "cpu", "auto"], {"default": "cuda"}),
                "精度": (["fp16", "fp32"], {"default": "fp16"}),
            }
        }

    RETURN_TYPES = ("QWEN3_TTS_MODEL",)
    RETURN_NAMES = ("模型",)
    FUNCTION = "load_model"
    CATEGORY = "Qwen3TTS"
    DESCRIPTION = "加载 Qwen3-TTS 模型。模型必须已存在于 ComfyUI/models/Qwen3TTS 目录中。"

    def load_model(self, 模型名称, 运行设备, 精度):
        model_name = 模型名称
        device = 运行设备
        precision = 精度
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading Qwen3TTS model: {model_name} on {device}...")
        
        # 固定模型读取路径：ComfyUI/models/qwen-tts
        qwen_models_dir = os.path.join(folder_paths.models_dir, "qwen-tts")
        
        # Clean model folder name (remove "Qwen/" prefix)
        # e.g. "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign" -> "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        model_folder_name = model_name.split("/")[-1]
        model_path = os.path.join(qwen_models_dir, model_folder_name)
        
        # Check for "messy" folder name from previous downloads (e.g., "1.7B" -> "1__7B")
        # 兼容旧版本的文件夹命名
        if not os.path.exists(model_path) or not os.path.isdir(model_path):
            messy_folder_name = model_folder_name.replace(".", "__")
            
            possible_messy_paths = [
                os.path.join(qwen_models_dir, messy_folder_name),
                os.path.join(qwen_models_dir, model_folder_name.replace("1.7B", "1__7B")),
                os.path.join(qwen_models_dir, model_folder_name.replace("0.6B", "0__6B")),
                os.path.join(qwen_models_dir, model_folder_name.replace("1.7B", "1-7B")),
                os.path.join(qwen_models_dir, model_folder_name.replace("0.6B", "0-6B")),
                os.path.join(qwen_models_dir, model_folder_name.replace(".", "-")),
            ]
            
            for bad_path in possible_messy_paths:
                if os.path.exists(bad_path) and os.path.isdir(bad_path):
                    print(f"Found legacy model folder: {bad_path}. Renaming to {model_path}...")
                    try:
                        shutil.move(bad_path, model_path)
                        break
                    except Exception as e:
                        print(f"Failed to rename folder: {e}")
        
        # 检查模型路径是否存在
        if not os.path.exists(model_path) or not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"模型未找到: {model_path}\n"
                f"请确保模型已下载到: {qwen_models_dir}\n"
                f"模型文件夹名称应为: {model_folder_name}"
            )
        
        # Load model using ComfyUI cache
        # RH: 强制使用本地文件，禁用下载功能（参考 ComfyUI-Qwen-TTS 的加载方式）
        dtype = torch.float16 if precision == "fp16" else torch.float32
        
        # 参考 ComfyUI-Qwen-TTS 的加载方式，但强制使用本地文件
        try:
            # 尝试使用 device_map 和 attn_implementation（如果支持）
            model = Qwen3TTSModel.from_pretrained(
                model_path, 
                device_map=device, 
                dtype=dtype,
                cache_dir=qwen_models_dir,
                local_files_only=True,  # RH: 强制仅使用本地文件
                force_download=False,   # RH: 禁用强制下载
                attn_implementation="flash_attention_2"  # 参考 ComfyUI-Qwen-TTS
            )
        except TypeError:
            # Fallback: 如果某些参数不支持，尝试简化版本
            try:
                model = Qwen3TTSModel.from_pretrained(
                    model_path, 
                    device_map=device, 
                    torch_dtype=dtype,
                    cache_dir=qwen_models_dir,
                    local_files_only=True,  # RH: 强制仅使用本地文件
                    force_download=False    # RH: 禁用强制下载
                )
            except TypeError:
                # 最简版本：只使用基本参数
                model = Qwen3TTSModel.from_pretrained(
                    model_path, 
                    cache_dir=qwen_models_dir,
                    local_files_only=True,  # RH: 强制仅使用本地文件
                    force_download=False    # RH: 禁用强制下载
                )
            
        return (model,)

class Qwen3TTSVoiceDesign:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "模型": ("QWEN3_TTS_MODEL",),
                "文本": ("STRING", {"multiline": True, "default": "Hello, this is a test."}),
                "提示词": ("STRING", {"multiline": True, "default": "A young female voice, energetic and bright."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "语言": (LANGUAGE_OPTIONS, {"default": "自动"}),
                "自动卸载模型": ("BOOLEAN", {"default": False, "label_on": "是", "label_off": "否"}),
                "最大生成Token数": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 64, "display": "number", "tooltip": "限制生成的最大长度。默认2048，通常足够。设为0则根据文本自动调整（不限制）。"}),
                "批量模式": ("BOOLEAN", {"default": False}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number"}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1, "display": "number"}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 2.0, "step": 0.01, "display": "number"}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 2.0, "step": 0.01, "display": "number"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("音频",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3TTS"
    DESCRIPTION = "使用声音设计（Voice Design）生成语音，基于提示词创建声音。\n⚠️ 需要加载带有 'VoiceDesign' 的模型（如 Qwen3-TTS-12Hz-1.7B-VoiceDesign）。"

    def generate(self, 模型, 文本, 提示词, 语言, seed=0, 批量模式=False, 自动卸载模型=False, 最大生成Token数=2048, top_p=1.0, top_k=50, temperature=0.9, repetition_penalty=1.05):
        model = 模型
        text = normalize_text_input(文本, 批量模式)
        instruct = 提示词
        language = 语言
        max_new_tokens = 最大生成Token数
        
        language = LANGUAGE_MAP.get(language, language)
        if language == "auto":
            language = None
            
        print(f"Generating Voice Design... Text len: {len(text)}")
        
        # generate_voice_design returns (audio_chunks, sample_rate)
        # audio_chunks is List[numpy.ndarray]
        
        # Handle max_new_tokens logic
        # If user sets it to a valid number, we pass it.
        # If 0 or very large, we might let the model decide, but generate_* usually expects a value or has a default.
        # Qwen-TTS generate methods usually accept kwargs.
        
        kwargs = {}
        if max_new_tokens > 0:
            kwargs["max_new_tokens"] = max_new_tokens
            
        kwargs["top_p"] = top_p
        kwargs["top_k"] = top_k
        kwargs["temperature"] = temperature
        kwargs["repetition_penalty"] = repetition_penalty
        
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        np_state = np.random.get_state()
        py_state = random.getstate()
        
        apply_seed(seed)

        try:
            outputs, sample_rate = model.generate_voice_design(
                text=text,
                instruct=instruct,
                language=language,
                **kwargs
            )
        except (ValueError, AttributeError) as e:
            error_msg = str(e).lower()
            if "does not support generate_voice_design" in error_msg or "generate_voice_design" in error_msg or "has no attribute" in error_msg:
                raise ValueError(
                    "【Qwen3TTS Error】您当前加载的模型不支持'声音设计(Voice Design)'功能。\n"
                    "请在加载器中选择带有 'VoiceDesign' 的模型：\n"
                    "  - Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign\n\n"
                    "[Qwen3TTS Error] The loaded model does not support 'Voice Design'.\n"
                    "Please select a 'VoiceDesign' model in the Loader:\n"
                    "  - Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
                ) from e
            raise e
        finally:
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            torch.set_rng_state(rng_state)
            np.random.set_state(np_state)
            random.setstate(py_state)
            if 自动卸载模型:
                print("Unloading model to CPU...")
                if hasattr(model, "model"):
                    model.model.to("cpu")
                if hasattr(model, "device"):
                    model.device = torch.device("cpu")
                torch.cuda.empty_cache()
                gc.collect()
        
        # Concatenate chunks
        full_audio = np.concatenate(outputs)
        
        # ComfyUI AUDIO format expects {'waveform': tensor, 'sample_rate': int}
        # waveform shape should be [batch_size, channels, samples]
        
        # full_audio is typically [samples] (mono) or [samples, channels]
        # Qwen-TTS usually outputs [samples] for mono audio.
        
        tensor_audio = torch.from_numpy(full_audio).float()
        
        if tensor_audio.ndim == 1:
            # [samples] -> [1, 1, samples]
            tensor_audio = tensor_audio.unsqueeze(0).unsqueeze(0)
        elif tensor_audio.ndim == 2:
            # Check if it's [channels, samples] or [samples, channels]
            # Assuming [samples, channels] from common audio libs, but need to be sure.
            # If [samples, channels], we want [1, channels, samples]
            if tensor_audio.shape[0] > tensor_audio.shape[1]: 
                # Likely [samples, channels]
                tensor_audio = tensor_audio.permute(1, 0).unsqueeze(0)
            else:
                # Likely [channels, samples]
                tensor_audio = tensor_audio.unsqueeze(0)
        
        return ({"waveform": tensor_audio, "sample_rate": sample_rate},)

class Qwen3TTSVoiceClone:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "模型": ("QWEN3_TTS_MODEL",),
                "文本": ("STRING", {"multiline": True, "default": "Hello, I am cloning this voice."}),
                "参考音频": ("AUDIO",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "参考文本": ("STRING", {"multiline": True, "default": ""}),
                "语言": (LANGUAGE_OPTIONS, {"default": "自动"}),
                "自动卸载模型": ("BOOLEAN", {"default": False, "label_on": "是", "label_off": "否"}),
                "最大生成Token数": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 64, "display": "number", "tooltip": "限制生成的最大长度。默认2048，通常足够。设为0则根据文本自动调整（不限制）。"}),
                "批量模式": ("BOOLEAN", {"default": False}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number"}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1, "display": "number"}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 2.0, "step": 0.01, "display": "number"}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 2.0, "step": 0.01, "display": "number"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "QWEN3_TTS_VOICE_PROMPT")
    RETURN_NAMES = ("音频", "角色预设")
    FUNCTION = "generate"
    CATEGORY = "Qwen3TTS"
    DESCRIPTION = "使用声音克隆（Voice Clone）生成语音，基于参考音频。\n⚠️ 需要加载带有 'Base' 的模型（如 Qwen3-TTS-12Hz-1.7B-Base）。"

    def generate(self, 模型, 文本, 参考音频, 参考文本, 语言, seed=0, 批量模式=False, 自动卸载模型=False, 最大生成Token数=2048, top_p=1.0, top_k=50, temperature=0.9, repetition_penalty=1.05):
        model = 模型
        text = normalize_text_input(文本, 批量模式)
        reference_audio = 参考音频
        reference_text = 参考文本
        language = 语言
        max_new_tokens = 最大生成Token数
        
        language = LANGUAGE_MAP.get(language, language)
        if language == "auto":
            language = None
            
        if reference_text is not None and reference_text.strip():
            reference_text = reference_text.strip()
            x_vector_only_mode = False
        else:
            reference_text = None
            x_vector_only_mode = True
            
        print(f"Generating Voice Clone... Text len: {len(text)}")
        
        # Convert ref audio
        ref_audio_np_tuple = comfy_audio_to_numpy(reference_audio)
        
        kwargs = {}
        if max_new_tokens > 0:
            kwargs["max_new_tokens"] = max_new_tokens
            
        kwargs["top_p"] = top_p
        kwargs["top_k"] = top_k
        kwargs["temperature"] = temperature
        kwargs["repetition_penalty"] = repetition_penalty
            
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        np_state = np.random.get_state()
        py_state = random.getstate()
        
        apply_seed(seed)

        try:
            outputs, sample_rate = model.generate_voice_clone(
                text=text,
                ref_audio=ref_audio_np_tuple,
                ref_text=reference_text,
                x_vector_only_mode=bool(x_vector_only_mode),
                language=language,
                **kwargs
            )
        except (ValueError, AttributeError) as e:
            error_msg = str(e).lower()
            if "does not support generate_voice_clone" in error_msg or "generate_voice_clone" in error_msg or "has no attribute" in error_msg:
                raise ValueError(
                    "【Qwen3TTS Error】您当前加载的模型不支持'声音克隆(Voice Clone)'功能。\n"
                    "请在加载器中选择带有 'Base' 的模型：\n"
                    "  - Qwen/Qwen3-TTS-12Hz-1.7B-Base\n"
                    "  - Qwen/Qwen3-TTS-12Hz-0.6B-Base\n\n"
                    "[Qwen3TTS Error] The loaded model does not support 'Voice Clone'.\n"
                    "Please select a 'Base' model in the Loader:\n"
                    "  - Qwen/Qwen3-TTS-12Hz-1.7B-Base\n"
                    "  - Qwen/Qwen3-TTS-12Hz-0.6B-Base"
                ) from e
            raise e
        finally:
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            torch.set_rng_state(rng_state)
            np.random.set_state(np_state)
            random.setstate(py_state)
            if 自动卸载模型:
                print("Unloading model to CPU...")
                if hasattr(model, "model"):
                    model.model.to("cpu")
                if hasattr(model, "device"):
                    model.device = torch.device("cpu")
                torch.cuda.empty_cache()
                gc.collect()
        
        full_audio = np.concatenate(outputs)
        
        # ComfyUI AUDIO format expects {'waveform': tensor, 'sample_rate': int}
        # waveform shape should be [batch_size, channels, samples]
        
        tensor_audio = torch.from_numpy(full_audio).float()
        
        if tensor_audio.ndim == 1:
            # [samples] -> [1, 1, samples]
            tensor_audio = tensor_audio.unsqueeze(0).unsqueeze(0)
        elif tensor_audio.ndim == 2:
            if tensor_audio.shape[0] > tensor_audio.shape[1]: 
                # Likely [samples, channels] -> [1, channels, samples]
                tensor_audio = tensor_audio.permute(1, 0).unsqueeze(0)
            else:
                # Likely [channels, samples] -> [1, channels, samples]
                tensor_audio = tensor_audio.unsqueeze(0)
        
        # Create Role Preset from input reference audio
        prompt_items = []
        try:
            ref_wav = ref_audio_np_tuple[0]
            ref_sr = ref_audio_np_tuple[1]
            if isinstance(ref_wav, np.ndarray):
                total_duration = _audio_duration_seconds(ref_wav, ref_sr)
                effective_duration = _effective_speech_duration(ref_wav, ref_sr)
                duration_for_check = effective_duration if effective_duration > 0 else total_duration
                if duration_for_check < 3.0:
                    print(
                        f"Warning: Reference audio effective duration is short "
                        f"(effective {effective_duration:.2f}s, total {total_duration:.2f}s). "
                        "Role preset might be unstable or empty. Recommended: >3s."
                    )

            prompt_items = model.create_voice_clone_prompt(
                ref_audio=ref_audio_np_tuple,
                ref_text=reference_text,
                x_vector_only_mode=x_vector_only_mode,
            )
            
            # Validate that prompt_items is not empty and contains valid embeddings
            if not prompt_items:
                raise ValueError("create_voice_clone_prompt returned empty list.")
            
            for i, item in enumerate(prompt_items):
                if item is None:
                    raise ValueError(f"Voice clone prompt item at index {i} is None.")
                if item.ref_spk_embedding is None:
                    raise ValueError(f"Speaker embedding at index {i} is None. Voice feature extraction failed.")
                if item.ref_spk_embedding.numel() == 0:
                    raise ValueError(f"Speaker embedding at index {i} is empty. Voice feature extraction failed.")
                    
        except Exception as e:
            error_msg = str(e)
            print(f"[Qwen3TTS] Warning: Failed to create role preset from reference audio: {error_msg}")
            try:
                prompt_items = _build_prompt_from_speaker_encoder(model, ref_audio_np_tuple)
                print("[Qwen3TTS] Fallback: created role preset with speaker encoder only.")
            except Exception as fallback_error:
                print(f"[Qwen3TTS] Fallback failed: {fallback_error}")
                print(f"[Qwen3TTS] This might be due to audio quality issues (too much silence, noise, or non-speech content).")
                print(f"[Qwen3TTS] Role preset output will be empty - you may need to use better reference audio or enable x_vector_only_mode.")
                prompt_items = []
                
        return ({"waveform": tensor_audio, "sample_rate": sample_rate}, prompt_items)

class Qwen3TTSVoiceClonePromptSave:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "角色预设": ("QWEN3_TTS_VOICE_PROMPT",),
                "角色名": ("STRING", {"default": ""}),
            },
            "optional": {
                "保存目录": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "QWEN3_TTS_VOICE_PROMPT")
    RETURN_NAMES = ("文件路径", "角色预设")
    OUTPUT_NODE = True
    FUNCTION = "save_prompt"
    CATEGORY = "Qwen3TTS"
    DESCRIPTION = "保存角色预设到本地文件，便于持久复用。"

    def save_prompt(self, 角色预设, 角色名, 保存目录=""):
        try:
            prompt_dict = _to_prompt_dict(角色预设)
            prompt_dict = _normalize_prompt_dict(prompt_dict)
        except ValueError as e:
            error_msg = str(e).lower()
            if "empty" in error_msg or "missing" in error_msg or "none" in error_msg:
                raise ValueError(
                    "角色预设为空，无法保存。\n\n"
                    "可能原因及解决方案：\n"
                    "1. 【音频质量问题】音频包含过多静音、噪音或非语音内容。\n"
                    "   → 解决方案：使用清晰的人声音频，避免背景音乐或噪音。\n\n"
                    "2. 【音频时长不足】音频实际有效语音部分过短。\n"
                    "   → 解决方案：确保音频包含至少 3-5 秒的连续清晰语音。\n\n"
                    "3. 【特征提取失败】可能是模型处理音频时出现问题。\n"
                    "   → 解决方案：尝试使用不同的音频文件，或在 'Voice Clone' 节点中使用 'x_vector_only_mode' 选项。\n\n"
                    "4. 【音频采样率不匹配】音频采样率可能不是 16kHz 或 24kHz。\n"
                    "   → 解决方案：将音频转换为 16kHz 或 24kHz 采样率。\n\n"
                    "请检查您的输入音频，确保音频质量良好后重试。\n\n"
                    "[Error] Role preset is empty. This usually happens when:\n"
                    "- Audio contains too much silence, noise, or non-speech content\n"
                    "- Audio duration is too short (need 3-5+ seconds of clear speech)\n"
                    "- Audio sample rate is not compatible (try 16kHz or 24kHz)\n\n"
                    "Please use clear speech audio with minimal background noise/silence."
                )
            raise

        role_name = (角色名 or "").strip()
        if not role_name:
            raise ValueError("【Qwen3TTS Error】角色名不能为空。")

        # Additional validation: check each embedding is valid
        ref_spk_list = prompt_dict.get("ref_spk_embedding", [])
        for i, emb in enumerate(ref_spk_list):
            if emb is None:
                raise ValueError(f"Speaker embedding at index {i} is None. Cannot save invalid preset.")
            if not torch.is_tensor(emb):
                raise ValueError(f"Speaker embedding at index {i} is not a tensor. Cannot save invalid preset.")
            if emb.numel() == 0:
                raise ValueError(f"Speaker embedding at index {i} is empty. Cannot save invalid preset.")
            if torch.isnan(emb).any() or torch.isinf(emb).any():
                raise ValueError(f"Speaker embedding at index {i} contains NaN or Inf values. Cannot save invalid preset.")

        filename = role_name
        if not filename.lower().endswith(".pt"):
            filename = f"{filename}.pt"

        save_dir = (保存目录 or "").strip()
        if not save_dir:
            if hasattr(folder_paths, "get_output_directory"):
                base_out = folder_paths.get_output_directory()
            else:
                base_out = getattr(folder_paths, "output_directory", None)
            
            if base_out:
                save_dir = os.path.join(base_out, "qwen_tts_presets")
            else:
                save_dir = os.path.join(current_dir, "output", "qwen_tts_presets")

        if not save_dir:
            # Fallback
            save_dir = os.path.join(current_dir, "output")
            
        os.makedirs(save_dir, exist_ok=True)

        out_path = os.path.join(save_dir, filename)
        prompt_with_role = dict(prompt_dict)
        prompt_with_role["role_name"] = role_name
        torch.save({"voice_clone_prompt": prompt_dict, "role_name": role_name}, out_path)
        return (out_path, prompt_with_role)

class Qwen3TTSRolePresetsInput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "已有角色预设": ("QWEN3_TTS_ROLE_PRESETS",),
                "角色预设1": ("QWEN3_TTS_VOICE_PROMPT",),
                "角色预设2": ("QWEN3_TTS_VOICE_PROMPT",),
                "角色预设3": ("QWEN3_TTS_VOICE_PROMPT",),
                "角色预设4": ("QWEN3_TTS_VOICE_PROMPT",),
                "角色预设5": ("QWEN3_TTS_VOICE_PROMPT",),
                "角色预设6": ("QWEN3_TTS_VOICE_PROMPT",),
            },
        }

    RETURN_TYPES = ("QWEN3_TTS_ROLE_PRESETS",)
    RETURN_NAMES = ("角色预设",)
    FUNCTION = "merge"
    CATEGORY = "Qwen3TTS"
    DESCRIPTION = "一次输入六个角色预设，可串联多个节点扩展数量。角色名从预设内读取。"

    def merge(
        self,
        已有角色预设=None,
        角色预设1=None,
        角色预设2=None,
        角色预设3=None,
        角色预设4=None,
        角色预设5=None,
        角色预设6=None,
    ):
        merged = {}
        if 已有角色预设 is not None:
            if not isinstance(已有角色预设, dict):
                raise ValueError("【Qwen3TTS Error】已有角色预设格式错误。")
            merged.update(已有角色预设)

        def _extract_role_name(preset_value):
            if preset_value is None:
                return ""
            if isinstance(preset_value, dict):
                for key in ("role", "role_name", "角色名", "name"):
                    val = preset_value.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
            if isinstance(preset_value, list) and len(preset_value) > 0:
                for key in ("role", "role_name", "name"):
                    val = _prompt_item_get(preset_value[0], key, None)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
            return ""

        def _add(preset_value):
            if preset_value is None:
                return
            role = _extract_role_name(preset_value)
            if not role:
                raise ValueError("【Qwen3TTS Error】角色预设缺少角色名信息。")
            prompt_dict = _normalize_prompt_dict(_to_prompt_dict(preset_value))
            merged[role] = prompt_dict

        _add(角色预设1)
        _add(角色预设2)
        _add(角色预设3)
        _add(角色预设4)
        _add(角色预设5)
        _add(角色预设6)

        if not merged:
            raise ValueError("【Qwen3TTS Error】未提供任何角色预设。")

        return (merged,)

class Qwen3TTSPromptSix:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "提示词1": ("STRING", {"multiline": True, "default": ""}),
                "提示词2": ("STRING", {"multiline": True, "default": ""}),
                "提示词3": ("STRING", {"multiline": True, "default": ""}),
                "提示词4": ("STRING", {"multiline": True, "default": ""}),
                "提示词5": ("STRING", {"multiline": True, "default": ""}),
                "提示词6": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("提示词1", "提示词2", "提示词3", "提示词4", "提示词5", "提示词6")
    FUNCTION = "output"
    CATEGORY = "Qwen3TTS"
    DESCRIPTION = "输入六个提示词并分别输出。"

    def output(self, 提示词1="", 提示词2="", 提示词3="", 提示词4="", 提示词5="", 提示词6=""):
        return (提示词1, 提示词2, 提示词3, 提示词4, 提示词5, 提示词6)

class Qwen3TTSPresetVoiceLoad:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "预设文件": (get_voice_prompt_files(), ),
            },
            "optional": {
                "手动输入文件名": ("STRING", {"default": "", "tooltip": "手动输入文件名或路径。如果指定，将忽略下拉列表的选择。可用于加载新生成但尚未显示在列表中的文件。"}),
            }
        }

    RETURN_TYPES = ("QWEN3_TTS_VOICE_PROMPT",)
    RETURN_NAMES = ("角色预设",)
    FUNCTION = "load_prompt"
    CATEGORY = "Qwen3TTS"
    DESCRIPTION = "从下拉列表中选择已保存的角色预设文件(.pt)。\n也可以通过'手动输入文件名'直接指定文件，无需刷新浏览器。"

    def load_prompt(self, 预设文件, 手动输入文件名=""):
        filename = (手动输入文件名 or "").strip()
        if not filename:
            filename = 预设文件
        
        if not filename or filename == "None":
            raise ValueError("No voice prompt file selected.")
        
        path = _resolve_prompt_path(filename)
        
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:
            raise ValueError(
                f"加载角色预设文件失败: {e}\n"
                "文件可能已损坏或格式不正确。"
            )
            
        if isinstance(payload, dict) and "voice_clone_prompt" in payload:
            prompt_dict = payload["voice_clone_prompt"]
        else:
            prompt_dict = payload
            
        role_from_payload = _extract_role_name_from_payload(payload, os.path.splitext(os.path.basename(filename))[0])

        try:
            prompt_dict = _normalize_prompt_dict(prompt_dict)
        except ValueError as e:
            raise ValueError(
                f"角色预设格式无效: {e}\n"
                "预设文件可能已损坏或不完整。请重新生成角色预设。"
            )
            
        ref_spk_list = prompt_dict.get("ref_spk_embedding", [])
        if not ref_spk_list or (isinstance(ref_spk_list, list) and len(ref_spk_list) == 0):
            raise ValueError(
                "角色预设中的声音特征为空。\n"
                "预设文件可能已损坏或不完整。请重新生成角色预设。"
            )

        prompt_with_role = dict(prompt_dict)
        if role_from_payload:
            prompt_with_role["role_name"] = role_from_payload
        return (prompt_with_role,)

class Qwen3TTSDialogueSynthesis:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "模型": ("QWEN3_TTS_MODEL",),
                "对白文本": ("STRING", {"multiline": True, "default": "旁白:这里是第一句\n御姐:这里是第二句\n小林:这里是第三句"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "语言": (LANGUAGE_OPTIONS, {"default": "自动"}),
                "角色映射(可选)": ("STRING", {"multiline": True, "default": ""}),
                "角色预设": ("QWEN3_TTS_ROLE_PRESETS",),
                "自动卸载模型": ("BOOLEAN", {"default": False, "label_on": "是", "label_off": "否"}),
                "最大生成Token数": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 64, "display": "number", "tooltip": "限制生成的最大长度。默认2048，通常足够。设为0则根据文本自动调整（不限制）。"}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number"}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1, "display": "number"}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 2.0, "step": 0.01, "display": "number"}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 2.0, "step": 0.01, "display": "number"}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("音频",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3TTS"
    DESCRIPTION = "输入多角色对白文本，自动将角色名映射为同名 .pt 预设并合成整段音频，可用“角色名=文件名”覆盖，且优先使用角色预设输入端口。"

    def generate(self, 模型, 对白文本, 语言, 角色映射=None, 角色预设=None, seed=0, 自动卸载模型=False, 最大生成Token数=2048, top_p=1.0, top_k=50, temperature=0.9, repetition_penalty=1.05, **kwargs):
        model = 模型
        dialogue_text = (对白文本 or "").strip()
        mapping_value = 角色映射
        if not mapping_value:
            mapping_value = kwargs.get("角色映射(可选)", kwargs.get("角色映射", ""))
        mapping_text = (mapping_value or "").strip()
        language = 语言
        max_new_tokens = 最大生成Token数

        if not dialogue_text:
            raise ValueError("【Qwen3TTS Error】对白文本不能为空。")

        language = LANGUAGE_MAP.get(language, language)
        if language == "auto":
            language = None

        role_to_prompt = {}
        role_override = {}
        input_presets = {}
        if 角色预设 is not None:
            if not isinstance(角色预设, dict):
                raise ValueError("【Qwen3TTS Error】角色预设输入格式错误。")
            input_presets = 角色预设

        if mapping_text:
            for line in mapping_text.splitlines():
                raw = line.strip()
                if not raw:
                    continue
                if "=" not in raw:
                    raise ValueError(f"【Qwen3TTS Error】角色映射格式错误: {raw}")
                role, path = raw.split("=", 1)
                role = role.strip()
                path = path.strip()
                if not role or not path:
                    raise ValueError(f"【Qwen3TTS Error】角色映射格式错误: {raw}")
                if not path.lower().endswith(".pt"):
                    path = f"{path}.pt"
                role_override[role] = path

        segments = []
        for line in dialogue_text.splitlines():
            raw = line.strip()
            if not raw:
                continue
            if "：" in raw:
                role, text = raw.split("：", 1)
            elif ":" in raw:
                role, text = raw.split(":", 1)
            else:
                raise ValueError(f"【Qwen3TTS Error】对白格式错误: {raw}")
            role = role.strip()
            text = text.strip()
            if not role or not text:
                raise ValueError(f"【Qwen3TTS Error】对白格式错误: {raw}")
            segments.append((role, text))

        for role, _ in segments:
            if role not in role_to_prompt:
                if role in input_presets:
                    role_to_prompt[role] = _normalize_prompt_dict(input_presets[role])
                else:
                    filename = role_override.get(role, f"{role}.pt")
                    resolved = _resolve_prompt_path(filename)
                    try:
                        payload = torch.load(resolved, map_location="cpu", weights_only=True)
                    except Exception as e:
                        raise ValueError(
                            f"【Qwen3TTS Error】未找到角色预设文件: {role} -> {filename}\n{e}"
                        )
                    if isinstance(payload, dict) and "voice_clone_prompt" in payload:
                        prompt_dict = payload["voice_clone_prompt"]
                    else:
                        prompt_dict = payload
                    try:
                        role_to_prompt[role] = _normalize_prompt_dict(prompt_dict)
                    except ValueError as e:
                        raise ValueError(
                            f"【Qwen3TTS Error】角色预设格式无效: {role} -> {filename}\n{e}"
                        )

        kwargs = {}
        if max_new_tokens > 0:
            kwargs["max_new_tokens"] = max_new_tokens
            
        kwargs["top_p"] = top_p
        kwargs["top_k"] = top_k
        kwargs["temperature"] = temperature
        kwargs["repetition_penalty"] = repetition_penalty

        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        np_state = np.random.get_state()
        py_state = random.getstate()
        
        apply_seed(seed)

        try:
            wavs_all = []
            sample_rate = None
            for role, text in segments:
                prompt = role_to_prompt[role]
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=prompt,
                    **kwargs
                )
                if sample_rate is None:
                    sample_rate = sr
                wavs_all.append(wavs[0])
        finally:
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            torch.set_rng_state(rng_state)
            np.random.set_state(np_state)
            random.setstate(py_state)
            if 自动卸载模型:
                print("Unloading model to CPU...")
                if hasattr(model, "model"):
                    model.model.to("cpu")
                if hasattr(model, "device"):
                    model.device = torch.device("cpu")
                torch.cuda.empty_cache()
                gc.collect()

        if not wavs_all:
            raise ValueError("【Qwen3TTS Error】对白文本为空或格式不正确。")

        full_audio = np.concatenate(wavs_all)
        tensor_audio = torch.from_numpy(full_audio).float()
        if tensor_audio.ndim == 1:
            tensor_audio = tensor_audio.unsqueeze(0).unsqueeze(0)
        elif tensor_audio.ndim == 2:
            if tensor_audio.shape[0] > tensor_audio.shape[1]:
                tensor_audio = tensor_audio.permute(1, 0).unsqueeze(0)
            else:
                tensor_audio = tensor_audio.unsqueeze(0)

        return ({"waveform": tensor_audio, "sample_rate": sample_rate},)

class Qwen3TTSVoiceCloneFromPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "模型": ("QWEN3_TTS_MODEL",),
                "文本": ("STRING", {"multiline": True, "default": "Hello, I am cloning this voice."}),
                "角色预设": ("QWEN3_TTS_VOICE_PROMPT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "语言": (LANGUAGE_OPTIONS, {"default": "自动"}),
                "批量模式": ("BOOLEAN", {"default": False}),
                "自动卸载模型": ("BOOLEAN", {"default": False, "label_on": "是", "label_off": "否"}),
                "最大生成Token数": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 64, "display": "number", "tooltip": "限制生成的最大长度。默认2048，通常足够。设为0则根据文本自动调整（不限制）。"}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number"}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1, "display": "number"}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 2.0, "step": 0.01, "display": "number"}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 2.0, "step": 0.01, "display": "number"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("音频",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3TTS"
    DESCRIPTION = "使用可复用语音克隆提示生成语音，无需重复提取参考特征。"

    def generate(self, 模型, 文本, 角色预设, 语言, seed=0, 批量模式=False, 自动卸载模型=False, 最大生成Token数=2048, top_p=1.0, top_k=50, temperature=0.9, repetition_penalty=1.05):
        model = 模型
        text = normalize_text_input(文本, 批量模式)
        voice_clone_prompt = 角色预设
        language = 语言
        max_new_tokens = 最大生成Token数
        
        language = LANGUAGE_MAP.get(language, language)
        if language == "auto":
            language = None

        if text is None or (isinstance(text, str) and not text.strip()) or (isinstance(text, list) and len(text) == 0):
            raise ValueError("【Qwen3TTS Error】文本不能为空。")

        kwargs = {}
        if max_new_tokens > 0:
            kwargs["max_new_tokens"] = max_new_tokens
            
        kwargs["top_p"] = top_p
        kwargs["top_k"] = top_k
        kwargs["temperature"] = temperature
        kwargs["repetition_penalty"] = repetition_penalty

        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        np_state = np.random.get_state()
        py_state = random.getstate()
        
        apply_seed(seed)

        try:
            outputs, sample_rate = model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=voice_clone_prompt,
                **kwargs
            )
        except (ValueError, AttributeError) as e:
            error_msg = str(e).lower()
            if "does not support generate_voice_clone" in error_msg or "generate_voice_clone" in error_msg or "has no attribute" in error_msg:
                raise ValueError(
                    "【Qwen3TTS Error】您当前加载的模型不支持'声音克隆(Voice Clone)'功能。\n"
                    "请在加载器中选择带有 'Base' 的模型：\n"
                    "  - Qwen/Qwen3-TTS-12Hz-1.7B-Base\n"
                    "  - Qwen/Qwen3-TTS-12Hz-0.6B-Base\n\n"
                    "[Qwen3TTS Error] The loaded model does not support 'Voice Clone'.\n"
                    "Please select a 'Base' model in the Loader:\n"
                    "  - Qwen/Qwen3-TTS-12Hz-1.7B-Base\n"
                    "  - Qwen/Qwen3-TTS-12Hz-0.6B-Base"
                ) from e
            raise e
        finally:
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            torch.set_rng_state(rng_state)
            np.random.set_state(np_state)
            random.setstate(py_state)
            if 自动卸载模型:
                print("Unloading model to CPU...")
                if hasattr(model, "model"):
                    model.model.to("cpu")
                if hasattr(model, "device"):
                    model.device = torch.device("cpu")
                torch.cuda.empty_cache()
                gc.collect()

        full_audio = np.concatenate(outputs)

        tensor_audio = torch.from_numpy(full_audio).float()

        if tensor_audio.ndim == 1:
            tensor_audio = tensor_audio.unsqueeze(0).unsqueeze(0)
        elif tensor_audio.ndim == 2:
            if tensor_audio.shape[0] > tensor_audio.shape[1]:
                tensor_audio = tensor_audio.permute(1, 0).unsqueeze(0)
            else:
                tensor_audio = tensor_audio.unsqueeze(0)

        return ({"waveform": tensor_audio, "sample_rate": sample_rate},)

class Qwen3TTSCustomVoice:
    SPEAKER_PRESETS = {
        "Vivian": "Bright, slightly edgy young female voice.",
        "Serena": "Warm, gentle young female voice.",
        "Uncle_Fu": "Seasoned male voice with a low, mellow timbre.",
        "Dylan": "Youthful Beijing male voice with a clear, natural timbre.",
        "Eric": "Lively Chengdu male voice with a slightly husky brightness.",
        "Ryan": "Dynamic male voice with strong rhythmic drive.",
        "Aiden": "Sunny American male voice with a clear midrange.",
        "Ono_Anna": "Playful Japanese female voice with a light, nimble timbre.",
        "Sohee": "Warm Korean female voice with rich emotion.",
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "模型": ("QWEN3_TTS_MODEL",),
                "文本": ("STRING", {"multiline": True, "default": "Hello, this is a custom voice."}),
                "预设说话人": (list(s.SPEAKER_PRESETS.keys()), {"default": "Vivian"}),
            },
            "optional": {
                "语言": (LANGUAGE_OPTIONS, {"default": "自动"}),
                "提示词": ("STRING", {"multiline": True, "default": ""}),
                "自动卸载模型": ("BOOLEAN", {"default": False, "label_on": "是", "label_off": "否"}),
                "最大生成Token数": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 64, "display": "number", "tooltip": "限制生成的最大长度。默认2048，通常足够。设为0则根据文本自动调整（不限制）。"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "批量模式": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("音频",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3TTS"
    DESCRIPTION = "使用自定义声音（Custom Voice）生成语音，支持预设说话人。\n⚠️ 需要加载带有 'CustomVoice' 的模型（如 Qwen3-TTS-12Hz-1.7B-CustomVoice）。"

    def generate(self, 模型, 文本, 预设说话人, 语言, 提示词, seed=0, 批量模式=False, 自动卸载模型=False, 最大生成Token数=2048):
        model = 模型
        text = normalize_text_input(文本, 批量模式)
        speaker = 预设说话人
        language = 语言
        instruct = 提示词
        max_new_tokens = 最大生成Token数
        
        language = LANGUAGE_MAP.get(language, language)
        if language == "auto":
            language = None
            
        # Handle speaker preset instruction
        # For CustomVoice, the 'speaker' param is the ID/Name used by the model.
        # But we also want to inject the 'instruct' if provided by our preset.
        
        user_instruct = (instruct or "").strip()
        if user_instruct:
            instruct = user_instruct
            print(f"User provided manual instruct, overriding preset for '{speaker}'.")
        else:
            if speaker in self.SPEAKER_PRESETS:
                preset_instruct = self.SPEAKER_PRESETS[speaker]
                if preset_instruct:
                    instruct = preset_instruct
                    print(f"Using preset instruct for '{speaker}': {instruct}")
            if not (instruct or "").strip():
                instruct = None
            
        print(f"Generating Custom Voice: {speaker}...")
        
        # Check if speaker is valid? 
        # model.get_supported_speakers()
        
        kwargs = {}
        if max_new_tokens > 0:
            kwargs["max_new_tokens"] = max_new_tokens
            
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        np_state = np.random.get_state()
        py_state = random.getstate()
        
        apply_seed(seed)

        try:
            outputs, sample_rate = model.generate_custom_voice(
                text=text,
                speaker=speaker,
                instruct=instruct,
                language=language,
                **kwargs
            )
        except (ValueError, AttributeError) as e:
            error_msg = str(e).lower()
            if "does not support generate_custom_voice" in error_msg or "generate_custom_voice" in error_msg or "has no attribute" in error_msg:
                raise ValueError(
                    "【Qwen3TTS Error】您当前加载的模型不支持'自定义声音(Custom Voice)'功能。\n"
                    "请在加载器中选择带有 'CustomVoice' 的模型：\n"
                    "  - Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice\n"
                    "  - Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice\n\n"
                    "[Qwen3TTS Error] The loaded model does not support 'Custom Voice'.\n"
                    "Please select a 'CustomVoice' model in the Loader:\n"
                    "  - Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice\n"
                    "  - Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
                ) from e
            raise e
        finally:
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            torch.set_rng_state(rng_state)
            np.random.set_state(np_state)
            random.setstate(py_state)
            if 自动卸载模型:
                print("Unloading model to CPU...")
                if hasattr(model, "model"):
                    model.model.to("cpu")
                if hasattr(model, "device"):
                    model.device = torch.device("cpu")
                torch.cuda.empty_cache()
                gc.collect()
        
        full_audio = np.concatenate(outputs)
        
        # ComfyUI AUDIO format expects {'waveform': tensor, 'sample_rate': int}
        # waveform shape should be [batch_size, channels, samples]
        
        tensor_audio = torch.from_numpy(full_audio).float()
        
        if tensor_audio.ndim == 1:
            # [samples] -> [1, 1, samples]
            tensor_audio = tensor_audio.unsqueeze(0).unsqueeze(0)
        elif tensor_audio.ndim == 2:
            if tensor_audio.shape[0] > tensor_audio.shape[1]: 
                # Likely [samples, channels] -> [1, channels, samples]
                tensor_audio = tensor_audio.permute(1, 0).unsqueeze(0)
            else:
                # Likely [channels, samples] -> [1, channels, samples]
                tensor_audio = tensor_audio.unsqueeze(0)
                
        return ({"waveform": tensor_audio, "sample_rate": sample_rate},)

NODE_CLASS_MAPPINGS = {
    "Qwen3TTSModelLoader": Qwen3TTSModelLoader,
    "Qwen3TTSVoiceDesign": Qwen3TTSVoiceDesign,
    "Qwen3TTSVoiceClone": Qwen3TTSVoiceClone,
    "Qwen3TTSVoiceClonePromptSave": Qwen3TTSVoiceClonePromptSave,
    "Qwen3TTSRolePresetsInput": Qwen3TTSRolePresetsInput,
    "Qwen3TTSPromptSix": Qwen3TTSPromptSix,
    "Qwen3TTSPresetVoiceLoad": Qwen3TTSPresetVoiceLoad,
    "Qwen3TTSDialogueSynthesis": Qwen3TTSDialogueSynthesis,
    "Qwen3TTSVoiceCloneFromPrompt": Qwen3TTSVoiceCloneFromPrompt,
    "Qwen3TTSCustomVoice": Qwen3TTSCustomVoice,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3TTSModelLoader": "Qwen3 TTS 模型加载",
    "Qwen3TTSVoiceDesign": "Qwen3 TTS 声音设计",
    "Qwen3TTSVoiceClone": "Qwen3 TTS 声音克隆",
    "Qwen3TTSVoiceClonePromptSave": "Qwen3 TTS 角色预设保存",
    "Qwen3TTSRolePresetsInput": "Qwen3 TTS 角色预设输入",
    "Qwen3TTSPromptSix": "Qwen3 TTS 提示词",
    "Qwen3TTSPresetVoiceLoad": "Qwen3 TTS 角色预设选择",
    "Qwen3TTSDialogueSynthesis": "Qwen3 TTS 多角色对话合成",
    "Qwen3TTSVoiceCloneFromPrompt": "Qwen3 TTS 语音克隆(角色预设)",
    "Qwen3TTSCustomVoice": "Qwen3 TTS 自定义声音",
}
