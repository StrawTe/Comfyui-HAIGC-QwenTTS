import os
import sys
import json
import re
import torch
import random
import numpy as np
import folder_paths
import gc
import shutil
import librosa
from comfy.utils import ProgressBar
try:
    from transformers import LogitsProcessor, LogitsProcessorList
except ImportError:
    # Define dummy classes if transformers is not available (though unlikely in ComfyUI)
    class LogitsProcessor:
        pass
    class LogitsProcessorList(list):
        pass

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

class ProgressLogitsProcessor(LogitsProcessor):
    def __init__(self, pbar):
        self.pbar = pbar

    def __call__(self, input_ids, scores):
        self.pbar.update(1)
        return scores

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

def apply_speed_change(wav_np, speed):
    if speed == 1.0 or speed <= 0:
        return wav_np
    try:
        # Ensure float32 for librosa
        if wav_np.dtype != np.float32:
             wav_np = wav_np.astype(np.float32)
        return librosa.effects.time_stretch(wav_np, rate=speed)
    except Exception as e:
        print(f"Time stretch failed: {e}")
        return wav_np

def normalize_text_input(text, batch_mode):
    if isinstance(text, list):
        return text if len(text) > 1 else (text[0] if len(text) == 1 else "")
    if text is None:
        return ""
    raw = str(text)
    stripped = raw.strip()
    if batch_mode:
        if "///" in raw:
            parts = raw.split("///")
            cleaned = [p.strip() for p in parts if p.strip()]
            return cleaned if len(cleaned) > 1 else (cleaned[0] if len(cleaned) == 1 else "")

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
        return _normalize_prompt_dict(prompt)
    if isinstance(prompt, list):
        ref_code = []
        ref_spk = []
        xvec = []
        icl = []
        ref_txt = []
        for it in prompt:
            ref_code.append(_prompt_item_get(it, "ref_code", None))
            ref_spk.append(_prompt_item_get(it, "ref_spk_embedding", None))
            xvec_val = bool(_prompt_item_get(it, "x_vector_only_mode", False))
            icl_val = _prompt_item_get(it, "icl_mode", None)
            if icl_val is None:
                icl_val = not xvec_val
            xvec.append(bool(xvec_val))
            icl.append(bool(icl_val))
            ref_txt.append(_prompt_item_get(it, "ref_text", None))
        
        raw_dict = {
            "ref_code": ref_code,
            "ref_spk_embedding": ref_spk,
            "x_vector_only_mode": xvec,
            "icl_mode": icl,
            "ref_text": ref_txt,
        }
        return _normalize_prompt_dict(raw_dict)
    raise ValueError("Invalid voice clone prompt format.")

def _normalize_prompt_dict(prompt_dict):
    if not isinstance(prompt_dict, dict):
        raise ValueError("Invalid voice clone prompt format.")
    ref_code_list = prompt_dict.get("ref_code", None)
    ref_spk_list = prompt_dict.get("ref_spk_embedding", None)
    xvec_list = prompt_dict.get("x_vector_only_mode", None)
    icl_list = prompt_dict.get("icl_mode", None)
    ref_txt_list = prompt_dict.get("ref_text", None)
    
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
    if ref_txt_list is None:
        ref_txt_list = [None] * len(ref_spk_list)
    if not isinstance(ref_txt_list, list):
        ref_txt_list = [ref_txt_list]

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

    # Validate ICL requirements: disable ICL if ref_code or ref_text is missing
    for i in range(len(ref_spk_list)):
        # Check for missing ref_code or ref_text
        r_code = norm_ref_code[i] if i < len(norm_ref_code) else None
        r_text = ref_txt_list[i] if i < len(ref_txt_list) else None
        
        # If ICL is requested but data is missing, fallback to x-vector only
        if i < len(icl_list) and icl_list[i]:
            if r_code is None or r_text is None:
                icl_list[i] = False
                if i < len(xvec_list):
                    xvec_list[i] = True

    # Ensure all fields are lists (even if single item) for consistent processing
    final_ref_code = norm_ref_code
    final_ref_spk = norm_ref_spk
    final_xvec = [bool(x) for x in xvec_list]
    final_icl = [bool(x) for x in icl_list]
    final_ref_txt = ref_txt_list

    final_dict = {
        "ref_code": final_ref_code,
        "ref_spk_embedding": final_ref_spk,
        "x_vector_only_mode": final_xvec,
        "icl_mode": final_icl,
        "ref_text": final_ref_txt,
    }

    return final_dict

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
    # Add ComfyUI output directory relative to this file
    candidates.append(os.path.abspath(os.path.join(current_dir, "../../output")))
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
    
    # Fallback: ComfyUI output directory relative to this file
    # This ensures we find files even if folder_paths is not fully initialized or behaves unexpectedly
    comfy_out = os.path.abspath(os.path.join(current_dir, "../../output"))
    if os.path.exists(comfy_out):
         search_paths.append(os.path.join(comfy_out, "qwen_tts_presets"))
         search_paths.append(comfy_out)

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
        return load_qwen_model(模型名称, 运行设备, 精度)

def load_qwen_model(model_name, device, precision):
        # Global cache for models
        global GLOBAL_QWEN_MODEL_CACHE
        if 'GLOBAL_QWEN_MODEL_CACHE' not in globals():
            GLOBAL_QWEN_MODEL_CACHE = {}
            
        cache_key = f"{model_name}_{device}_{precision}"
        if cache_key in GLOBAL_QWEN_MODEL_CACHE:
            print(f"Loading Qwen3TTS model from cache: {model_name}")
            return (GLOBAL_QWEN_MODEL_CACHE[cache_key],)

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
        
        # Update cache
        GLOBAL_QWEN_MODEL_CACHE[cache_key] = model
        return (model,)

class Qwen3TTSVoiceDesign:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "模型": ("QWEN3_TTS_MODEL",),
                "文本": ("STRING", {"multiline": True, "default": "Hello, this is a test."}),
                "提示词": ("STRING", {"multiline": True, "default": "A young female voice, energetic and bright."}),
                "语言": (LANGUAGE_OPTIONS, {"default": "自动"}),
                "自动卸载模型": ("BOOLEAN", {"default": False, "label_on": "是", "label_off": "否"}),
                "最大生成Token数": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 64, "display": "number", "tooltip": "限制生成的最大长度。默认2048，通常足够。设为0则根据文本自动调整（不限制）。"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "randomize"}),
                "语速": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1, "display": "number"}),
            },
            "optional": {
                "批量模式": ("BOOLEAN", {"default": False}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05, "display": "number", "tooltip": "核采样 (Top-P)。\n常用范围：0.1 ~ 1.0\n解释：控制生成时选择候选词汇的概率累积阈值。数值越接近 1，候选词汇池越大，生成的内容越多样；数值越小，生成越聚焦。"}),
                "top_k": ("INT", {"default": 50, "min": 10, "max": 100, "step": 1, "display": "number", "tooltip": "Top-K 采样。\n常用范围：10 ~ 100\n解释：限制每次生成时仅从概率最高的前 K 个词汇中选择。K 值越大，生成的随机性越高；K 值越小，生成越稳定。"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.05, "display": "number", "tooltip": "温度系数 (Temperature)。\n常用范围：0.1 ~ 2.0\n解释：调整生成内容的随机性和多样性。数值越低（如 0.1），生成越保守、重复；数值越高（如 2.0），生成越有创意但可能出现逻辑混乱。"}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.05, "display": "number", "tooltip": "重复惩罚 (Repetition Penalty)。\n常用范围：1.0 ~ 2.0\n解释：抑制生成内容中的重复表述。数值越高（如 1.5），越不容易出现重复句子；数值为 1.0 时，无惩罚效果。"}),
                "启用高级采样配置": ("BOOLEAN", {"default": False, "label_on": "开启", "label_off": "关闭", "tooltip": "开启后，上方的高级采样参数（Top-P, Top-K 等）才会生效。关闭时使用模型默认配置。"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("音频",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3TTS"
    DESCRIPTION = "使用声音设计（Voice Design）生成语音。\n需要加载带有 'VoiceDesign' 的模型。"

    def generate(self, 模型, 文本, 提示词, 语言, seed=0, 批量模式=False, 自动卸载模型=False, 最大生成Token数=2048, top_p=0.8, top_k=50, temperature=0.8, repetition_penalty=1.1, 启用高级采样配置=False, 语速=1.0):
        model = 模型
        text = normalize_text_input(文本, 批量模式)
        instruct = 提示词
        language = 语言
        max_new_tokens = 最大生成Token数
        
        language = LANGUAGE_MAP.get(language, language)
        if language == "auto":
            language = None

        kwargs = {}
        if max_new_tokens > 0:
            kwargs["max_new_tokens"] = max_new_tokens
            # Add progress bar
            pbar = ProgressBar(max_new_tokens)
            kwargs["logits_processor"] = LogitsProcessorList([ProgressLogitsProcessor(pbar)])
            
        if 启用高级采样配置:
            kwargs["top_p"] = top_p
            kwargs["top_k"] = top_k
            kwargs["temperature"] = temperature
            kwargs["repetition_penalty"] = repetition_penalty
        else:
            kwargs["top_p"] = None
            kwargs["top_k"] = None
            kwargs["temperature"] = None
            kwargs["repetition_penalty"] = None

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
            if 自动卸载模型:
                print("Unloading model to CPU...")
                if hasattr(model, "model"):
                    model.model.to("cpu")
                if hasattr(model, "device"):
                    model.device = torch.device("cpu")
                torch.cuda.empty_cache()
                gc.collect()

        full_audio = np.concatenate(outputs)
        
        # Apply speed change
        full_audio = apply_speed_change(full_audio, 语速)
        
        tensor_audio = torch.from_numpy(full_audio).float()
        
        if tensor_audio.ndim == 1:
            tensor_audio = tensor_audio.unsqueeze(0).unsqueeze(0)
        elif tensor_audio.ndim == 2:
            if tensor_audio.shape[0] > tensor_audio.shape[1]:
                tensor_audio = tensor_audio.permute(1, 0).unsqueeze(0)
            else:
                tensor_audio = tensor_audio.unsqueeze(0)
        
        return ({"waveform": tensor_audio, "sample_rate": sample_rate},)

class Qwen3TTSVoiceClone:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "模型": ("QWEN3_TTS_MODEL",),
                "参考音频": ("AUDIO",),
                "文本": ("STRING", {"multiline": True, "default": "Hello, I am cloning this voice."}),
                "参考文本": ("STRING", {"multiline": True, "default": ""}),
                "语言": (LANGUAGE_OPTIONS, {"default": "自动"}),
                "自动卸载模型": ("BOOLEAN", {"default": False, "label_on": "是", "label_off": "否"}),
                "最大生成Token数": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 64, "display": "number", "tooltip": "限制生成的最大长度。默认2048，通常足够。设为0则根据文本自动调整（不限制）。"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "randomize"}),
                "语速": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1, "display": "number"}),
            },
            "optional": {
                "批量模式": ("BOOLEAN", {"default": False}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05, "display": "number", "tooltip": "核采样 (Top-P)。\n常用范围：0.1 ~ 1.0\n解释：控制生成时选择候选词汇的概率累积阈值。数值越接近 1，候选词汇池越大，生成的内容越多样；数值越小，生成越聚焦。"}),
                "top_k": ("INT", {"default": 50, "min": 10, "max": 100, "step": 1, "display": "number", "tooltip": "Top-K 采样。\n常用范围：10 ~ 100\n解释：限制每次生成时仅从概率最高的前 K 个词汇中选择。K 值越大，生成的随机性越高；K 值越小，生成越稳定。"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.05, "display": "number", "tooltip": "温度系数 (Temperature)。\n常用范围：0.1 ~ 2.0\n解释：调整生成内容的随机性和多样性。数值越低（如 0.1），生成越保守、重复；数值越高（如 2.0），生成越有创意但可能出现逻辑混乱。"}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.05, "display": "number", "tooltip": "重复惩罚 (Repetition Penalty)。\n常用范围：1.0 ~ 2.0\n解释：抑制生成内容中的重复表述。数值越高（如 1.5），越不容易出现重复句子；数值为 1.0 时，无惩罚效果。"}),
                "启用高级采样配置": ("BOOLEAN", {"default": False, "label_on": "开启", "label_off": "关闭", "tooltip": "开启后，上方的高级采样参数（Top-P, Top-K 等）才会生效。关闭时使用模型默认配置。"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "QWEN3_TTS_VOICE_PROMPT")
    RETURN_NAMES = ("音频", "角色预设")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "generate"
    CATEGORY = "Qwen3TTS"
    DESCRIPTION = "使用声音克隆（Voice Clone）生成语音，基于参考音频。\n⚠️ 需要加载带有 'Base' 的模型（如 Qwen3-TTS-12Hz-1.7B-Base）。"

    def generate(self, 模型, 文本, 参考音频, 参考文本, 语言, seed=0, 批量模式=False, 自动卸载模型=False, 最大生成Token数=2048, top_p=0.8, top_k=50, temperature=0.8, repetition_penalty=1.1, 启用高级采样配置=False, 语速=1.0):
        model = 模型
        
        # --- Batch Processing Setup ---
        ref_wav_tensor = 参考音频['waveform'] # [B, C, S]
        ref_sr = 参考音频['sample_rate']
        batch_size = ref_wav_tensor.shape[0]
        
        ref_texts = []
        if batch_size > 1:
            # Parse formatted reference text: "音频1：内容1\n音频2：内容2"
            lines = (参考文本 or "").strip().split('\n')
            parsed = {}
            pattern = re.compile(r"^音频(\d+)[：:](.*)$")
            for line in lines:
                m = pattern.match(line.strip())
                if m:
                    idx = int(m.group(1)) - 1
                    content = m.group(2).strip()
                    parsed[idx] = content
            
            if parsed:
                for i in range(batch_size):
                    ref_texts.append(parsed.get(i, None))
            else:
                # Fallback: if lines match batch size, use lines
                non_empty_lines = [l.strip() for l in lines if l.strip()]
                if len(non_empty_lines) == batch_size:
                    ref_texts = non_empty_lines
                else:
                    ref_texts = [参考文本] * batch_size
        else:
            ref_texts = [参考文本]

        # Target Text Processing
        target_texts = []
        raw_text = normalize_text_input(文本, 批量模式) 
        
        # RH: Auto-split text if batch_size > 1 and input is multiline string
        if not isinstance(raw_text, list) and batch_size > 1 and isinstance(raw_text, str):
             lines = [line.strip() for line in raw_text.strip().splitlines() if line.strip()]
             if len(lines) == batch_size:
                 print(f"Auto-split text into {len(lines)} parts to match audio batch size.")
                 raw_text = lines
        
        if isinstance(raw_text, list):
             if len(raw_text) == batch_size:
                 target_texts = raw_text
             else:
                 target_texts = [raw_text[i % len(raw_text)] for i in range(batch_size)]
        else:
             target_texts = [raw_text] * batch_size

        language = LANGUAGE_MAP.get(语言, 语言)
        if language == "auto": language = None
        
        kwargs = {}
        if 最大生成Token数 > 0: 
            kwargs["max_new_tokens"] = 最大生成Token数
            # Add progress bar
            pbar = ProgressBar(最大生成Token数)
            kwargs["logits_processor"] = LogitsProcessorList([ProgressLogitsProcessor(pbar)])
        
        if 启用高级采样配置:
            kwargs.update({"top_p": top_p, "top_k": top_k, "temperature": temperature, "repetition_penalty": repetition_penalty})
        else:
            kwargs.update({"top_p": None, "top_k": None, "temperature": None, "repetition_penalty": None})
        
        apply_seed(seed)
        
        # Prepare inputs for batch generation
        ref_audio_list = []
        for i in range(batch_size):
            single_wav = ref_wav_tensor[i]
            if single_wav.shape[0] > 1:
                wav_np = single_wav.mean(dim=0).cpu().numpy()
            else:
                wav_np = single_wav.squeeze().cpu().numpy()
            
            # Apply speed change to reference audio
            wav_np = apply_speed_change(wav_np, 语速)
            
            ref_audio_list.append((wav_np, ref_sr))

        final_prompt_items = []
        
        print(f"Generating Voice Clone... Batch Size: {batch_size}")

        try:
            # 1. Create Prompts (Iterative to allow fallback)
            prompt_items_batch = []
            for i in range(batch_size):
                curr_ref_text = ref_texts[i]
                curr_ref_audio_tuple = ref_audio_list[i]
                
                if curr_ref_text and curr_ref_text.strip():
                    curr_ref_text = curr_ref_text.strip()
                    curr_x_vec_mode = False
                else:
                    curr_ref_text = None
                    curr_x_vec_mode = True
                
                try:
                    p_items = model.create_voice_clone_prompt(
                        ref_audio=curr_ref_audio_tuple,
                        ref_text=curr_ref_text,
                        x_vector_only_mode=bool(curr_x_vec_mode),
                    )
                    prompt_items_batch.extend(p_items)
                except Exception as e:
                    print(f"Warning: Failed to create prompt for batch {i}: {e}")
                    # Fallback to speaker encoder only
                    try:
                         fallback_items = _build_prompt_from_speaker_encoder(model, curr_ref_audio_tuple)
                         prompt_items_batch.extend(fallback_items)
                    except:
                         raise ValueError(f"Failed to create prompt for batch {i} and fallback failed.")

            final_prompt_items = prompt_items_batch

            # 2. Batch Generate
            wavs_out, sample_rate = model.generate_voice_clone(
                text=target_texts,
                language=language,
                voice_clone_prompt=final_prompt_items,
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
            if 自动卸载模型:
                print("Unloading model to CPU...")
                if hasattr(model, "model"):
                    model.model.to("cpu")
                if hasattr(model, "device"):
                    model.device = torch.device("cpu")
                torch.cuda.empty_cache()
                gc.collect()
        
        # 3. Process Outputs (No padding, Variable Lengths)
        final_audio_list = []
        for wav in wavs_out:
            t = torch.from_numpy(wav).float()
            # Ensure [1, 1, Samples]
            if t.ndim == 1:
                t = t.unsqueeze(0).unsqueeze(0)
            elif t.ndim == 2:
                 t = t.unsqueeze(0)
            
            final_audio_list.append({"waveform": t, "sample_rate": sample_rate})
            
        return (final_audio_list, final_prompt_items)

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
        if 角色预设 is None:
             raise ValueError("Role preset input is None. Please ensure the upstream node has generated a valid preset.")

        # --- Check for Multi-Role Preset Dict (from Master Node Dialogue Mode) ---
        # Format: { "RoleName": { "ref_spk_embedding": ... }, ... }
        is_multi_role = False
        if isinstance(角色预设, dict) and "ref_spk_embedding" not in 角色预设 and "ref_code" not in 角色预设:
             # It's likely a container dict. Let's verify if values look like presets.
             if len(角色预设) > 0:
                 # Check the first value to see if it's a preset-like object (dict or list)
                 first_val = next(iter(角色预设.values()))
                 if isinstance(first_val, (dict, list)): 
                     is_multi_role = True
             else:
                 raise ValueError("Role preset input is an empty dictionary.")

        # --- Resolve Save Directory (Common) ---
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
            save_dir = os.path.join(current_dir, "output")
            
        os.makedirs(save_dir, exist_ok=True)

        # --- Multi-Role Processing ---
        if is_multi_role:
            saved_paths = []
            print(f"[Qwen3TTS Save] Detected Multi-Role Presets ({len(角色预设)} roles). Saving individually...")
            
            for r_name, p_data in 角色预设.items():
                try:
                    # Validate/Normalize individual preset
                    p_dict = _to_prompt_dict(p_data)
                    p_dict = _normalize_prompt_dict(p_dict)
                    
                    # Determine filename: Use key from dict
                    filename = r_name
                    if not filename.lower().endswith(".pt"):
                        filename = f"{filename}.pt"
                    
                    out_path = os.path.join(save_dir, filename)
                    
                    # Save
                    torch.save({"voice_clone_prompt": p_dict, "role_name": r_name}, out_path)
                    saved_paths.append(out_path)
                    print(f"  - Saved: {out_path}")
                    
                except Exception as e:
                    print(f"  - Failed to save role '{r_name}': {e}")
            
            if not saved_paths:
                raise ValueError("Failed to save any roles from the multi-role input.")
            
            # Return the directory or the first file path
            return (saved_paths[0], 角色预设)

        # --- Single Role Processing (Legacy) ---
        try:
            prompt_dict = _to_prompt_dict(角色预设)
            prompt_dict = _normalize_prompt_dict(prompt_dict)
        except ValueError as e:
            error_msg = str(e).lower()
            if "empty" in error_msg or "missing" in error_msg or "none" in error_msg:
                raise ValueError(
                    "角色预设为空或格式无效，无法保存。\n"
                    "Role preset is empty or invalid.\n"
                    f"Details: {str(e)}"
                )
            raise

        role_name = (角色名 or "").strip()
        if not role_name:
            raise ValueError("【Qwen3TTS Error】角色名不能为空 (Role Name is required).")

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
                "自定义预设路径": ("STRING", {"default": "", "tooltip": "指定查找角色预设文件的自定义文件夹路径。优先在此路径查找。未指定则使用默认路径。"}),
            }
        }

    RETURN_TYPES = ("QWEN3_TTS_VOICE_PROMPT",)
    RETURN_NAMES = ("角色预设",)
    FUNCTION = "load_prompt"
    CATEGORY = "Qwen3TTS"
    DESCRIPTION = "从下拉列表中选择已保存的角色预设文件(.pt)。\n也可以通过'手动输入文件名'直接指定文件，无需刷新浏览器。"

    def load_prompt(self, 预设文件, 手动输入文件名="", 自定义预设路径=""):
        filename = (手动输入文件名 or "").strip()
        if not filename:
            filename = 预设文件
        
        if not filename or filename == "None":
            raise ValueError("No voice prompt file selected.")
        
        path = None
        custom_dir = (自定义预设路径 or "").strip()
        if custom_dir:
            # Try to find in custom directory first
            # Handle potential absolute path in filename implicitly by os.path.join behavior (if absolute, custom_dir is ignored)
            # But better to be explicit about relative check
            check_path = os.path.join(custom_dir, filename)
            if os.path.exists(check_path):
                path = check_path
            elif not filename.lower().endswith(".pt"):
                check_path_pt = f"{check_path}.pt"
                if os.path.exists(check_path_pt):
                    path = check_path_pt
        
        if not path:
            # Fallback to default resolution logic
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
        
        # RH: Fix "Boolean value of Tensor with more than one value is ambiguous"
        is_empty = False
        if ref_spk_list is None:
            is_empty = True
        elif isinstance(ref_spk_list, list) and len(ref_spk_list) == 0:
            is_empty = True
        elif isinstance(ref_spk_list, torch.Tensor) and ref_spk_list.numel() == 0:
            is_empty = True
            
        if is_empty:
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
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "randomize"}),
                "语速": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1, "display": "number"}),
            },
            "optional": {
                "语言": (LANGUAGE_OPTIONS, {"default": "自动"}),
                "角色映射(可选)": ("STRING", {"multiline": True, "default": ""}),
                "角色预设": ("QWEN3_TTS_ROLE_PRESETS",),
                "启用停顿控制": ("BOOLEAN", {"default": True, "label_on": "开启", "label_off": "关闭"}),
                "自动卸载模型": ("BOOLEAN", {"default": False, "label_on": "是", "label_off": "否"}),
                "最大生成Token数": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 64, "display": "number", "tooltip": "限制生成的最大长度。默认2048，通常足够。设为0则根据文本自动调整（不限制）。"}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05, "display": "number", "tooltip": "核采样 (Top-P)。\n常用范围：0.1 ~ 1.0\n解释：控制生成时选择候选词汇的概率累积阈值。数值越接近 1，候选词汇池越大，生成的内容越多样；数值越小，生成越聚焦。"}),
                "top_k": ("INT", {"default": 50, "min": 10, "max": 100, "step": 1, "display": "number", "tooltip": "Top-K 采样。\n常用范围：10 ~ 100\n解释：限制每次生成时仅从概率最高的前 K 个词汇中选择。K 值越大，生成的随机性越高；K 值越小，生成越稳定。"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.05, "display": "number", "tooltip": "温度系数 (Temperature)。\n常用范围：0.1 ~ 2.0\n解释：调整生成内容的随机性和多样性。数值越低（如 0.1），生成越保守、重复；数值越高（如 2.0），生成越有创意但可能出现逻辑混乱。"}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.05, "display": "number", "tooltip": "重复惩罚 (Repetition Penalty)。\n常用范围：1.0 ~ 2.0\n解释：抑制生成内容中的重复表述。数值越高（如 1.5），越不容易出现重复句子；数值为 1.0 时，无惩罚效果。"}),
                "启用高级采样配置": ("BOOLEAN", {"default": False, "label_on": "开启", "label_off": "关闭", "tooltip": "开启后，上方的高级采样参数（Top-P, Top-K 等）才会生效。关闭时使用模型默认配置。"}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("音频",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3TTS"
    DESCRIPTION = "输入多角色对白文本，自动将角色名映射为同名 .pt 预设并合成整段音频，可用“角色名=文件名”覆盖，且优先使用角色预设输入端口。支持使用 '=Ns' (如 =2s) 添加静音停顿。"

    def generate(self, 模型, 对白文本, 语言, 角色映射=None, 角色预设=None, 启用停顿控制=True, seed=0, 自动卸载模型=False, 最大生成Token数=2048, top_p=0.8, top_k=50, temperature=0.8, repetition_penalty=1.1, 启用高级采样配置=False, 语速=1.0, **kwargs):
        model = 模型
        
        apply_seed(seed)
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
            
            # Remove voice description if present, e.g. Role(Description) -> Role
            r_name_clean = role.replace("（", "(").replace("）", ")")
            if "(" in r_name_clean and r_name_clean.endswith(")"):
                start_idx = r_name_clean.find("(")
                end_idx = r_name_clean.rfind(")")
                if start_idx < end_idx:
                    role = r_name_clean[:start_idx].strip()

            text = text.strip()
            if not role or not text:
                raise ValueError(f"【Qwen3TTS Error】对白格式错误: {raw}")
            segments.append((role, text))

        for role, _ in segments:
            if role not in role_to_prompt:
                # 1. Check in input_presets (memory/passed in dict)
                if role in input_presets:
                    role_to_prompt[role] = _normalize_prompt_dict(input_presets[role])
                else:
                    # 2. Try to load from file
                    filename = role_override.get(role, f"{role}.pt")
                    try:
                        resolved = _resolve_prompt_path(filename)
                        payload = torch.load(resolved, map_location="cpu", weights_only=True)
                        if isinstance(payload, dict) and "voice_clone_prompt" in payload:
                            prompt_dict = payload["voice_clone_prompt"]
                        else:
                            prompt_dict = payload
                        role_to_prompt[role] = _normalize_prompt_dict(prompt_dict)
                    except (FileNotFoundError, ValueError, Exception) as e:
                         # 3. Double check input_presets (Just in case)
                         if role in input_presets:
                              role_to_prompt[role] = _normalize_prompt_dict(input_presets[role])
                         else:
                              # Truly missing
                              raise ValueError(
                                f"【Qwen3TTS Error】未找到角色预设文件: {role} -> {filename}\n"
                                f"请确保该角色有对应的 .pt 预设文件，或者在 Master 节点启用了自动设计功能。\n{e}"
                            )

        kwargs = {}
        if max_new_tokens > 0:
            kwargs["max_new_tokens"] = max_new_tokens
            
        if 启用高级采样配置:
            kwargs["top_p"] = top_p
            kwargs["top_k"] = top_k
            kwargs["temperature"] = temperature
            kwargs["repetition_penalty"] = repetition_penalty
        else:
            kwargs["top_p"] = None
            kwargs["top_k"] = None
            kwargs["temperature"] = None
            kwargs["repetition_penalty"] = None

        apply_seed(seed)

        try:
            wavs_all = []
            sample_rate = None
            
            # Default sample rate if not yet known (Qwen3 usually 24k, but we try to get from first gen)
            # If the very first segment is silence, we might need a default.
            # But usually we can expect at least one speech segment.
            # If everything is silence, we default to 24000.
            DEFAULT_SR = 24000

            # 1. Collect all segments to process
            all_parts = []
            
            for role, text in segments:
                if 启用停顿控制:
                    pattern = r'=(\d+(?:\.\d+)?)s'
                    parts = re.split(pattern, text)
                    idx = 0
                    while idx < len(parts):
                        part_text = parts[idx]
                        if part_text and part_text.strip():
                            all_parts.append({
                                "type": "text", 
                                "content": part_text.strip(),
                                "role": role
                            })
                        
                        if idx + 1 < len(parts):
                            duration_str = parts[idx+1]
                            try:
                                duration = float(duration_str)
                                all_parts.append({"type": "silence", "duration": duration})
                            except ValueError:
                                pass
                            idx += 2
                        else:
                            idx += 1
                else:
                    if text and text.strip():
                        all_parts.append({
                            "type": "text",
                            "content": text.strip(),
                            "role": role
                        })

            # 2. Prepare batch for text segments
            batch_texts = []
            batch_prompts = []
            batch_indices = [] # to map back to all_parts
            
            for i, part in enumerate(all_parts):
                if part["type"] == "text":
                    batch_texts.append(part["content"])
                    batch_prompts.append(role_to_prompt[part["role"]])
                    batch_indices.append(i)

            # 3. Batch generate
            if batch_texts:
                print(f"Batch generating {len(batch_texts)} segments...")
                # We can reuse generate_voice_clone with lists
                # It returns a list of numpy arrays (wavs)
                
                # Note: Qwen3TTSModel.generate_voice_clone handles list inputs
                # voice_clone_prompt can also be a list of prompts corresponding to texts
                
                wavs, sr = model.generate_voice_clone(
                    text=batch_texts,
                    language=language,
                    voice_clone_prompt=batch_prompts,
                    **kwargs
                )
                
                if sample_rate is None:
                    sample_rate = sr
                
                # Map back to all_parts
                for i, wav in enumerate(wavs):
                    original_idx = batch_indices[i]
                    all_parts[original_idx]["wav"] = wav
            
            # 4. Reconstruct full audio
            wavs_all = []
            current_sr = sample_rate if sample_rate is not None else DEFAULT_SR
            
            for part in all_parts:
                if part["type"] == "text":
                    if "wav" in part:
                        wavs_all.append(part["wav"])
                elif part["type"] == "silence":
                    duration = part["duration"]
                    if duration <= 0:
                        continue
                    num_samples = int(duration * current_sr)
                    if num_samples > 0:
                        silence_wav = np.zeros(num_samples, dtype=np.float32)
                        wavs_all.append(silence_wav)


        finally:
            if 自动卸载模型:
                print("Unloading model to CPU...")
                if hasattr(model, "model"):
                    model.model.to("cpu")
                if hasattr(model, "device"):
                    model.device = torch.device("cpu")
                torch.cuda.empty_cache()
                gc.collect()

        if not wavs_all:
            # If user only input silence, we might have silence segments but no text segments generated?
            # If so, wavs_all has silence arrays.
            # But if dialogue_text was empty, we raised error earlier.
            # If dialogue_text was "=2s", we have silence.
            pass

        if not wavs_all:
             raise ValueError("【Qwen3TTS Error】未能生成任何音频。")

        full_audio = np.concatenate(wavs_all)
        
        # Apply speed change
        full_audio = apply_speed_change(full_audio, 语速)
        
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
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "randomize"}),
                "语速": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1, "display": "number"}),
            },
            "optional": {
                "语言": (LANGUAGE_OPTIONS, {"default": "自动"}),
                "批量模式": ("BOOLEAN", {"default": False}),
                "自动卸载模型": ("BOOLEAN", {"default": False, "label_on": "是", "label_off": "否"}),
                "最大生成Token数": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 64, "display": "number", "tooltip": "限制生成的最大长度。默认2048，通常足够。设为0则根据文本自动调整（不限制）。"}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05, "display": "number", "tooltip": "核采样 (Top-P)。\n常用范围：0.1 ~ 1.0\n解释：控制生成时选择候选词汇的概率累积阈值。数值越接近 1，候选词汇池越大，生成的内容越多样；数值越小，生成越聚焦。"}),
                "top_k": ("INT", {"default": 50, "min": 10, "max": 100, "step": 1, "display": "number", "tooltip": "Top-K 采样。\n常用范围：10 ~ 100\n解释：限制每次生成时仅从概率最高的前 K 个词汇中选择。K 值越大，生成的随机性越高；K 值越小，生成越稳定。"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.05, "display": "number", "tooltip": "温度系数 (Temperature)。\n常用范围：0.1 ~ 2.0\n解释：调整生成内容的随机性和多样性。数值越低（如 0.1），生成越保守、重复；数值越高（如 2.0），生成越有创意但可能出现逻辑混乱。"}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.05, "display": "number", "tooltip": "重复惩罚 (Repetition Penalty)。\n常用范围：1.0 ~ 2.0\n解释：抑制生成内容中的重复表述。数值越高（如 1.5），越不容易出现重复句子；数值为 1.0 时，无惩罚效果。"}),
                "启用高级采样配置": ("BOOLEAN", {"default": False, "label_on": "开启", "label_off": "关闭", "tooltip": "开启后，上方的高级采样参数（Top-P, Top-K 等）才会生效。关闭时使用模型默认配置。"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("音频",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3TTS"
    DESCRIPTION = "使用可复用语音克隆提示生成语音，无需重复提取参考特征。"

    def generate(self, 模型, 文本, 角色预设, 语言, seed=0, 批量模式=False, 自动卸载模型=False, 最大生成Token数=2048, top_p=0.8, top_k=50, temperature=0.8, repetition_penalty=1.1, 启用高级采样配置=False, 语速=1.0):
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
            
        if 启用高级采样配置:
            kwargs["top_p"] = top_p
            kwargs["top_k"] = top_k
            kwargs["temperature"] = temperature
            kwargs["repetition_penalty"] = repetition_penalty
        else:
            kwargs["top_p"] = None
            kwargs["top_k"] = None
            kwargs["temperature"] = None
            kwargs["repetition_penalty"] = None

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
            if 自动卸载模型:
                print("Unloading model to CPU...")
                if hasattr(model, "model"):
                    model.model.to("cpu")
                if hasattr(model, "device"):
                    model.device = torch.device("cpu")
                torch.cuda.empty_cache()
                gc.collect()

        full_audio = np.concatenate(outputs)

        # Apply speed change
        full_audio = apply_speed_change(full_audio, 语速)

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
                "语速": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1, "display": "number"}),
            },
            "optional": {
                "语言": (LANGUAGE_OPTIONS, {"default": "自动"}),
                "提示词": ("STRING", {"multiline": True, "default": ""}),
                "自动卸载模型": ("BOOLEAN", {"default": False, "label_on": "是", "label_off": "否"}),
                "最大生成Token数": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 64, "display": "number", "tooltip": "限制生成的最大长度。默认2048，通常足够。设为0则根据文本自动调整（不限制）。"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "randomize"}),
                "批量模式": ("BOOLEAN", {"default": False}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05, "display": "number", "tooltip": "核采样 (Top-P)。\n常用范围：0.1 ~ 1.0\n解释：控制生成时选择候选词汇的概率累积阈值。数值越接近 1，候选词汇池越大，生成的内容越多样；数值越小，生成越聚焦。"}),
                "top_k": ("INT", {"default": 50, "min": 10, "max": 100, "step": 1, "display": "number", "tooltip": "Top-K 采样。\n常用范围：10 ~ 100\n解释：限制每次生成时仅从概率最高的前 K 个词汇中选择。K 值越大，生成的随机性越高；K 值越小，生成越稳定。"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.05, "display": "number", "tooltip": "温度系数 (Temperature)。\n常用范围：0.1 ~ 2.0\n解释：调整生成内容的随机性和多样性。数值越低（如 0.1），生成越保守、重复；数值越高（如 2.0），生成越有创意但可能出现逻辑混乱。"}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.05, "display": "number", "tooltip": "重复惩罚 (Repetition Penalty)。\n常用范围：1.0 ~ 2.0\n解释：抑制生成内容中的重复表述。数值越高（如 1.5），越不容易出现重复句子；数值为 1.0 时，无惩罚效果。"}),
                "启用高级采样配置": ("BOOLEAN", {"default": False, "label_on": "开启", "label_off": "关闭", "tooltip": "开启后，上方的高级采样参数（Top-P, Top-K 等）才会生效。关闭时使用模型默认配置。"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("音频",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3TTS"
    DESCRIPTION = "使用自定义声音（Custom Voice）生成语音，支持预设说话人。\n⚠️ 需要加载带有 'CustomVoice' 的模型（如 Qwen3-TTS-12Hz-1.7B-CustomVoice）。"

    def generate(self, 模型, 文本, 预设说话人, 语言, 提示词, seed=0, 批量模式=False, 自动卸载模型=False, 最大生成Token数=2048, top_p=0.8, top_k=50, temperature=0.8, repetition_penalty=1.1, 启用高级采样配置=False, 语速=1.0):
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
            # Add progress bar
            pbar = ProgressBar(max_new_tokens)
            kwargs["logits_processor"] = LogitsProcessorList([ProgressLogitsProcessor(pbar)])
            
        if 启用高级采样配置:
            kwargs["top_p"] = top_p
            kwargs["top_k"] = top_k
            kwargs["temperature"] = temperature
            kwargs["repetition_penalty"] = repetition_penalty
        else:
            kwargs["top_p"] = None
            kwargs["top_k"] = None
            kwargs["temperature"] = None
            kwargs["repetition_penalty"] = None

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
            if 自动卸载模型:
                print("Unloading model to CPU...")
                if hasattr(model, "model"):
                    model.model.to("cpu")
                if hasattr(model, "device"):
                    model.device = torch.device("cpu")
                torch.cuda.empty_cache()
                gc.collect()
        
        full_audio = np.concatenate(outputs)
        
        # Apply speed change
        full_audio = apply_speed_change(full_audio, 语速)
        
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

from .constants import *


class Qwen3TTSVoiceDescription:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "基础音色": (AGE_GENDER_DESC, {"default": "不选择"}),
                "音色质感": (TIMBRE_DESC, {"default": "不选择"}),
                "情感状态": (EMOTION_DESC, {"default": "不选择"}),
                "风格场景": (STYLE_DESC, {"default": "不选择"}),
            },
            "optional": {
                "上一个提示词": ("STRING", {"forceInput": True}),
                "自定义描述": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("提示词",)
    FUNCTION = "process"
    CATEGORY = "Qwen3TTS"
    DESCRIPTION = "构建语音音色描述提示词，支持无限串联组合。"

    def process(self, 基础音色, 音色质感, 情感状态, 风格场景, 上一个提示词=None, 自定义描述=""):
        parts = []
        if 上一个提示词 and 上一个提示词.strip():
            parts.append(上一个提示词.strip())
        
        current_parts = []
        
        def _get_en(text):
            return VOICE_DESC_MAP.get(text, text)

        if 基础音色 and 基础音色 != "不选择":
            current_parts.append(_get_en(基础音色))
        
        for tex in [音色质感, 情感状态, 风格场景]:
            if tex and tex != "不选择":
                current_parts.append(_get_en(tex))
                
        if 自定义描述 and 自定义描述.strip():
            current_parts.append(自定义描述.strip())
            
        if current_parts:
            # Use ", " separator for descriptors as per official example style
            combined_current = ", ".join(current_parts)
            # Add period at the end if not present (optional, but good for sentence structure)
            if not combined_current.endswith(".") and not combined_current.endswith("。"):
                 combined_current += "."
            parts.append(combined_current)
            
        final_prompt = " ".join(parts)
        return (final_prompt,)

class Qwen3TTSBatchVoiceCloneInput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "音频1": ("AUDIO",),
                "内容1": ("STRING", {"multiline": True}),
            },
            "optional": {
                "音频2": ("AUDIO",),
                "内容2": ("STRING", {"multiline": True}),
                "音频3": ("AUDIO",),
                "内容3": ("STRING", {"multiline": True}),
                "音频4": ("AUDIO",),
                "内容4": ("STRING", {"multiline": True}),
                "音频5": ("AUDIO",),
                "内容5": ("STRING", {"multiline": True}),
                "音频6": ("AUDIO",),
                "内容6": ("STRING", {"multiline": True}),
                "已有批量音频": ("AUDIO",),
                "已有批量文本": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("批量音频", "格式化文本")
    FUNCTION = "generate_batch"
    CATEGORY = "Qwen3TTS"
    DESCRIPTION = "输入6组参考音频-内容对，支持串联扩展。输出批量音频和格式化文本，连接到Master节点的参考音频和文本端口。"

    def generate_batch(self, 音频1, 内容1, 音频2=None, 内容2=None, 音频3=None, 内容3=None, 音频4=None, 内容4=None, 音频5=None, 内容5=None, 音频6=None, 内容6=None, 已有批量音频=None, 已有批量文本=None):
        audio_list = []
        text_list = []
        
        # 1. Add existing batch
        current_idx = 1
        if 已有批量音频 is not None:
             # If existing batch provided, we need to decompose it? 
             # But we can't easily decompose the text if it's already formatted.
             # Wait, if we chain, we just append to the batch tensor and append to the text string.
             # We assume '已有批量文本' is already formatted as "音频1: ...".
             # We need to find the last index in the existing text to continue numbering.
             
             wavs = 已有批量音频['waveform'] # [B, C, S]
             sr = 已有批量音频['sample_rate']
             
             # If we just want to output a combined batch, we can collect everything first.
             # But for numbering, we need to know how many were before.
             if 已有批量文本:
                 # Try to find the last "音频N：" pattern
                 matches = re.findall(r"音频(\d+)：", 已有批量文本)
                 if matches:
                     current_idx = int(matches[-1]) + 1
            
        inputs = [
            (音频1, 内容1), (音频2, 内容2), (音频3, 内容3),
            (音频4, 内容4), (音频5, 内容5), (音频6, 内容6)
        ]
        
        new_audios = [] # List of tensors [C, S]
        new_sr = None
        
        for i, (aud, txt) in enumerate(inputs):
            if aud is not None and txt and txt.strip():
                if new_sr is None:
                    new_sr = aud['sample_rate']
                elif new_sr != aud['sample_rate']:
                    # Simple check, real resampling is hard here without loading model
                    pass 
                
                wav = aud['waveform'] # [B, C, S]
                # Take all items in batch (usually 1)
                for b in range(wav.shape[0]):
                    new_audios.append(wav[b])
                    # No prefix, just clean content. Replace newlines with space to ensure 1 line per entry.
                    clean_txt = txt.strip().replace('\n', ' ')
                    text_list.append(clean_txt)
                    current_idx += 1

        if not new_audios and 已有批量音频 is None:
             raise ValueError("未提供任何有效的音频和内容。")

        # Combine Audios
        final_wav_list = []
        final_sr = new_sr if new_sr else (已有批量音频['sample_rate'] if 已有批量音频 else 24000)
        
        if 已有批量音频 is not None:
            old_wavs = 已有批量音频['waveform']
            for b in range(old_wavs.shape[0]):
                final_wav_list.append(old_wavs[b])
        
        final_wav_list.extend(new_audios)
        
        if not final_wav_list:
             raise ValueError("结果音频列表为空。")

        # Normalize Channels (Fix for stack error)
        # Ensure all tensors have the same number of channels (use max found)
        max_channels = max(w.shape[0] for w in final_wav_list)
        normalized_wav_list = []
        for w in final_wav_list:
            current_channels = w.shape[0]
            if current_channels < max_channels:
                # Expand channels (e.g., Mono to Stereo)
                # [1, S] -> [2, S] via repeat
                repeats = max_channels // current_channels
                w = w.repeat(repeats, 1)
                # If division wasn't exact (e.g. 2 to 3), we might need more complex logic, 
                # but ComfyUI audio is usually 1 or 2 channels.
                # If we have 1 and 2, repeats=2.
            normalized_wav_list.append(w)
        
        final_wav_list = normalized_wav_list

        # Pad Audios to max length
        max_len = max(w.shape[-1] for w in final_wav_list)
        padded_wavs = []
        for w in final_wav_list:
            # w is [C, S]
            pad_amt = max_len - w.shape[-1]
            if pad_amt > 0:
                # Pad last dim
                padded = torch.nn.functional.pad(w, (0, pad_amt))
                padded_wavs.append(padded)
            else:
                padded_wavs.append(w)
        
        # Stack -> [Total_Batch, C, Max_S]
        batch_tensor = torch.stack(padded_wavs)
        
        # Combine Texts
        final_text_parts = []
        if 已有批量文本:
             # Strip prefixes from existing text to ensure consistency (convert to raw format)
             existing_lines = 已有批量文本.strip().split('\n')
             pattern = re.compile(r"^音频\d+[：:](.*)$")
             for line in existing_lines:
                 line = line.strip()
                 if not line: continue
                 m = pattern.match(line)
                 if m:
                     final_text_parts.append(m.group(1).strip())
                 else:
                     final_text_parts.append(line)
        
        final_text_parts.extend(text_list)
        final_text = "\n".join(final_text_parts)
        
        return ({"waveform": batch_tensor, "sample_rate": final_sr}, final_text)

class Qwen3TTSMaster:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "文本": ("STRING", {"multiline": True, "default": "你好，这是一段测试文本。"}),
                "工作模式": (["智能对话 (Auto Dialogue)", "声音设计 (Voice Design)", "声音克隆 (Voice Clone)", "自定义声音 (Custom Voice)"], {"default": "声音设计 (Voice Design)"}),
            },
            "optional": {
                # --- Model Loading ---
                "内置模型": (["自动 (Auto)"] + [
                    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                ],),
                "运行设备": (["auto", "cuda", "cpu"],),
                "精度": (["fp16", "fp32"],),

                # --- Voice Design ---
                "提示词": ("STRING", {"multiline": True, "default": "A gentle female voice."}),
                "角色预设模式": (["设计角色预设", "本地角色预设", "自动角色预设"], {"default": "设计角色预设"}),
                "本地预设文件": (["None"] + get_voice_prompt_files(),),
                
                # --- Voice Clone ---
                "参考音频": ("AUDIO",),
                "参考文本": ("STRING", {"multiline": True, "default": ""}),
                
                # --- Custom Voice ---
                "预设说话人": (list(Qwen3TTSCustomVoice.SPEAKER_PRESETS.keys()), {"default": "Vivian"}),
                
                # --- Dialogue ---
                "角色映射": ("STRING", {"multiline": True, "default": ""}),
                "角色预设": ("QWEN3_TTS_ROLE_PRESETS",),
                "启用停顿控制": ("BOOLEAN", {"default": True, "label_on": "开启", "label_off": "关闭"}),
                "批量保存角色预设": ("BOOLEAN", {"default": False, "label_on": "开启", "label_off": "关闭", "tooltip": "在对话模式下，自动保存所有参与角色的预设文件(.pt)到输出目录。"}),

                # --- Common ---
                "语言": (LANGUAGE_OPTIONS, {"default": "自动"}),
                "随机种子": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "自动卸载模型": ("BOOLEAN", {"default": False, "label_on": "是", "label_off": "否"}),
                "最大生成Token数": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 64, "display": "number", "tooltip": "限制生成的最大长度。默认2048，通常足够。设为0则根据文本自动调整（不限制）。"}),
                
                # --- Advanced ---
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05, "display": "number", "tooltip": "核采样 (Top-P)。\n常用范围：0.1 ~ 1.0\n解释：控制生成时选择候选词汇的概率累积阈值。数值越接近 1，候选词汇池越大，生成的内容越多样；数值越小，生成越聚焦。"}),
                "top_k": ("INT", {"default": 50, "min": 10, "max": 100, "step": 1, "display": "number", "tooltip": "Top-K 采样。\n常用范围：10 ~ 100\n解释：限制每次生成时仅从概率最高的前 K 个词汇中选择。K 值越大，生成的随机性越高；K 值越小，生成越稳定。"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.05, "display": "number", "tooltip": "温度系数 (Temperature)。\n常用范围：0.1 ~ 2.0\n解释：调整生成内容的随机性和多样性。数值越低（如 0.1），生成越保守、重复；数值越高（如 2.0），生成越有创意但可能出现逻辑混乱。"}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.05, "display": "number", "tooltip": "重复惩罚 (Repetition Penalty)。\n常用范围：1.0 ~ 2.0\n解释：抑制生成内容中的重复表述。数值越高（如 1.5），越不容易出现重复句子；数值为 1.0 时，无惩罚效果。"}),
                "启用高级采样配置": ("BOOLEAN", {"default": False, "label_on": "开启", "label_off": "关闭", "tooltip": "开启后，上方的高级采样参数（Top-P, Top-K 等）才会生效。关闭时使用模型默认配置。"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "QWEN3_TTS_VOICE_PROMPT")
    RETURN_NAMES = ("音频", "角色预设")
    FUNCTION = "generate"
    CATEGORY = "Qwen3TTS"
    DESCRIPTION = "全能节点：集成声音设计、克隆、自定义声音及智能对话功能。支持内置模型自动加载。自动模式下，根据输入（参考音频、角色映射）智能切换工作模式。"

    def generate(self, 文本, 工作模式, 内置模型="自动 (Auto)", 运行设备="auto", 精度="fp16", 提示词=None, 角色预设模式="设计角色预设", 本地预设文件="None", 参考音频=None, 参考文本=None, 预设说话人=None, 角色映射=None, 角色预设=None, 启用停顿控制=True, 批量保存角色预设=False, 语言="自动", 随机种子=0, 自动卸载模型=False, 最大生成Token数=2048, top_p=0.8, top_k=50, temperature=0.8, repetition_penalty=1.1, 启用高级采样配置=False):
        seed = 随机种子
        apply_seed(seed)

        # 1. Smart Mode Detection (Override 工作模式 based on inputs)
        active_mode = 工作模式
        if 角色映射 and 角色映射.strip():
            active_mode = "智能对话 (Auto Dialogue)"
        elif 参考音频 is not None:
            active_mode = "声音克隆 (Voice Clone)"
            
        # 2. Determine Model
        target_model_name = 内置模型
        if target_model_name == "自动 (Auto)":
            if "Voice Design" in active_mode:
                if 角色预设模式 == "本地角色预设":
                    target_model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
                    # If using local preset file, update instruct for model check if needed, 
                    # though VoiceDesign node handles it.
                    if 本地预设文件 and 本地预设文件 != "None":
                        提示词 = 本地预设文件
                elif 角色预设模式 == "自动角色预设":
                    try:
                        _resolve_prompt_path(提示词)
                        target_model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
                    except Exception:
                        target_model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
                else:
                    target_model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
            elif "Custom Voice" in active_mode:
                target_model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
            else: # Clone, Dialogue
                target_model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        
        qwen_model = load_qwen_model(target_model_name, 运行设备, 精度)[0]
        
        print(f"Qwen3TTS Master: Mode='{active_mode}', Model='{getattr(qwen_model, 'name_or_path', 'Unknown')}'")

        # --- 1.5 Auto-Design Missing Roles (for Dialogue Mode) ---
        generated_presets = {}
        if "Dialogue" in active_mode:
            # 1. Parse Dialogue to find roles
            dialogue_text = (文本 or "").strip()
            mapping_text = (角色映射 or "").strip()
            input_presets = 角色预设 or {}
            
            # Parse Mapping
            role_override_map = {}
            if mapping_text:
                for line in mapping_text.splitlines():
                    if "=" in line:
                        r, p = line.split("=", 1)
                        role_override_map[r.strip()] = p.strip()
            
            # Parse Dialogue Segments
            parsed_segments = []
            role_prompt_map = {} # Store extracted voice descriptions
            
            if dialogue_text:
                for line in dialogue_text.splitlines():
                    raw = line.strip()
                    if not raw: continue
                    
                    role_part = None
                    text_part = None
                    
                    if "：" in raw:
                        role_part, text_part = raw.split("：" , 1)
                    elif ":" in raw:
                        role_part, text_part = raw.split(":", 1)
                    
                    if role_part and text_part:
                        r_name = role_part.strip()
                        t_text = text_part.strip()
                        
                        # Extract voice description from brackets if present
                        # Check for （...） or (...)
                        voice_desc = None
                        
                        # Normalize brackets
                        r_name_clean = r_name.replace("（", "(").replace("）", ")")
                        
                        if "(" in r_name_clean and r_name_clean.endswith(")"):
                            start_idx = r_name_clean.find("(")
                            end_idx = r_name_clean.rfind(")")
                            if start_idx < end_idx:
                                voice_desc = r_name_clean[start_idx+1:end_idx].strip()
                                r_name = r_name_clean[:start_idx].strip()
                        
                        if voice_desc:
                            role_prompt_map[r_name] = voice_desc
                            
                        parsed_segments.append((r_name, t_text))

            unique_roles = set(r for r, t in parsed_segments)
            roles_to_design = []
            
            for role in unique_roles:
                if role in input_presets:
                    continue
                
                filename = role_override_map.get(role, f"{role}.pt")
                should_design = False
                
                if 角色预设模式 == "设计角色预设":
                    should_design = True
                elif 角色预设模式 == "自动角色预设":
                    try:
                        _resolve_prompt_path(filename)
                    except FileNotFoundError:
                        should_design = True
                
                if should_design:
                    roles_to_design.append(role)
            
            # 2. Design Voices for Missing/Requested Roles
            if roles_to_design:
                print(f"[Qwen3TTS Master] Roles to design: {roles_to_design}. Mode: {角色预设模式}")
                
                # Parse '提示词' for explicit assignments and anonymous pool
                explicit_prompt_map = {}
                anonymous_prompts = []
                
                raw_prompts = (提示词 or "").strip().splitlines()
                for line in raw_prompts:
                    line = line.strip()
                    if not line: continue
                    
                    if "：" in line:
                        r, p = line.split("：" , 1)
                        explicit_prompt_map[r.strip()] = p.strip()
                    elif ":" in line:
                        r, p = line.split(":", 1)
                        explicit_prompt_map[r.strip()] = p.strip()
                    else:
                        anonymous_prompts.append(line)
                
                used_anonymous_indices = set()
                
                def get_anonymous_prompt():
                    if not anonymous_prompts:
                        return "A clear, high-quality voice."
                    
                    # If we have unused prompts, pick one
                    available_indices = [i for i in range(len(anonymous_prompts)) if i not in used_anonymous_indices]
                    
                    if available_indices:
                        idx = random.choice(available_indices)
                        used_anonymous_indices.add(idx)
                        return anonymous_prompts[idx]
                    
                    # If all used, combine 2-3 random prompts
                    # Ensure we don't try to sample more than available
                    count = len(anonymous_prompts)
                    k = random.randint(2, min(3, count)) if count >= 2 else 1
                    selected = random.sample(anonymous_prompts, k)
                    return "，".join(selected)

                # A. Generate Audio Samples using Voice Design Model
                design_model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
                temp_audio_samples = {} # role -> (wav, sr, text)
                
                try:
                    design_model = load_qwen_model(design_model_name, 运行设备, 精度)[0]
                    
                    for role in roles_to_design:
                        # Determine prompt
                        design_prompt = None
                        
                        # Priority 1: Inline extraction (role_prompt_map)
                        if role in role_prompt_map:
                             design_prompt = role_prompt_map[role]
                             print(f"  - Using inline description for '{role}': {design_prompt}")
                        
                        # Priority 2: Explicit assignment in prompt box
                        elif role in explicit_prompt_map:
                             design_prompt = explicit_prompt_map[role]
                             print(f"  - Using explicit assignment for '{role}': {design_prompt}")
                             
                        # Priority 3: Anonymous pool / Combination
                        else:
                             design_prompt = get_anonymous_prompt()
                             print(f"  - Using auto-matched prompt for '{role}': {design_prompt}")
                        
                        # Find sample text (first utterance)
                        sample_text = f"Hello, I am {role}."
                        for r, t in parsed_segments:
                            if r == role and t:
                                sample_text = t[:100] # Use first 100 chars
                                break
                        
                        print(f"  - Designing Voice for '{role}': prompt='{design_prompt}', text='{sample_text[:30]}...'")
                        
                        # Generate Audio (Voice Design)
                        outs, sr = design_model.generate_voice_design(
                            text=sample_text,
                            instruct=design_prompt,
                            top_p=top_p, top_k=top_k, temperature=temperature
                        )
                        full_wav = np.concatenate(outs)
                        temp_audio_samples[role] = (full_wav, sr, sample_text)
                    
                    # Unload Design Model if needed
                    if 自动卸载模型:
                        if hasattr(design_model, "model"):
                            design_model.model.to("cpu")
                        torch.cuda.empty_cache()

                except Exception as e:
                    raise ValueError(f"【Qwen3TTS Master Error】自动设计角色音色失败 (Auto-Design Failed): {e}") from e

                # B. Extract Features using Base Model (create_voice_clone_prompt)
                if temp_audio_samples:
                    base_model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
                    try:
                        print(f"[Qwen3TTS Master] Switching to Base model to extract features for designed voices...")
                        base_model = load_qwen_model(base_model_name, 运行设备, 精度)[0]
                        
                        for role, (wav, sr, text) in temp_audio_samples.items():
                             # Create Voice Clone Prompt (Preset) using Base Model
                            prompt_items = base_model.create_voice_clone_prompt(
                                ref_audio=(wav, sr),
                                ref_text=text
                            )
                            # Convert to dict for compatibility
                            generated_presets[role] = _to_prompt_dict(prompt_items)
                            
                        print(f"[Qwen3TTS Master] Successfully designed and encoded {len(generated_presets)} voices.")
                        
                    except Exception as e:
                         raise ValueError(f"【Qwen3TTS Master Error】自动设计角色特征提取失败 (Feature Extraction Failed): {e}\n请确保能正常加载 Base 模型。") from e

            # 3. Reload/Ensure Base Model for Dialogue
            target_main_model = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
            # RH: Check if current model is Base. 
            # qwen_model.name_or_path might be full path or short name. 
            # Safest check is to see if it supports voice clone prompt creation (Base usually does, Design doesn't for inference)
            # OR just check the name.
            current_model_name = getattr(qwen_model, "name_or_path", "")
            if "Base" not in current_model_name and "base" not in current_model_name:
                 print(f"[Qwen3TTS Master] Switching back to Base model: {target_main_model}")
                 qwen_model = load_qwen_model(target_main_model, 运行设备, 精度)[0]

        generated_audio = None
        role_preset_out = None

        # 2. Execution
        if "Dialogue" in active_mode:
            # Merge generated presets
            final_presets = (角色预设 or {}).copy()
            final_presets.update(generated_presets)
            
            # Debug print to verify presets are passed
            # print(f"[Qwen3TTS Master] Final Presets Keys: {list(final_presets.keys())}")
            
            # Ensure all presets are valid dictionaries
            valid_presets = {}
            for role, p in final_presets.items():
                if p is None:
                    print(f"Warning: Role '{role}' has None preset. Skipping.")
                    continue
                valid_presets[role] = p

            # --- Batch Save Presets if Enabled ---
            if 批量保存角色预设:
                print(f"[Qwen3TTS Master] Batch saving role presets...")
                # Use default output directory
                if hasattr(folder_paths, "get_output_directory"):
                    base_out = folder_paths.get_output_directory()
                else:
                    base_out = getattr(folder_paths, "output_directory", None)
                
                if not base_out:
                    base_out = os.path.join(current_dir, "output")
                
                presets_save_dir = os.path.join(base_out, "qwen_tts_presets")
                os.makedirs(presets_save_dir, exist_ok=True)
                
                count_saved = 0
                for role, p in valid_presets.items():
                    # Only save roles that are actually used in the dialogue
                    if 'unique_roles' in locals() and role not in unique_roles:
                         continue
                         
                    try:
                        filename = f"{role}.pt"
                        out_path = os.path.join(presets_save_dir, filename)
                        
                        torch.save({"voice_clone_prompt": p, "role_name": role}, out_path)
                        print(f"  - Saved preset: {out_path}")
                        count_saved += 1
                    except Exception as e:
                        print(f"  - Failed to save preset for '{role}': {e}")
                
                print(f"[Qwen3TTS Master] Saved {count_saved} role presets to {presets_save_dir}")

            dialogue_node = Qwen3TTSDialogueSynthesis()
            result = dialogue_node.generate(
                模型=qwen_model,
                对白文本=文本,
                语言=语言,
                角色映射=角色映射,
                角色预设=valid_presets,
                启用停顿控制=启用停顿控制,
                seed=随机种子,
                自动卸载模型=自动卸载模型,
                最大生成Token数=最大生成Token数,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                启用高级采样配置=启用高级采样配置
            )
            generated_audio = result[0]
            # Return the dict of presets generated for the dialogue
            role_preset_out = valid_presets
            
        elif "Voice Design" in active_mode:
            design_node = Qwen3TTSVoiceDesign()
            result = design_node.generate(
                模型=qwen_model,
                文本=文本,
                提示词=提示词,
                角色预设模式=角色预设模式,
                语言=语言,
                本地预设文件=本地预设文件,
                seed=随机种子,
                自动卸载模型=自动卸载模型,
                最大生成Token数=最大生成Token数,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                启用高级采样配置=启用高级采样配置
            )
            generated_audio = result[0]
            
            # 3. Post-Process: Generate Preset from Designed Voice
            if generated_audio is not None:
                try:
                    wav_tensor = generated_audio["waveform"]
                    sample_rate = generated_audio["sample_rate"]
                    # [batch, channels, samples] -> [samples] (take first)
                    if wav_tensor.dim() == 3:
                         wav_np = wav_tensor[0].mean(dim=0).cpu().numpy()
                    else:
                         wav_np = wav_tensor.squeeze().cpu().numpy()
                         
                    ref_audio_tuple = (wav_np, sample_rate)
                    
                    # Create prompt using the model
                    prompt_items = qwen_model.create_voice_clone_prompt(
                        ref_audio=ref_audio_tuple,
                        ref_text=文本, 
                    )
                    role_preset_out = prompt_items
                except Exception as e:
                    print(f"Warning: Failed to generate Role Preset in Voice Design mode: {e}")
                    role_preset_out = None
            
        elif "Voice Clone" in active_mode:
            clone_node = Qwen3TTSVoiceClone()
            result = clone_node.generate(
                模型=qwen_model,
                文本=文本,
                参考音频=参考音频,
                参考文本=参考文本,
                语言=语言,
                seed=随机种子,
                自动卸载模型=自动卸载模型,
                最大生成Token数=最大生成Token数,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                启用高级采样配置=启用高级采样配置
            )
            generated_audio = result[0]
            role_preset_out = result[1]
            
        elif "Custom Voice" in active_mode:
            custom_node = Qwen3TTSCustomVoice()
            result = custom_node.generate(
                模型=qwen_model,
                文本=文本,
                预设说话人=预设说话人,
                语言=语言,
                提示词=提示词,
                seed=随机种子,
                自动卸载模型=自动卸载模型,
                最大生成Token数=最大生成Token数,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                启用高级采样配置=启用高级采样配置
            )
            generated_audio = result[0]
            role_preset_out = None
            
        else:
            raise ValueError(f"未知的模式: {active_mode}")
            
        return (generated_audio, role_preset_out)

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
    "Qwen3TTSVoiceDescription": Qwen3TTSVoiceDescription,
    "Qwen3TTSBatchVoiceCloneInput": Qwen3TTSBatchVoiceCloneInput,
    "Qwen3TTSMaster": Qwen3TTSMaster,
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
    "Qwen3TTSVoiceDescription": "Qwen3 TTS 音色描述",
    "Qwen3TTSBatchVoiceCloneInput": "Qwen3 TTS 批量语音克隆输入",
    "Qwen3TTSMaster": "Qwen3 TTS 全能节点(Master)",
}
