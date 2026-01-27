from .constants import *

class Qwen3TTSVoicePresets:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "女童音色": (PRESET_CHILD_FEMALE, {"default": "不选择"}),
                "男童音色": (PRESET_CHILD_MALE, {"default": "不选择"}),
                "少女音色": (PRESET_YOUNG_FEMALE, {"default": "不选择"}),
                "少年音色": (PRESET_YOUNG_MALE, {"default": "不选择"}),
                "青年女音色": (PRESET_YOUNG_ADULT_FEMALE, {"default": "不选择"}),
                "青年男音色": (PRESET_YOUNG_ADULT_MALE, {"default": "不选择"}),
                "中年女音色": (PRESET_MIDDLE_AGED_FEMALE, {"default": "不选择"}),
                "中年男音色": (PRESET_MIDDLE_AGED_MALE, {"default": "不选择"}),
                "老年女音色": (PRESET_OLD_FEMALE, {"default": "不选择"}),
                "老年男音色": (PRESET_OLD_MALE, {"default": "不选择"}),
            },
            "optional": {
                "上一个提示词": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("提示词",)
    FUNCTION = "process"
    CATEGORY = "Qwen3TTS"
    DESCRIPTION = "提供预设的年龄和性别音色组合，支持串联。"

    def process(self, 女童音色, 男童音色, 少女音色, 少年音色, 青年女音色, 青年男音色, 中年女音色, 中年男音色, 老年女音色, 老年男音色, 上一个提示词=None):
        parts = []
        if 上一个提示词 and 上一个提示词.strip():
            parts.append(上一个提示词.strip())
        
        current_parts = []
        
        def _get_en(text):
            return VOICE_DESC_MAP.get(text, text)
        
        for tex in [女童音色, 男童音色, 少女音色, 少年音色, 青年女音色, 青年男音色, 中年女音色, 中年男音色, 老年女音色, 老年男音色]:
            if tex and tex != "不选择":
                current_parts.append(_get_en(tex))
                
        if current_parts:
            combined_current = ", ".join(current_parts)
            if not combined_current.endswith(".") and not combined_current.endswith("。"):
                 combined_current += "."
            parts.append(combined_current)
            
        final_prompt = " ".join(parts)
        return (final_prompt,)

NODE_CLASS_MAPPINGS = {
    "Qwen3TTSVoicePresets": Qwen3TTSVoicePresets
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3TTSVoicePresets": "Qwen3 TTS 预设音色"
}
