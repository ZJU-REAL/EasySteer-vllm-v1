# SPDX-License-Identifier: Apache-2.0
"""Per-family class-name lists for steering-hook discovery.

Discovery is structural first (see steer_vectors.models); these lists
are the fallback for module layouts structural discovery cannot
identify, organized alphabetically by model family.
"""

SUPPORTED_DECODER_LAYERS: list[str] = [
    # A
    "ApertusDecoderLayer",
    "ArceeDecoderLayer",
    "ArcticDecoderLayer",
    "AriaTextDecoderLayer",
    # B
    "BaiChuanDecoderLayer",
    "BailingMoeBlock",
    "BambaAttentionDecoderLayer",
    "BambaMixerDecoderLayer",
    "BertLayer",
    "BloomBlock",
    # C
    "ChameleonDecoderLayer",
    "ChameleonSwinDecoderLayer",
    "CohereDecoderLayer",
    # D
    "DbrxBlock",
    "DeciLMDecoderLayer",
    "DecoderLayer",
    "DeepseekDecoderLayer",
    "DeepseekV2DecoderLayer",
    "Dots1DecoderLayer",
    # E
    "Ernie4_5_MoeDecoderLayer",
    "Ernie4_5_VLMoeDecoderLayer",
    "Exaone4DecoderLayer",
    "ExaoneDecoderLayer",
    # F
    "FalconDecoderLayer",
    "FalconH1AttentionDecoderLayer",
    "FalconH1SSMDecoderLayer",
    "FlashDecoderLayer",
    "FlexOlmoDecoderLayer",
    # G
    "Gemma2DecoderLayer",
    "Gemma3DecoderLayer",
    "Gemma3nDecoderLayer",
    "GemmaDecoderLayer",
    "Glm4DecoderLayer",
    "Glm4MoeDecoderLayer",
    "GLMBlock",
    "GPT2Block",
    "GPTBigCodeBlock",
    "GPTJBlock",
    "GPTNeoXLayer",
    "GraniteDecoderLayer",
    "GraniteMoeDecoderLayer",
    "GraniteMoeHybridAttentionDecoderLayer",
    "GraniteMoeHybridMambaDecoderLayer",
    "GraniteMoeSharedDecoderLayer",
    "Grok1DecoderLayer",
    # H
    "HunYuanDecoderLayer",
    # I
    "InternLM2VEDecoderLayer",
    "InternLMDecoderLayer",
    # J
    "JAISBlock",
    "JambaAttentionDecoderLayer",
    "JambaMambaDecoderLayer",
    # L
    "Lfm2AttentionDecoderLayer",
    "Lfm2MoeAttentionDecoderLayer",
    "Lfm2MoeShortConvDecoderLayer",
    "Lfm2ShortConvDecoderLayer",
    "Llama4DecoderLayer",
    "LlamaDecoderLayer",
    # M
    "Mamba2DecoderLayer",
    "MambaDecoderLayer",
    "MiniCPM3DecoderLayer",
    "MiniCPMDecoderLayer",
    "MiniMaxText01DecoderLayer",
    "MixtralDecoderLayer",
    "MolmoDecoderLayer",
    "MolmoDecoderNormAfterLayer",
    "MPTBlock",
    # N
    "NemotronDecoderLayer",
    "NemotronHAttentionDecoderLayer",
    "NemotronHMambaDecoderLayer",
    "NemotronHMLPDecoderLayer",
    "NemotronHMoEDecoderLayer",
    # O
    "Olmo2DecoderLayer",
    "OlmoDecoderLayer",
    "OlmoeDecoderLayer",
    "OPTDecoderLayer",
    "OrionDecoderLayer",
    # P
    "PersimmonDecoderLayer",
    "PhiLayer",
    "PhiMoEDecoderLayer",
    "Plamo2DecoderLayer",
    # Q
    "Qwen2DecoderLayer",
    "Qwen2MoeDecoderLayer",
    "Qwen3DecoderLayer",
    "Qwen3MoeDecoderLayer",
    "Qwen3NextDecoderLayer",
    "QWenBlock",
    # S
    "SeedOssDecoderLayer",
    "SolarDecoderLayer",
    "StablelmDecoderLayer",
    "Starcoder2DecoderLayer",
    "Step3TextDecoderLayer",
    # T
    "TransformerBlock",
    # W
    "WhisperDecoderLayer",
    # Z
    "Zamba2AttentionDecoderLayer",
    "Zamba2HybridLayer",
    "Zamba2MambaDecoderLayer",
]


SUPPORTED_MOE_LAYERS: list[str] = [
    # Qwen family
    "Qwen2MoeSparseMoeBlock",
    "Qwen3MoeSparseMoeBlock",
    "Qwen3NextSparseMoeBlock",
    "QwenMoE",
    "Qwen2MoE",
    # Mixtral / Llama family
    "MixtralMoE",
    "Llama4MoE",
    "PhiMoE",
    # DeepSeek family
    "DeepseekMoE",
    "DeepseekV2MoE",
    # Kimi
    "KimiMoE",
    # GLM
    "Glm4MoE",
    "GLMMoE",
    # Ernie
    "Ernie4MoE",
    "Ernie4_5_MoeMoE",
    "Ernie4_5_VLMoeMoE",
    # Others
    "DbrxExperts",
    "DbrxMoE",
    "ArcticMoE",
    "JambaMoE",
    "Grok1MoE",
    "GraniteMoeMoE",
    "MiniMaxText01MoE",
    "MiniMaxM2MoE",
    "MiniCPMMoE",
    "OlmoeMoE",
    "FlexOlmoMoE",
    "NemotronHMoE",
    "BailingMoE",
    "Dots1MoE",
    "NomicMoE",
]
