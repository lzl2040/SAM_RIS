import torch
import open_clip

# 模型选择与加载
model_name = 'ViT-L-14-336'
pretrained = 'openai'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载模型和 tokenizer
text_model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
tokenizer = open_clip.get_tokenizer(model_name)
text_model = text_model.to(device).eval()

# 输入文本
texts = ["a cat on the sofa", "a mouse"]
input_ids = tokenizer(texts).to(device)  # shape: [2, 77]
print("Input IDs shape:", input_ids.shape)

# 手动前向过程（仿照 encode_text 的实现）
with torch.no_grad():
    # Token embedding + positional embedding
    x = text_model.token_embedding(input_ids)  # shape: [2, 77, d_model]
    print(x.shape)
    x = x + text_model.positional_embedding  # broadcast to [2, 77, d_model]
    
    # Transpose for transformer input: [sequence_len, batch_size, d_model]
    # x = x.permute(1, 0, 2)                       # [77, 2, 768] ✅
    attn_mask = text_model.attn_mask.to(device)  # [77, 77]

    # Transformer encoder
    x = text_model.transformer(x, attn_mask=attn_mask)

    # Transpose back: [batch_size, sequence_len, d_model]
    # x = x.permute(1, 0, 2)

    # LayerNorm
    x = text_model.ln_final(x)
    print(x.shape)

    # 提取 <EOT> token 所在位置的 hidden state
    eot_indices = input_ids.argmax(dim=-1)
    print(eot_indices) # [6, 3]
    text_features = x[torch.arange(x.shape[0]), eot_indices]

    # 映射到最终文本向量空间
    text_features = text_features @ text_model.text_projection

    # 归一化（可选）
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

print("Text features shape:", text_features.shape)  # [2, 768]
