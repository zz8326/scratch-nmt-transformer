import torch
from model import Transformer
from preproc import CMNTranslationDataset
from transformers import AutoTokenizer
from beam_search import beam_search

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化 tokenizer & model
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = Transformer(
    vocab_size=tokenizer.vocab_size,
    max_len=64,
    heads=4,
    d_model=128,
    ffn=512,
    num_layers=4
).to(DEVICE)
model.load_state_dict(torch.load("Transformer.pt", map_location=DEVICE))
model.eval()

def greedy_decode(model, tokenizer, sentence, max_len=64, device=DEVICE):
    model.eval()
    with torch.no_grad():
        tokens = tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
        src = tokens["input_ids"].to(device)
        bos_id = tokenizer.bos_token_id or tokenizer.cls_token_id or 0
        eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id or 2
        generated = torch.full((1, 1), bos_id, dtype=torch.long).to(device)

        for _ in range(max_len):
            output = model(src, generated)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == eos_id or generated.shape[1] >= max_len:
                break
        # Token IDs & decoding
        token_ids = generated[0].tolist()
        output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        return output_text, token_ids

def translate(model, tokenizer, sentence, method="beam", beam_size=3, max_len=64, device=DEVICE):
    """
    method: 'beam' or 'greedy'
    """
    if method == "beam":
        output_text, token_ids = beam_search(
            model, tokenizer, sentence,
            beam_size=beam_size,
            max_len=max_len,
            device=device
        )
    elif method == "greedy":
        output_text, token_ids = greedy_decode(
            model, tokenizer, sentence,
            max_len=max_len,
            device=device
        )
    else:
        raise ValueError("method must be 'beam' or 'greedy'")
    return output_text, token_ids

def demo():
    demo_inputs = [
        "How are you?",
        "Excuse me, do you speak English?",
        "Have you already fed the horses?",
        "I believe he is coming tomorrow."
    ]
    for en in demo_inputs:
        zh, ids = translate(model, tokenizer, en, method="beam", beam_size=3)
        print(f"[EN] {en}")
        print(f"[ZH] {zh}")
        print(f"Token IDs: {ids}")
        print("")

def cli():
    print("Transformer 翻譯小工具 (Ctrl+C 結束)")
    while True:
        en = input("輸入英文句子: ")
        zh, ids = translate(model, tokenizer, en, method="beam", beam_size=3)
        print(f"→ {zh}\n")

if __name__ == "__main__":
    # demo()  # 範例
    cli()   # 命令列互動
