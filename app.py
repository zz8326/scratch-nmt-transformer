import tkinter as tk
import torch
from transformers import AutoTokenizer
from model import Transformer
from beam_search import beam_search

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型與 tokenizer
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

def run_translate(model, tokenizer, sentence):
    zh, ids = beam_search(
        model, tokenizer, sentence,
        beam_size=3,
        max_len=64,
        device=DEVICE
    )
    return zh

def on_entry_click(event):
    if entry.get() == placeholder:
        entry.delete(0, "end")  # 清空 entry
        entry.config(fg='black')

def on_focusout(event):
    if entry.get() == "":
        entry.insert(0, placeholder)
        entry.config(fg='grey')

def on_translate():
    en = entry.get()
    if en == placeholder or not en.strip():
        result_var.set("請先輸入英文句子！")
        return
    zh = run_translate(model, tokenizer, en)
    result_var.set(zh)

root = tk.Tk()
root.title("Transformer 翻譯 Demo")

placeholder = "請輸入英文句子..."
entry = tk.Entry(root, width=40, fg='grey')
entry.insert(0, placeholder)
entry.bind('<FocusIn>', on_entry_click)
entry.bind('<FocusOut>', on_focusout)
entry.pack(pady=8)

btn = tk.Button(root, text="翻譯", command=on_translate)
btn.pack(pady=4)

result_var = tk.StringVar()
result = tk.Label(root, textvariable=result_var, width=60, height=5, wraplength=400, bg='#f0f0f0')
result.pack(pady=8)

root.mainloop()
