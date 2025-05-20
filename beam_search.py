import torch
import torch.nn.functional as F

def beam_search(model, tokenizer, src_sentence, beam_size=3, max_len=64, device='cuda'):
    model.eval()
    with torch.no_grad():
        # Tokenize input sentence
        tokens = tokenizer(src_sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
        src = tokens["input_ids"].to(device)
        if src.ndim == 1:
            src = src.unsqueeze(0)

        bos_id = tokenizer.bos_token_id or tokenizer.cls_token_id or 0
        eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id or 2

        beam = [(torch.tensor([[bos_id]], device=device), 0.0)]  # (sequence, score)

        for _ in range(max_len):
            candidates = []
            for seq, score in beam:
                if seq[0, -1].item() == eos_id:
                    candidates.append((seq, score))
                    continue
                out = model(src, seq)
                logits = out[:, -1, :]  # [1, vocab]
                log_probs = F.log_softmax(logits, dim=-1)
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)

                for log_prob, idx in zip(topk_log_probs[0], topk_indices[0]):
                    new_seq = torch.cat([seq, idx.view(1, 1)], dim=1)
                    new_score = score + log_prob.item()
                    candidates.append((new_seq, new_score))

            # Select top beam_size sequences
            beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]

            # All beams ended?
            if all(seq[0, -1].item() == eos_id for seq, _ in beam):
                break

        best_seq = beam[0][0][0]
        output = tokenizer.decode(best_seq, skip_special_tokens=True)
        return output, best_seq.tolist() 