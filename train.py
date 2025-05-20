import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from preproc import CMNTranslationDataset
from transformers import AutoTokenizer
from torch.utils.data import random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from loss import LabelSmoothingLoss
from transformers import get_cosine_schedule_with_warmup
from torch.nn.functional import softmax
from beam_search import beam_search
from model import Transformer  
import gc
import matplotlib.pyplot as plt


MAX_LEN = 64
BATCH_SIZE = 16
D_MODEL = 128
FF_DIM = 512
HEADS = 4
EPOCHS = 100
NUM_LAYERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "Transformer_v2.pt"


def Trainer(data):
    dataset = CMNTranslationDataset(data)
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_training_steps = len(train_loader) * EPOCHS
    num_warmup_steps = num_training_steps // 10
    pad_token_id = dataset.tokenizer.pad_token_id

    model = Transformer(
        vocab_size=dataset.tokenizer.vocab_size,
        max_len=dataset.max_len,
        heads=HEADS,
        d_model=D_MODEL,
        ffn=FF_DIM,
        num_layers=NUM_LAYERS,
        padding_idx=pad_token_id,
        dropout=0.3
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    criterion = LabelSmoothingLoss(0.2, vocab_size=dataset.tokenizer.vocab_size, ignore_index=pad_token_id)

    best_accuracy = 0.0
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    patience = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    for epoch in range(EPOCHS):     
        model.train()
        train_loss = 0
        vaildation_loss = 0
        train_correct = 0
        train_total = 0

        for src, tgt in tqdm(train_loader):
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            decoder_input = tgt[:, :-1] # [B, T-1]
            labels = tgt[:, 1:] # [B, T-1]
      

            out = model(src, decoder_input) # [B, T-1, vocab_size]
            out = out.reshape(-1, dataset.tokenizer.vocab_size) # [B*(T-1), vocab size]
            labels = labels.reshape(-1) # [B*(T-1)]
            
            # Loss
            loss = criterion(out, labels)

            pred = out.argmax(-1)
            mask = (labels != dataset.tokenizer.pad_token_id)

            non_pad_tokens = mask.sum().item()
            train_loss += loss.item() * non_pad_tokens
            train_correct += torch.eq(pred, labels).masked_select(mask).cpu().sum().item()
            train_total += non_pad_tokens
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  
        

        avg_train_loss = train_loss / train_total
        train_accuracy  = train_correct/train_total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        print(f"[{epoch+1}/{EPOCHS}]  Train Acc: {train_accuracy:.4f}, Train Loss: {avg_train_loss:.4f}")


        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for src, tgt in tqdm(val_loader):
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                decoder_input = tgt[:, :-1]
                labels = tgt[:, 1:]
      
                output = model(src, decoder_input)
                output = output.reshape(-1, dataset.tokenizer.vocab_size) # [B*T, vocab size]
                labels = labels.reshape(-1)

                
                loss = criterion(output, labels)
                mask = (labels != dataset.tokenizer.pad_token_id)
                non_pad_tokens = mask.sum().item()
                vaildation_loss += loss.item() * non_pad_tokens
                total += non_pad_tokens

                pred = output.argmax(dim = -1)
             
                correct += torch.eq(pred, labels).masked_select(mask).sum().cpu().item()


            avg_val_loss = vaildation_loss / total
            val_accuracy = correct / total
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            print(f'[{epoch+1}/{EPOCHS}] Val Accuracy: {val_accuracy:.4f}, Val Loss: { avg_val_loss:.4f}')

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_PATH)
            print(f'Model saved in {MODEL_PATH}(ACC:{val_accuracy:.2f})')
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                print("Early stopping")
                break


        
        demo_inputs = [ "How are you?", 
                       "It is very hot here in the summer.", 
                       "My father is retiring next spring."] #"How are you?", "Excuse me, do you speak English?",
        for sentence in demo_inputs:
            output_text, token_ids = beam_search(
                model, dataset.tokenizer, sentence,
                beam_size=3,
                max_len=MAX_LEN,
                device=DEVICE
            )
            print(f"[EN] {sentence}")
            print(f"[ZH] {output_text}")
            print(f"Token IDs: {token_ids}")
        torch.cuda.empty_cache()
        gc.collect()   
    
    # Loss curve
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.savefig('loss_curve.png')

    # Accuracy curve
    plt.figure()
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.savefig('acc_curve.png')


if __name__ == '__main__':

    Trainer('./data/cmn.txt')
