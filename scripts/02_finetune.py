"""Step 2: LoRA fine-tune on CodeAlpaca, merge adapters, save FP16 checkpoint, evaluate.
Writes checkpoints/lora, checkpoints/merged_fp16, and results/quality.csv row variant=finetuned_fp16."""
import torch
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

from _common import setup
from src.data import load_codealpaca, load_mbpp
from src.evaluate import heldout_loss, mbpp_pass_at_1
from src.model import load_finetuned, load_model, load_tokenizer
from src.utils import append_csv

cfg, log = setup("02_finetune")
tok = load_tokenizer(cfg)
train, heldout = load_codealpaca(cfg)
if cfg.get("_smoke"):
    train = train[:200]
t = cfg["train"]
max_len = cfg["data"]["max_seq_len"]


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ex = self.rows[i]
        p = tok(ex["prompt"], add_special_tokens=False).input_ids
        r = tok(ex["response"] + tok.eos_token, add_special_tokens=False).input_ids
        ids = (p + r)[:max_len]
        labels = ([-100] * len(p) + r)[:max_len]  # loss on response tokens only
        return {"input_ids": ids, "labels": labels}


def collate(batch):
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), tok.pad_token_id)
    lab = torch.full((len(batch), L), -100)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for j, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[j, :n] = torch.tensor(b["input_ids"])
        lab[j, :n] = torch.tensor(b["labels"])
        att[j, :n] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


# fp32 master weights + fp16 autocast: the stable recipe on T4 (no bf16 there)
model = load_model(cfg, dtype="float32")
model.train()
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(
    r=t["lora_r"], lora_alpha=t["lora_alpha"], lora_dropout=t["lora_dropout"],
    target_modules=t["target_modules"], task_type="CAUSAL_LM"))
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=cfg["paths"]["adapter_dir"] + "_runs",
    num_train_epochs=t["epochs"],
    per_device_train_batch_size=t["batch_size"],
    gradient_accumulation_steps=t["grad_accum"],
    learning_rate=t["lr"],
    lr_scheduler_type="cosine",
    warmup_ratio=t["warmup_ratio"],
    logging_steps=t["logging_steps"],
    save_strategy="no",
    report_to="none",
    fp16=torch.cuda.is_available(),
    seed=cfg["seed"],
    remove_unused_columns=False,
)
trainer = Trainer(model=model, args=args, train_dataset=SFTDataset(train), data_collator=collate)
trainer.train()

model.save_pretrained(cfg["paths"]["adapter_dir"])
merged = model.merge_and_unload()
merged.to(torch.float16).save_pretrained(cfg["paths"]["merged_dir"])
tok.save_pretrained(cfg["paths"]["merged_dir"])
log.info("saved merged FP16 model to %s", cfg["paths"]["merged_dir"])

del model, merged, trainer
torch.cuda.empty_cache()

# evaluate the merged model exactly the way every later step will
model = load_finetuned(cfg)
problems = load_mbpp(cfg)
loss = heldout_loss(model, tok, heldout, cfg)
p1 = mbpp_pass_at_1(model, tok, problems, cfg,
                    save_to=f"{cfg['paths']['results_dir']}/generations/finetuned_fp16.jsonl")
log.info("finetuned heldout loss %.4f  MBPP pass@1 %.4f", loss, p1)
append_csv(f"{cfg['paths']['results_dir']}/quality.csv", {
    "variant": "finetuned_fp16", "stage": "finetuned", "precision": "fp16",
    "heldout_loss": round(loss, 5), "mbpp_pass1": round(p1, 4),
    "n_mbpp": len(problems), "avg_weight_bits": 16.0,
})
