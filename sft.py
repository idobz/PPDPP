import argparse
import json
import logging
import os

import torch
from pytorch_transformers import WarmupLinearSchedule
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    BertConfig,
    BertTokenizer,
    RobertaConfig,
    RobertaTokenizer,
)

import data_reader
import utils
import wandb
from agent import PPDPP

# python sft.py --gpu="0 1" --do_train --overwrite_output_dir --per_gpu_train_batch_size=8 --per_gpu_eval_batch_size=8

tok = {"bert": BertTokenizer, "roberta": RobertaTokenizer}
cfg = {"bert": BertConfig, "roberta": RobertaConfig}


class DataFrame(Dataset):
    def __init__(self, data, args):
        self.source_ids = data["source_ids"]
        self.target_ids = data["target_ids"]
        self.max_len = args.max_seq_length

    def __getitem__(self, index):
        return self.source_ids[index][: self.max_len], self.target_ids[index]

    def __len__(self):
        return len(self.source_ids)


def collate_fn(data):
    source_ids, target_ids = zip(*data)

    input_ids = [torch.tensor(source_id).long() for source_id in source_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

    attention_mask = input_ids.ne(0)
    labels = torch.tensor(target_ids).long()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def train(args, train_dataset, model, tokenizer):
    run = wandb.init(project="PPDPP", config=args)
    wandb.watch(model, log_freq=100)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, len(args.device_id))
    train_dataloader = DataLoader(
        DataFrame(train_dataset, args),
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    t_total = (
        len(train_dataloader)
        // args.gradient_accumulation_steps
        * args.num_train_epochs
    )

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=args.warmup_steps, t_total=t_total
    )

    # multi-gpu training (should be after apex fp16 initialization)
    if len(args.device_id) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.device_id)

    # Train!
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logging.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps,
    )
    logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", t_total)

    tr_loss = 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    utils.set_random_seed(
        args.seed
    )  # Added here for reproductibility (even between python 2 and 3)
    global_step = 0

    # total_rouge = evaluate(args, model, tokenizer, save_output=True)
    best_f1 = 0  # total_rouge[2]

    for e in train_iterator:
        logging.info("training for epoch {} ...".format(e))
        print("training for epoch {} ...".format(e))
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            # batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch["input_ids"].to(args.device),
                "attention_mask": batch["attention_mask"].to(args.device),
                "labels": batch["labels"].to(args.device),
            }
            outputs = model(**inputs)
            loss = outputs  # [0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if len(args.device_id) > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
            global_step += 1

            wandb.log({"loss": loss.item()}, step=global_step)

        # Log metrics
        results = evaluate(args, model, tokenizer, save_output=True)

        if results[0] > best_f1:
            # Save model checkpoint
            best_f1 = results[0]
            output_dir = os.path.join(args.output_dir, "best_checkpoint")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            torch.save(
                model_to_save.state_dict(),
                os.path.join(output_dir, "pytorch_model.bin"),
            )
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logging.info("Saving model checkpoint to %s", output_dir)

    run.finish()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, save_output=False):
    # Evaluate the model on the validation or test set
    eval_dataset = data_reader.load_and_cache_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, len(args.device_id))

    eval_dataloader = DataLoader(
        DataFrame(eval_dataset, args),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    # Eval!
    logging.info("***** Running evaluation *****")
    logging.info("  Num examples = %d", len(eval_dataset))
    logging.info("  Batch size = %d", args.eval_batch_size)
    preds = []
    targets = []
    scores = []
    sources = []

    # Evaluate the model
    model_to_eval = model.module if hasattr(model, "module") else model
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        with torch.no_grad():
            pred = model_to_eval(
                input_ids=batch["input_ids"].to(args.device),
                attention_mask=batch["attention_mask"].to(args.device),
            )

            scores.extend([p[0] for p in pred.cpu().tolist()])
            preds.extend(pred.argmax(dim=-1).cpu().tolist())
            targets.extend(batch["labels"].tolist())
            sources.extend(
                [
                    tokenizer.decode(
                        g, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    for g in batch["input_ids"]
                ]
            )

    # Save evaluation results if required
    if save_output:
        with open(
            os.path.join(
                args.output_dir,
                "{}_{}_{}.score".format(
                    args.data_name,
                    args.set_name,
                    list(filter(None, args.model_name_or_path.split("/"))).pop(),
                ),
            ),
            "w",
        ) as outfile:
            for target, pred, score, source in zip(targets, preds, scores, sources):
                outfile.write("{}\t{}\t{}\t{}\n".format(target, pred, score, source))

    # Calculate and return evaluation metrics
    precision = precision_score(targets, preds, average="macro", zero_division=0)
    recall = recall_score(targets, preds, average="macro", zero_division=0)
    f1 = f1_score(targets, preds, average="macro", zero_division=0)
    auto_scores = [precision, recall, f1]
    logging.info(auto_scores)
    print(auto_scores)
    return auto_scores


def main():
    # Load configuration from file or use default
    config_file = "data/config/sft_config.json"
    if os.path.exists(config_file):
        with open(config_file) as f:
            config = json.load(f)
    else:
        # Use default configuration if file doesn't exist
        config = {
            "data_name": "cb",
            "set_name": "valid",
            "model_name": "roberta",
            "model_name_or_path": "roberta-large",
            "output_dir": "models",
            "data_dir": "./data",
            "cache_dir": "./cache",
            "do_train": True,
            "do_eval": True,
            "overwrite_output_dir": True,
            "overwrite_cache": True,
            "do_lower_case": True,
            "max_seq_length": 512,
            "seed": 42,
            "gpu": "",
            "per_gpu_train_batch_size": 4,
            "per_gpu_eval_batch_size": 1,
            "num_train_epochs": 10,
            "gradient_accumulation_steps": 1,
            "warmup_steps": 400,
            "learning_rate": 6e-6,
            "weight_decay": 0.01,
            "adam_epsilon": 1e-8,
            "max_grad_norm": 1.0,
            "local_rank": -1,
        }
        # Save default config for future use
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)

    # Convert config dict to Namespace for compatibility
    args = argparse.Namespace(**config)

    # Set up output directory
    args.output_dir = os.path.join(args.output_dir, args.data_name, args.model_name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG, filename=args.output_dir + "/log.txt", filemode="a"
    )

    # Check if output directory already exists and is not empty
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    device, device_id = utils.set_cuda(args)
    args.device = device
    args.device_id = device_id

    # Set random seed for reproducibility
    utils.set_random_seed(args.seed)

    # Load pre-trained model and tokenizer
    config = cfg[args.model_name].from_pretrained(
        args.model_name_or_path, cache_dir=args.cache_dir
    )
    tokenizer = tok[args.model_name].from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir,
    )

    # Initialize the PPDPP model
    model = PPDPP(args, config, tokenizer)

    # Load and prepare the training dataset
    # This is one place to change: read just DAs (and maybe the last textual message)
    train_dataset = data_reader.load_and_cache_examples(args, tokenizer, evaluate=False)

    # Move the model to the specified device (CPU/GPU)
    model.to(args.device)

    logging.info("Training/evaluation parameters %s", args)
    output_dir = os.path.join(args.output_dir, "best_checkpoint")

    # Training
    # switch to taining HF API
    if args.do_train:
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logging.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        tokenizer.save_pretrained(output_dir)

    # Evaluation
    if args.do_eval:
        # Load a trained model and vocabulary that you have fine-tuned
        if hasattr(model, "module"):
            model.module.load_state_dict(
                torch.load(os.path.join(output_dir, "pytorch_model.bin"))
            )
        else:
            model.load_state_dict(
                torch.load(os.path.join(output_dir, "pytorch_model.bin"))
            )
        tokenizer = tok[args.model_name].from_pretrained(
            output_dir, do_lower_case=args.do_lower_case
        )
        model.to(args.device)
        args.set_name = "test"
        evaluate(args, model, tokenizer, save_output=True)


if __name__ == "__main__":
    main()
