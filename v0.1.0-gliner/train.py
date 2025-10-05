import os
from pathlib import Path
from typing import Optional

import torch
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
import typer
from datasets import load_dataset, load_from_disk
from gliner import GLiNER
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from gliner.data_processing.collator import DataCollator
from transformers import TrainerCallback, get_scheduler
from gliner.training import Trainer, TrainingArguments
from wasabi import msg
import srsly

# additional libraries
import math
from transformers import set_seed
import numpy as np
import random

# optimizer stuff
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor
import torch.nn as nn

def main(
    # fmt: off
    base_model: str = typer.Argument(..., help="Base model used for training."),
    output_dir: Path = typer.Argument(..., help="Path to store the output model."),
    checkpoint_dir: Path = typer.Option(Path("checkpoints"), help="Path for storing checkpoints."),
    push_to_hub: Optional[str] = typer.Option(None, help="If set, will upload the trained model to the provided Huggingface model namespace."),
    dataset: str = typer.Option("etdvprg/gold-ml-batch1", help="Path to the PHMartialLawNER dataset."),
    local_dataset: str = typer.Option(None, help="If set, this is where the dataset would be retrieved. Leave blank if the dataset would be retrieved remotely."),
    random_seed: int = typer.Option(42, help="Random seed for reproducibility."),
    # training parameters
    num_steps: int = typer.Option(500, help="Number of steps to run training."),
    batch_size: int = typer.Option(16, help="Batch size used for training."),
    gradient_accumulation_steps: int = typer.Option(2, help="Number of steps to accumulate gradients before performing a backward/update pass."),
    # fmt: on
):  
    # Preliminary Setup
    # ------------------------------
    # set up model storage in the proper directory
    os.makedirs(output_dir, exist_ok=True); 
    os.makedirs(checkpoint_dir, exist_ok=True);

    # setting to decide if dataset should be pushed
    if push_to_hub:
        api_token = os.getenv("HF_TOKEN")
        if not api_token:
            msg.fail("HF_TOKEN is missing! Won't be able to --push-to-hub", exits=1)

    # for reproducibility
    fix_seed(random_seed)

    # Load and Format the dataset
    # ------------------------------
    msg.info(f"Formatting the {dataset} dataset")
    
    if local_dataset:
        corpus_path = Path(f"assets/corpus/{str(local_dataset)}/dataset_full")
        ds = load_from_disk(corpus_path)

        if ds is None:
            msg.fail(f"Local Dataset :: {local_dataset} not found! Exiting", exits=1)
    else:
        ds = load_dataset(dataset, trust_remote_code=True)
        if ds is None:
            msg.fail(f"Remote Dataset :: {dataset} not found! Exiting", exits=1)

    # Retrieve the NER labels
    # ------------------------------
    label_map = srsly.read_json("assets/mapped_labels.json")

    if label_map is None:
        msg.fail("mapped_labels.json not found! Cannot convert entity type", exits=1)
        return # to ease the compiler
    
    # label wizardy to be used in format_to_gliner
    labels = label_map["labels"]
    iob_mapping = label_map["iob_mapping"]
    
    # Remap all the iob ids to their original labels
    # ----------------------------------------------
    # the iob ids and their respective labels
    id2label = {}
    for i in range(1, len(iob_mapping)):
        id2label.update({i : convert_iobIndex_to_baseEntity(i, labels)})
    
    def format_to_gliner(example):
        """
        Formats the dataset into a format that can be processed by gliner
        """

        tokens = example["tokens"]
        ner_tags = example["ner_tags"]

        ner = []
        current_entity = None
        for idx, tag in enumerate(ner_tags):
            if tag in id2label:
                if current_entity is None:
                    current_entity = [idx, idx, id2label[tag]]
                elif (
                    tag == ner_tags[current_entity[0]]
                    or tag == ner_tags[current_entity[0]] + 1
                ):
                    current_entity[1] = idx
                else:
                    ner.append(current_entity)
                    current_entity = [idx, idx, id2label[tag]]
            else:
                if current_entity is not None:
                    ner.append(current_entity)
                    current_entity = None

        if current_entity is not None:
            ner.append(current_entity)

        # helps solve the edge case where there are no entities which would lead to a training error
        if len(ner) > 0:
            return {"tokenized_text": tokens, "ner": ner}
        else:
            return {"tokenized_text": tokens, "ner": ner, "labels": labels}

    train_dataset = [format_to_gliner(eg) for eg in ds["train"].to_list()]
    eval_dataset = [format_to_gliner(eg) for eg in ds["validation"].to_list()]

    msg.info(f"Sample training data: {train_dataset[0]}")

    # Setting up the Training arguments
    # ---------------------------------

    data_size = len(train_dataset)
    num_batches = data_size // batch_size
    num_epochs = max(1, num_steps // num_batches)
    learning_rate = 5e-6
    num_training_steps = num_epochs * (data_size // batch_size * gradient_accumulation_steps) 
    num_warmup_steps = int(0.1 * num_training_steps)

    msg.info(
        f"Finetuning the {base_model} model saving checkpoints to {checkpoint_dir}"
    )

    # Perform training

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = GLiNER.from_pretrained(base_model, labels=labels)
    model._keys_to_ignore_on_save = []

    data_collator = DataCollator(
        model.config,
        data_processor=model.data_processor,
        prepare_labels=True,
    )
    model.to(device)

    tokenizer = model.data_processor.transformer_tokenizer

    # setup GaLore
    non_galore_params, galore_params = setup_GaLore_params(model)

    optimizer = GaLoreAdamW( 
        [
            {'params': non_galore_params}, # Non-linear layers
            {
                'params': galore_params,
                'rank': 512,
                'update_proj_gap': num_training_steps // 4,
                'scale': 4,
                'proj_type': 'std'
                # 'proj_type': 'continuous' # For subspace descent (faster but kinda worse performance)
                # 'names': [None] * len(galore_params)
            }
        ],
        lr= learning_rate,
        weight_decay=0.01
        
    )

    lr_scheduler = get_scheduler(
        name="linear", # or "cosine", "polynomial", "constant_with_warmup", etc.
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        learning_rate=learning_rate,
        # weight_decay=0.01,
        # others_lr=1e-5,
        # others_weight_decay=0.01,
        # lr_scheduler_type="linear",  # cosine
        # warmup_ratio=0.1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,

        num_train_epochs=num_epochs,
        # evaluation_strategy="steps", # deprecated
        eval_strategy="steps",
        eval_steps=int(num_steps * 0.1),
        
        # checkpoint saving strat
        save_strategy="steps",
        save_steps=int(num_steps * 0.1),
        save_total_limit=10,
        

        logging_strategy="steps",
        logging_steps=int(num_steps * 0.1),
        
        dataloader_num_workers=0,
        use_cpu=False,
        report_to="none",
        # load_best_model_at_end=True, # might fix some training errors
        # torch_empty_cache_steps=2,
        gradient_accumulation_steps=gradient_accumulation_steps,

        # galore optimizer
        # from here: https://huggingface.co/blog/galore
        # optim="galore_adamw",
        # optim_target_modules=["q_proj", "v_proj"],
        # for reproducibility
        data_seed=random_seed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
        callbacks=[ModelCallback],
        optimizers=(optimizer, lr_scheduler)
    )

    # we train first
    trainer.train()

    # 3. Get the path to the best model checkpoint
    best_checkpoint_path = trainer.state.best_model_checkpoint

    if best_checkpoint_path is not None:
        # 4. Load the weights from the best checkpoint into your model
        msg.info(f"Loading best model from {best_checkpoint_path}")
        
        # Use your model's native loading method (GLiNER.from_pretrained)
        # to load the checkpoint weights.
        model = GLiNER.from_pretrained(best_checkpoint_path, labels=labels)
        model._keys_to_ignore_on_save = [] # Re-add the fix
        
        # Move the newly loaded model back to the correct device
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
    else:
        # If no evaluation was run or if the best model wasn't tracked
        msg.warn("Could not find the best model checkpoint. Saving the final model.")
        # The 'model' variable still holds the weights from the last training step

    # 5. Save the (now best-performing) model and tokenizer to the final output directory
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    msg.good(f"Best model saved to {output_dir}")

    # trainer.train()
    # trainer.save_model(str(output_dir))
    # tokenizer.save_pretrained(output_dir)
    # msg.good(f"Best model saved to {output_dir}")
    

    if push_to_hub: # temporarily disabled
        msg.info(f"Pushing model to HuggingFace Hub")
        model = GLiNER.from_pretrained(output_dir)
        model.push_to_hub(push_to_hub, token=api_token)

def convert_iobIndex_to_baseEntity(iob_idx : int, labels: list) -> str | None:
    """
    Converts the iob index (the keys of the iob_mapping) 
    into their original labels (without -I or -B)

    Parameters
    ----------
    iob_idx : int
        The index of the given iob label.
    labels : list[str]
       The list of labels specified in the label mappings

    Returns
    -------
    str
        The original label without the -I or -B suffix.
    None
        If the iob_idx is 0 or invalid.
    """
    # wizardry
    # reverses the iob_mapping formula
    label_idx = math.floor((iob_idx-1) / 2)

    # edge case, this shouldn't be reached
    if iob_idx <= 0 or label_idx >= len(labels):
        return None
    
    return labels[label_idx]

def fix_seed(seed : int) :
    set_seed(seed) 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # for reproducibility across runs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True) # broken might need to fix

def setup_GaLore_params(model):
    galore_params = []
    non_galore_params = []
    # Include all 2D layers you want GaLore on (e.g., FFN weights, Attention weights)
    GALORE_TARGET_MODULES = ["query_proj.weight", "value_proj.weight", "key_proj.weight", "dense.weight", "out_project.weight", "project_start.weight", "project_end.weight"]
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            non_galore_params.append(param)
            continue
            
        is_galore_target = any(target in name for target in GALORE_TARGET_MODULES)
        
        is_2d_matrix = param.dim() >= 2
        
        if is_galore_target and is_2d_matrix:
            galore_params.append(param)
        else:
            non_galore_params.append(param)

    return non_galore_params, galore_params
        


class ModelCallback(TrainerCallback):
    
    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        model_name = model.__class__.__name__ if model else "UnknownModel"

        info = f"Starting training for {model_name} " 
        info += f"Training with {args.num_train_epochs} epochs, batch size {args.per_device_train_batch_size}"
        msg.info(info + "\n")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            info = "\n📊 Evaluation at step {state.global_step}"
            if "eval_loss" in metrics:
                info += (f"  Loss: {metrics['eval_loss']:.4f}")
            if "eval_precision" in metrics:
                info += (f"  Precision: {metrics['eval_precision']:.4f}")
            if "eval_recall" in metrics:
                info += (f"  Recall: {metrics['eval_recall']:.4f}")
            if "eval_f1" in metrics:
                info += (f"  F1: {metrics['eval_f1']:.4f}")
            msg.info(info + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print(f"Step {state.global_step} | {logs}")


if __name__ == "__main__":
    typer.run(main)
