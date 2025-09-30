import os
from pathlib import Path
from typing import Optional

import torch
import typer
from datasets import load_dataset, load_from_disk
from gliner import GLiNER
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from gliner.data_processing.collator import DataCollator
from gliner.training import Trainer, TrainingArguments
from wasabi import msg
import srsly

import math

def main(
    # fmt: off
    base_model: str = typer.Argument(..., help="Base model used for training."),
    output_dir: Path = typer.Argument(..., help="Path to store the output model."),
    checkpoint_dir: Path = typer.Option(Path("checkpoints"), help="Path for storing checkpoints."),
    push_to_hub: Optional[str] = typer.Option(None, help="If set, will upload the trained model to the provided Huggingface model namespace."),
    num_steps: int = typer.Option(500, help="Number of steps to run training."),
    batch_size: int = typer.Option(8, help="Batch size used for training."),
    dataset: str = typer.Option("etdvprg/gold-ml-batch1", help="Path to the PHMartialLawNER dataset."),
    local_dataset: str = typer.Option(None, help="The name of the local dataset. Leave blank if the dataset would be retrieved remotely."),
    size: str = typer.Option("small", help="Size of the GLiNER model to use."),
    # fmt: on
):  
    # set up model storage in the proper directory
    os.makedirs(f'v0.1.0-gliner/models/gliner_{size}', exist_ok=True); 
    os.makedirs(f'v0.1.0-gliner/checkpoints/ckpt_gliner_gliner_{size}', exist_ok=True);

    # setting to decide if dataset should be pushed
    if push_to_hub:
        api_token = os.getenv("HF_TOKEN")
        if not api_token:
            msg.fail("HF_TOKEN is missing! Won't be able to --push-to-hub", exits=1)

    # Load and Format the dataset
    msg.info(f"Formatting the {dataset} dataset")
    
    if local_dataset:
        corpus_path = Path(f"experiments/corpus/{str(local_dataset)}")
        ds = load_from_disk(corpus_path)
        if ds is None:
            msg.fail(f"Dataset :: {local_dataset} not found! Exiting", exits=1)
    else:
        ds = load_dataset(dataset)

    # Retrieve the NER labels
    label_map = srsly.read_json("assets/mapped_labels.json")

    if label_map is None:
            msg.fail("mapped_labels.json not found! Cannot convert entity type", exits=1)
            return # to ease the compiler
    
    # label wizardy to be used in format_to_gliner
    labels = label_map["labels"]
    iob_mapping = label_map["iob_mapping"]
    def convert_iobIndex_to_baseEntity(iob_idx : int) -> str | None:
        """
        Converts the iob index (the keys of the iob_mapping) 
        into their original labels (without -I or -B)
        """
        # edge case, this shouldn't be reached
        if iob_idx == 0:
            return None
        
        # wizardry
        label_idx = math.floor((iob_idx-1) / 2)
        return labels[label_idx]
    
    # the iob ids and their respective labels
    id2label = {}
    for i in range(1, len(iob_mapping)):
        id2label.update({i : convert_iobIndex_to_baseEntity(i)})
    
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

        return {"tokenized_text": tokens, "ner": ner}

    train_dataset = [format_to_gliner(eg) for eg in ds["train"].to_list()]
    eval_dataset = [format_to_gliner(eg) for eg in ds["validation"].to_list()]

    # Perform training
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = GLiNER.from_pretrained(base_model)

    data_collator = DataCollator(
        model.config,
        data_processor=model.data_processor,
        prepare_labels=True,
    )
    model.to(device)

    data_size = len(train_dataset)
    num_batches = data_size // batch_size
    num_epochs = max(1, num_steps // num_batches)

    msg.info(
        f"Finetuning the {base_model} model, saving checkpoints to {checkpoint_dir}"
    )

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        learning_rate=5e-6,
        weight_decay=0.01,
        others_lr=1e-5,
        others_weight_decay=0.01,
        lr_scheduler_type="linear",  # cosine
        warmup_ratio=0.1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        # evaluation_strategy="steps", # deprecated
        eval_strategy="steps",
        save_steps=num_steps * 2,
        save_total_limit=10,
        dataloader_num_workers=0,
        use_cpu=False,
        report_to="none",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    msg.good(f"Best model saved to {output_dir}")

    if push_to_hub:
        msg.info(f"Pushing model to HuggingFace Hub")
        model = GLiNER.from_pretrained(output_dir)
        model.push_to_hub(push_to_hub, token=api_token)


if __name__ == "__main__":
    typer.run(main)
