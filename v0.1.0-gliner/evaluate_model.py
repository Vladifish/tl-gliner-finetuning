from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, Optional

import spacy
import srsly
import torch
import typer
from datasets import Dataset, load_dataset, load_from_disk
from spacy.scorer import Scorer
from spacy.tokens import Doc, Span
from spacy.training import Example
from wasabi import msg

import math

label_map = srsly.read_json("assets/mapped_labels.json")

if label_map is None:
        msg.fail("mapped_labels.json not found! Cannot convert entity type", exits=1)

labels = label_map["labels"]
msg.info(labels)

def main(
    # fmt: off
    output_path: Path = typer.Argument(..., help="Path to store the metrics in JSON format."),
    model_name: str = typer.Option("ljvmiranda921/tl_gliner_small", show_default=True, help="GliNER model to use for evaluation."),
    dataset: str = typer.Option("./model/gliner_small", help="Dataset to evaluate upon."),
    threshold: float = typer.Option(0.5, help="The threshold of the GliNER model (controls the degree to which a hit is considered an entity)."),
    dataset_config: Optional[str] = typer.Option(None, help="Configuration for loading the dataset."),
    chunk_size: int = typer.Option(250, help="Size of the text chunk to be processed at once."),
    local_dataset: str = typer.Option(None, help="The name of the local dataset. Leave blank if the dataset would be retrieved remotely."),
    random_seed: int = typer.Option(42, help="Random seed for reproducibility."),
    # fmt: on
):

    # Process the test data first
    msg.info("Processing test dataset")

    if local_dataset is None:
        ds = load_dataset(dataset, dataset_config, split="test")
    else:
        corpus_path = Path(f"assets/corpus/{str(local_dataset)}/dataset_full")
        ds = load_from_disk(corpus_path)["test"] # this is a dataset dictionary anyways
    
    ref_docs = convert_hf_to_spacy_docs(ds)

    msg.info("Loading GliNER model")
    nlp = spacy.blank("tl")
    nlp.add_pipe(
        "gliner_spacy",
        config={
            "gliner_model": model_name,
            "chunk_size": chunk_size,
            "labels": labels,
            "threshold": threshold,
            "style": "ent",
            "map_location": "cuda" if torch.cuda.is_available() else "cpu",
        },
    )
    msg.text("Getting predictions")
    docs = deepcopy(ref_docs)
    pred_docs = list(nlp.pipe(docs))
    pred_docs = [update_entity_labels(doc) for doc in pred_docs]

    # Get the scores
    examples = [
        Example(reference=ref, predicted=pred) for ref, pred in zip(ref_docs, pred_docs)
    ]
    scores = Scorer.score_spans(examples, "ents")

    msg.info(f"Results for {dataset} ({model_name})")
    msg.text(scores)
    srsly.write_json(output_path, data=scores, indent=2)
    msg.good(f"Saving outputs to {output_path}")

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


def convert_hf_to_spacy_docs(dataset: "Dataset") -> Iterable[Doc]:
    nlp = spacy.blank("tl")
    examples = dataset.to_list()
    entity_types = {
        int(idx): convert_iobIndex_to_baseEntity(int(idx))
        for idx in label_map["iob_mapping"].keys()
        if idx != "0"
    }
    msg.text(f"Using entity types: {entity_types}")

    docs = []
    for example in examples:
        tokens = example["tokens"]
        ner_tags = example["ner_tags"]
        doc = Doc(nlp.vocab, words=tokens)

        entities = []
        start_idx = None
        entity_type = None

        for idx, tag in enumerate(ner_tags):
            if tag in entity_types:
                if start_idx is None:
                    start_idx = idx
                    entity_type = entity_types[tag]
                elif entity_type != entity_types.get(tag, None):
                    entities.append(Span(doc, start_idx, idx, label=entity_type))
                    start_idx = idx
                    entity_type = entity_types[tag]
            else:
                if start_idx is not None:
                    entities.append(Span(doc, start_idx, idx, label=entity_type))
                    start_idx = None

        if start_idx is not None:
            entities.append(Span(doc, start_idx, len(tokens), label=entity_type))
        doc.ents = entities
        docs.append(doc)

    return docs

def update_entity_labels(doc: Doc) -> Doc:
    updated_ents = []
    for ent in doc.ents:
        new_label = ent.label_
        updated_span = Span(doc, ent.start, ent.end, label=new_label)
        updated_ents.append(updated_span)

    new_doc = Doc(
        doc.vocab,
        words=[token.text for token in doc],
        spaces=[token.whitespace_ for token in doc],
    )
    new_doc.ents = updated_ents
    return new_doc


if __name__ == "__main__":
    typer.run(main)
