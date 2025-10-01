import typer
import warnings
from pathlib import Path
import srsly

def generate_iob(output_path: Path = typer.Option(Path("../../assets/"), help="Path to save the IOB mapping JSON file.")):
    """
    Generates IOB mapping for the given labels and saves it to a JSON file. 
    The labels are hardcoded, and should just be edited here directly.
    """

    import os; 
    os.makedirs(f'../assets/', exist_ok=True); 

    labels = [
        "Person-Individual",
        "Person-Collective",
        "Organization-Political",
        "Organization-Government",
        "Organization-Military",
        "Organization-Other",
        "Location",
        "Object",
        "Time",
        "Event-Local",
        "Event-International",
        "Production-Media",
        "Production-Government",
        "Production-Doctrine",
        "Numerical Statistics"
    ]

    label_map = {}
    label_map.update({"labels": labels})

    IOB_mapping = {}
    IOB_mapping.update({0: "O"})

    i = 1
    for label in labels:
        IOB_mapping.update({i: f"B-{label}"})
        i += 1
        IOB_mapping.update({i: f"I-{label}"})
        i += 1

    label_map.update({"iob_mapping": IOB_mapping})
    srsly.write_json(f"{output_path}/mapped_labels.json", data=label_map, indent=2)

if __name__ == "__main__":
    typer.run(generate_iob)
