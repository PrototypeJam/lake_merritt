# scripts/generate_fdl_dataset.py
from __future__ import annotations
import csv, json, random, uuid
from pathlib import Path

def main():
    script_dir = Path(__file__).resolve().parent
    templates_path = script_dir / "fdl_templates.json"
    data_dir = script_dir.parent / "data"
    data_dir.mkdir(exist_ok=True)
    output_path = data_dir / "duty_of_loyalty_benchmark.csv"

    templates = json.loads(templates_path.read_text(encoding="utf-8"))
    rows = []

    for tpl in templates:
        base_meta = {
            "scenario_id": tpl["scenario_id"],
            "notes": tpl.get("notes", ""),
            "requires_disclosure": tpl.get("requires_disclosure", False),
        }

        # Create the choices list with labels
        choices = [
            (tpl["choice_a"], tpl["label_a"]),
            (tpl["choice_b"], tpl["label_b"]),
            (tpl["choice_c"], tpl["label_c"])
        ]
        
        # Shuffle the choices to randomize their positions
        random.shuffle(choices)

        # Map the shuffled choices to a, b, c positions
        shuffled = {
            "choice_a": choices[0][0], "label_a": choices[0][1],
            "choice_b": choices[1][0], "label_b": choices[1][1],
            "choice_c": choices[2][0], "label_c": choices[2][1],
        }

        # Helper function to find choice text by label
        def get_choice_by_label(label):
            for i, (choice_text, choice_label) in enumerate(choices):
                if choice_label == label:
                    return choice_text
            return ""

        # Create ambiguous variant row
        rows.append({
            "item_id": str(uuid.uuid4()), 
            "variant": "ambiguous",
            "input": f'{tpl["question_core"]}\n\n{tpl["ambiguous_prompt"]}',
            "correct_label": "unknown",
            "expected_output": get_choice_by_label("unknown"),  # For ambiguous, expected is the "unknown" choice
            **base_meta, 
            **shuffled
        })
        
        # Create disambiguated variant row
        correct_label_disambig = tpl["correct_label_for_disambiguated"]
        rows.append({
            "item_id": str(uuid.uuid4()), 
            "variant": "disambiguated",
            "input": f'{tpl["question_core"]}\n\nDetails: {tpl["disambiguated_details"]}\n\nSelect the best choice.',
            "correct_label": correct_label_disambig,
            "expected_output": get_choice_by_label(correct_label_disambig),  # For disambiguated, expected is the correct choice
            **base_meta, 
            **shuffled
        })

    # Write to CSV with expected_output included
    with output_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "item_id", "scenario_id", "variant", "input", "expected_output",
            "choice_a", "choice_b", "choice_c",
            "label_a", "label_b", "label_c",
            "correct_label", "requires_disclosure", "notes"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"CSV includes expected_output column for Mode B compatibility")

if __name__ == "__main__":
    main()