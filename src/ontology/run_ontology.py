from designs.ontology_parser import OntologyParser

import os
import pdb
from pathlib import Path
from owlready2 import * 


def main():
    
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2] 
    project_root_str = str(project_root)

    if "src" not in str(project_root_str): 
        parent_dir = project_root_str + "/src/ontology"
    assets_dir = os.path.join(parent_dir, "assets")
    ontology_path = os.path.join(assets_dir,"ontologies" )
    file = os.path.join(ontology_path, "in_cabin_ontology.owl")
    
    dataset_path = project_root / "dataset" 
    dataset_file = dataset_path / "test_set_ontology.csv"
    parser = OntologyParser(file)
    message = parser.parse_observations(dataset_path=dataset_file)
    parser.print_results()
    print(message)

if __name__ == "__main__":
    main()
