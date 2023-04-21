# M7
PBR23M7

Group members: Nikodem Kropielnicki, Bogusława Tlołka, Joanna Wdziękońska

Project based on [VulCurator: A Vulnerability-Fixing Commit Detector](https://www.researchgate.net/publication/365271012_VulCurator_a_vulnerability-fixing_commit_detector)

[Missing files](https://drive.google.com/drive/folders/18usdkhUGGeJv-KajKBC3FLotrVKzw8oX?usp=sharing)

[Overleaf](https://www.overleaf.com/project/6401cbc79c98f06ccca972de)

[Team Policies and Team Expectations Agreement](https://docs.google.com/document/d/1NiMzeDmkhrVOwA-ww2HNvBGpNvi6pWlYKUMZcJOjwcc/edit?fbclid=IwAR0SU23NwHS6iq3GWpPhtfsw6Alw2vJIvw-Ev0GwSviptOUNEaUJNSfjK6I)

[GoogleColab](https://colab.research.google.com/drive/1AECPUDhOZEPUq2euqjAt7CRaEUZHo-O8#scrollTo=RcjCkM9xGdpe)

Progress is tracked in Github Projects

# Reproduction of results:

Note: Some files were too big to put them on Github, they are in the missing files link above (on GoogleColab there is a section which uploads them to project)

For Tensorflow dataset:

To train message classifier: 

`python message_classifier.py --dataset_path tf_vuln_dataset.csv --model_path model/tf_message_classifier.sav`

To train issue classifier:

`python issue_classifier.py --dataset_path tf_vuln_dataset.csv --model_path model/tf_issue_classifier.sav`

To finetune CodeBERT for patch classifier: 

`python vulfixminer_finetune.py --dataset_path tf_vuln_dataset.csv --finetune_model_path model/tf_patch_vulfixminer_finetuned_model.sav`

To train patch classifier: 

`python vulfixminer.py --dataset_path tf_vuln_dataset.csv --model_path model/tf_patch_vulfixminer.sav --finetune_model_path model/tf_patch_vulfixminer_finetuned_model.sav --train_prob_path probs/tf_patch_vulfixminer_train_prob.txt --test_prob_path probs/tf_patch_vulfixminer_test_prob.txt`

To run ensemble classifier: 

`python variant_ensemble.py --config_file tf_dataset.conf`

Similarly, for SAP dataset (some classifiers about 1,5h on GoogleColab!):

To train message classifier: 

`python message_classifier.py --dataset_path sub_enhanced_dataset_th_100.txt --model_path model/sap_message_classifier.sav`

To train issue classifier: 

`python issue_classifier.py --dataset_path sub_enhanced_dataset_th_100.txt --model_path model/sap_issue_classifier.sav`

To finetune CodeBERT for patch classifier: 
`python vulfixminer_finetune.py --dataset_path sap_patch_dataset.csv --finetune_model_path model/sap_patch_vulfixminer_finetuned_model.sav`

To train patch classifier: 

`python vulfixminer.py --dataset_path sap_patch_dataset.csv --model_path model/sap_patch_vulfixminer.sav --finetune_model_path model/sap_patch_vulfixminer_finetuned_model.sav --train_prob_path probs/sap_patch_vulfixminer_train_prob.txt --test_prob_path probs/sap_patch_vulfixminer_test_prob.txt`

To run ensemble classifier: 

`python variant_ensemble.py --config_file sap_dataset.conf`
