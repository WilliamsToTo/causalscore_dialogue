import os
from interact_models import conditional_classifier_Estep, create_pseudo_classification_dataset

def ESConv_EM_train():
    print("ESConv_EM_train")
    start_step = 0
    EM_steps = 10
    for i in range(start_step, start_step+EM_steps):
        print(f"Starting iteration {i}")
        #E step
        conditional_classifier_Estep(dependent_classifier="models/roberta_ESConv_dependent_classifier",
                                 conditional_dependent_classifier=f"models/roberta_ESConv_conditionalCI_classifier_EM{i}",
                                 old_dataset_file=f"datasets/ESConv/relevance_eval/EM_datasets/ESConv_train_causal_EM{i}.json",
                                 new_dataset_file=f"datasets/ESConv/relevance_eval/EM_datasets/ESConv_train_causal_EM{i+1}.json")

        create_pseudo_classification_dataset(file_path=f"datasets/ESConv/relevance_eval/EM_datasets/ESConv_train_causal_EM{i+1}.json",
                                     train_save_path=f"datasets/ESConv/relevance_eval/EM_datasets/ESConv_train_causal_EM{i+1}.csv",
                                     valid_save_path=f"datasets/ESConv/relevance_eval/EM_datasets/ESConv_valid_causal_EM{i+1}.csv",
                                     true_train_dataset_path="datasets/ESConv/relevance_eval/train_condition_tc_dataset.csv",
                                     true_valid_dataset_path="datasets/ESConv/relevance_eval/valid_condition_tc_dataset.csv")

        # M step
        M_step_signal = os.system(f"python script/run_text_classification.py "
                                  f"--train_file  datasets/ESConv/relevance_eval/EM_datasets/ESConv_train_causal_EM{i+1}.csv "
                                  f"--validation_file datasets/ESConv/relevance_eval/EM_datasets/ESConv_valid_causal_EM{i+1}.csv "
                                  f"--model_name_or_path models/roberta_ESConv_conditionalCI_classifier_EM{i} "
                                  f"--output_dir models/roberta_ESConv_conditionalCI_classifier_EM{i+1} "
                                  f"--num_train_epochs 1")
        print(f"Step {i} M signal: ", M_step_signal)

ESConv_EM_train()
