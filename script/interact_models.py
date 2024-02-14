import os
import json
import pandas as pd
import torch
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
)
import krippendorff
from scipy.stats import pointbiserialr, pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from evaluate import load
from interact_llm import send_query


texts = [
    "Hi! How are you tonight?",
    "I've been better.",
    "I just buried my dog yesterday.",
    "She was going to turn 15 next month",
    "Ouch! I'm so sorry to hear that.",
    "I never lost a dog, but I've lost cats.",
    "Is 14 a good age for a dog?",
    "I have my first dog so I don't know much, ",
    "he is 9 years old and came with our house.",
    "Actually, yes. She was old and had dementia.",
    "The oldest dog I evet had was 15.",
    "I'm so sorry, losing a pet is very hard on us.",
    "It’s happy to have a dog."
]

def inference_abstract_model(texts, model_path, batch_size=16):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

    # add prefix
    for id, text in enumerate(texts):
        texts[id] = f"abstraction: {text}"

    generated_sents = []
    for idx in tqdm(range(0, len(texts), batch_size)):
        batch = texts[idx: idx+batch_size]
        input_ids = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(**input_ids, num_beams=10, max_length=500)
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        generated_sents += preds

    for sent_id, sent in enumerate(generated_sents):
        generated_sents[sent_id] = generated_sents[sent_id].replace("<pad>", "").replace("</s>", "").strip()
        
    return generated_sents


def inference_classifier_model(texts, model, tokenizer, batch_size=16):
    device = model.device

    predicted_labels = []
    probabilities = []
    with torch.no_grad():
        for idx in range(0, len(texts), batch_size):
            batch = texts[idx: idx + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            logits = model(**inputs).logits
            prob = torch.nn.functional.softmax(logits, dim=1)
            predicted_class_id = prob.argmax(dim=1)
            preds = [model.config.id2label[id.item()] for id in predicted_class_id]
            predicted_labels += preds
            probabilities += prob.tolist()


    if len(texts)==len(predicted_labels):
        return predicted_labels, probabilities
    else:
        raise Exception("lengths of input and output are not matched ")

def inference_dependence_dataset(dataset_path, save_path, model, tokenizer):
    dataset = json.load(open(dataset_path, "r"))
    new_dataset = []
    for example in tqdm(dataset):
        example["dependence_utterance"] = []
        response = example["response"]
        history = example["history"][:-1]
        for i in range(len(history)):
            start_idx = history[i].index(":") + 1
            history[i] = history[i][start_idx:].strip()

        texts = []
        for utterance in history:
            text = utterance + " </s> <s> " + response
            texts.append(text)
        print(texts)
        predicted_labels, labels_probability = inference_classifier_model(texts, model=model, tokenizer=tokenizer, batch_size=16)
        for text, label, prob in zip(texts, predicted_labels, labels_probability):
            if label == 1:
                example["dependence_utterance"].append([text.split("</s> <s>")[0].strip(), prob[label]])
        new_dataset.append(example)
    json.dump(new_dataset, open(save_path, "w+"), indent=4)


def inference_conditional_dependence_dataset(dataset_path, save_path, model, tokenizer):
    dataset = json.load(open(dataset_path, "r"))
    new_dataset = []
    for example in tqdm(dataset):
        example["conditional_dependence_utterances"] = []
        history = example["history"]
        dependent_utterances = example["dependence_utterance"]
        response = example["response"]

        for first_utterance in dependent_utterances:
            texts = []
            for second_utterance in dependent_utterances:
                if first_utterance[0] != second_utterance[0]:
                    text = first_utterance[0] + " </s> <s> " + second_utterance[0] + " </s> <s> " + response
                    texts.append(text)
            predicted_labels, labels_probability = inference_classifier_model(texts, model=model, tokenizer=tokenizer, batch_size=16)

            dep_count = 0
            label_1_probs = []
            for text, label, probs in zip(texts, predicted_labels, labels_probability):
                # if first utterance is conditional dependent with all second utterance, first utterance is a direct cause.
                label_1_probs.append(probs[1])
                if label == 1:
                    dep_count += 1
            # whether first utterance close to response
            #in_scope = False
            #for utterance in history[-5:]:
            #    if first_utterance[0].strip() in utterance:
            #        in_scope = True
            if dep_count == len(texts):
                example["conditional_dependence_utterances"].append((first_utterance[0], label_1_probs))

        new_dataset.append(example)
    json.dump(new_dataset, open(save_path, "w+"), indent=4)

def create_combinations(l):
    combinations = []
    for ele1 in l:
        for ele2 in l:
            if ele1 != ele2:
                combinations.append((ele1, ele2))
    return combinations

def inference_twoVariableConditional_dependence_dataset(dataset_path, save_path, model, tokenizer):
    dataset = json.load(open(dataset_path, "r"))
    new_dataset = []
    for example in tqdm(dataset):
        example["twoVariableConditional_dependence_utterances"] = []
        history = example["history"]
        dependent_utterances = example["conditional_dependence_utterances"]
        response = example["response"]

        if len(dependent_utterances) < 3:
            example["twoVariableConditional_dependence_utterances"] = dependent_utterances
        else:
            for first_index, first_utterance in enumerate(dependent_utterances):
                texts = []
                twoVariableCombinations = create_combinations(dependent_utterances[:first_index] + dependent_utterances[first_index+1:])
                for variables in twoVariableCombinations:
                    if first_utterance != variables[0] and first_utterance != variables[1]:
                        text = first_utterance + " </s> <s> " + variables[0] + " </s> <s> " + variables[1] + " </s> <s> " + response
                        texts.append(text)
                predicted_labels, labels_probability  = inference_classifier_model(texts, model=model, tokenizer=tokenizer, batch_size=16)

                dep_count = 0
                for text, label in zip(texts, predicted_labels):
                    # if first utterance is conditional dependent with all second utterance, first utterance is a direct cause.
                    if label == 1:
                        dep_count += 1
                # whether first utterance close to response
                #in_scope = False
                #for utterance in history[-10:]:
                #    if first_utterance.strip() in utterance:
                #        in_scope = True
                if dep_count > 1:
                    example["twoVariableConditional_dependence_utterances"].append(first_utterance)

        new_dataset.append(example)
    json.dump(new_dataset, open(save_path, "w+"), indent=4)


def conditional_classifier_Estep(dependent_classifier="models/roberta_ESConv_dependent_classifier",
                                 conditional_dependent_classifier="models/roberta_ESConv_conditionalCI_classifier",
                                 old_dataset_file="datasets/ESConv/relevance_eval/EM_datasets/ESConv_train_causal_EM0.json",
                                 new_dataset_file="datasets/ESConv/relevance_eval/EM_datasets/ESConv_train_causal_EM1.json"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path=dependent_classifier
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    inference_dependence_dataset(dataset_path=old_dataset_file, save_path=new_dataset_file,
                                 model=model, tokenizer=tokenizer)

    model_path=conditional_dependent_classifier
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    inference_conditional_dependence_dataset(dataset_path=new_dataset_file, save_path=new_dataset_file,
                                            model=model, tokenizer=tokenizer)

def create_pseudo_classification_dataset(file_path, train_save_path, valid_save_path, true_train_dataset_path,
                                         true_valid_dataset_path):
    dataset = json.load(open(file_path, "r"))
    positive_pairs = []
    negative_pairs = []
    irrelevant_utterance_candidates = [
        "I think that's kind of an awful thing to say. I didn't realize there was a height requirement for forest rangers.",
        "I understand. My mom and dad were scientist so now, that is what I do",
        "Thank you so much. I used you as a reference on my application. I hope that is ok!",
        "You are very talented. I wish my rat tail soup would be popular.",
        "Wow. That is a lot of paper!",
        "Just recently. I am still working on my own app. Is anyone else in on your app idea?"
    ]
    for example in dataset:
        if len(example["conditional_dependence_utterances"]) > 2:
            conditional_dependence_utterances = example["conditional_dependence_utterances"]
            history = example["history"]
            response = example["response"]
            dependent_utterances = example["dependence_utterance"]
            for i in range(len(dependent_utterances)):
                dependent_utterances[i] = dependent_utterances[i][0].strip()

            for i in range(len(conditional_dependence_utterances)):
                conditional_dependence_utterances[i] = conditional_dependence_utterances[i].strip()

            for i in range(len(history)):
                start_idx = history[i].index(":") + 1
                history[i] = history[i][start_idx:].strip()

            direct_cause_utterances = []
            for conditional_utterance in conditional_dependence_utterances:
                if conditional_utterance in history[-5:]:
                    direct_cause_utterances.append(conditional_utterance)

            irrelevant_utterances = []
            for utterance in history[:-2]:
                if (utterance not in direct_cause_utterances) and (utterance not in dependent_utterances):
                    irrelevant_utterances.append(utterance)

            for cause_utterance in direct_cause_utterances:
                for dependent_utterance in dependent_utterances:
                    if cause_utterance != dependent_utterance:
                        positive_pairs.append(
                            (cause_utterance + " </s> <s> " + dependent_utterance + " </s> <s> " + response, 1))
                        while True:
                            if len(irrelevant_utterances) > 0:
                                random_irrelevant_utter = random.sample(irrelevant_utterances, 1)[0]
                            else:
                                random_irrelevant_utter = random.sample(irrelevant_utterance_candidates, 1)[0]
                            if random_irrelevant_utter.strip() != cause_utterance:
                                negative_pairs.append((random_irrelevant_utter.strip() + " </s> <s> " +
                                                       dependent_utterance + " </s> <s> " + response, 0))
                                break
    print(len(positive_pairs), len(negative_pairs))
    negative_pairs = negative_pairs[:len(positive_pairs)]
    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)
    valid_idx = int(len(all_pairs) * 0.9)
    train_pairs = all_pairs[:valid_idx]
    valid_pairs = all_pairs[valid_idx:]
    print(len(train_pairs), len(valid_pairs))

    train_dataset = pd.DataFrame(train_pairs, columns=["utterance", "label"])
    valid_dataset = pd.DataFrame(valid_pairs, columns=["utterance", "label"])

    true_train_dataset = pd.read_csv(true_train_dataset_path)
    true_valid_dataset = pd.read_csv(true_valid_dataset_path)

    train_dataset = pd.concat([train_dataset, true_train_dataset])
    valid_dataset = pd.concat([valid_dataset, true_valid_dataset])

    train_dataset.to_csv(train_save_path, index=False)
    valid_dataset.to_csv(valid_save_path, index=False)


def compute_causal_metric_score(read_path, CI_model, I_model, CI_tokenizer, I_tokenizer, save_path):
    print(read_path)
    dataset = json.load(open(read_path, "r"))
    new_dataset = []
    system_names = {"alpaca_lora_response", "Blenderbot_400M_response", "Blenderbot_ft_response", "human_response"}
    # compute response A metric scores
    for system in system_names:
        for item in dataset:
            response_key = f"response_from_{system}"
            if response_key in item:
                response = item[response_key]["utterance"]
                history = item["history"]
                item[f"{response_key}_dependence_utterance"] = []
                item[f"{response_key}_conditional_dependence_utterances"] = []
                # compute dependence score
                texts = []
                for utterance in history:
                    text = utterance["utterance"] + " </s> <s> " + response
                    texts.append(text)
                predicted_labels, labels_probability = inference_classifier_model(texts, model=I_model, tokenizer=I_tokenizer,
                                                                                  batch_size=16)
                for text, label, prob in zip(texts, predicted_labels, labels_probability):
                    if label == 1:
                        item[f"{response_key}_dependence_utterance"].append([text.split("</s> <s>")[0].strip(), prob[label]])
                # compute conditional dependence score
                dependent_utterances = item[f"{response_key}_dependence_utterance"]
                for first_utterance in dependent_utterances:
                    texts = []
                    for second_utterance in dependent_utterances:
                        if first_utterance[0] != second_utterance[0]:
                            text = first_utterance[0] + " </s> <s> " + second_utterance[0] + " </s> <s> " + response
                            texts.append(text)
                    predicted_labels, labels_probability = inference_classifier_model(texts, model=CI_model,
                                                                                      tokenizer=CI_tokenizer, batch_size=16)

                    dep_count = 0
                    label_1_probs = []
                    for text, label, probs in zip(texts, predicted_labels, labels_probability):
                        # if first utterance is conditional dependent with all second utterance, first utterance is a direct cause.
                        label_1_probs.append(probs[1])
                        # if label == 1:
                        #     dep_count += 1
                    # whether first utterance close to response
                    #in_scope = False
                    #for utterance in history[-5:]:
                    #    if first_utterance[0].strip() in utterance:
                    #        in_scope = True
                    # if dep_count == len(texts):
                    item[f"{response_key}_conditional_dependence_utterances"].append((first_utterance[0], label_1_probs))
                new_dataset.append(item)
    json.dump(new_dataset, open(save_path, "w+"), indent=4)

def top_k_elements(lst, k=3):
    """
    Return the top k elements of a list in descending order.

    Args:
    lst (list): The list from which to find the top k elements.
    k (int): The number of top elements to return.

    Returns:
    list: A list containing the top k elements.
    """
    if not isinstance(lst, list):
        lst = [lst]
    # Sort the list in descending order and return the first k elements
    return sorted(lst, reverse=True)[:k]

def compute_causal_metric_score_2(read_path, CI_model, I_model, CI_tokenizer, I_tokenizer, save_path, score_name="causal_score"):
    print(read_path)
    dataset = json.load(open(read_path, "r"))
    new_dataset = []
    system_names = ["alpaca_lora_response", "Blenderbot_400M_response", "Blenderbot_ft_response", "human_response"]
    pairs = []
    for i in range(len(system_names)):
        for j in range(i+1, len(system_names)):
            pairs.append([system_names[i], system_names[j]])
    print(pairs)
    # compute causal score of each response
    for item in dataset:
        history = item["history"]
        for system in system_names:
            response = item[system]["utterance"]
            dependent_utterances = []
            # compute dependence score
            texts = []
            for utterance in history:
                text = utterance["utterance"] + " </s> <s> " + response
                texts.append(text)
            predicted_labels, labels_probability = inference_classifier_model(texts, model=I_model,
                                                                              tokenizer=I_tokenizer,
                                                                              batch_size=16)
            for text, label, prob in zip(texts, predicted_labels, labels_probability):
                if label == 1:
                    dependent_utterances.append([text.split("</s> <s>")[0].strip(), prob[label]])
            # compute conditional dependence score
            conditional_dependence_utterances = []
            if len(dependent_utterances) > 1:
                for first_utterance in dependent_utterances:
                    texts = []
                    for second_utterance in dependent_utterances:
                        if first_utterance[0] != second_utterance[0]:
                            text = first_utterance[0] + " </s> <s> " + second_utterance[0] + " </s> <s> " + response
                            texts.append(text)
                    predicted_labels, labels_probability = inference_classifier_model(texts, model=CI_model,
                                                                                      tokenizer=CI_tokenizer,
                                                                                      batch_size=16)
                    label_1_probs = []
                    for text, label, probs in zip(texts, predicted_labels, labels_probability):
                        label_1_probs.append(probs[1])
                    conditional_dependence_utterances.append((first_utterance[0], label_1_probs))
            else:
                conditional_dependence_utterances = dependent_utterances
            top_k=3
            # compute evaluation score of response
            dependent_scores = []
            for dependent_utterance in dependent_utterances:
                dependent_scores.append(dependent_utterance[1])
            conditional_scores = []
            for causal_utterance in conditional_dependence_utterances:
                conditional_scores.append(np.mean(top_k_elements(causal_utterance[1], top_k)))

            if len(dependent_scores) > 0 and len(conditional_scores) > 0:
                item[system][score_name] = (np.mean(top_k_elements(dependent_scores, top_k)) + np.mean(conditional_scores))/2
            elif len(dependent_scores) > 0 and len(conditional_scores) == 0 :
                item[system][score_name] = np.mean(top_k_elements(dependent_scores, top_k))/2
            else:
                item[system][score_name] = 0
        for pair in pairs:
            item[f"{pair[0]}-vs-{pair[1]}"][pair[0]][score_name] = item[pair[0]][score_name]
            item[f"{pair[0]}-vs-{pair[1]}"][pair[1]][score_name] = item[pair[1]][score_name]

        new_dataset.append(item)
    json.dump(new_dataset, open(save_path, "w+"), indent=4)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

datasets_name = ["dream", "ESConv", "msc"] #"msc", "dream", "ESConv",
for dataset_name in datasets_name:
    print(dataset_name)
    # load models
    I_model_path=f"models/roberta_{dataset_name}_dependent_classifier"
    I_tokenizer = AutoTokenizer.from_pretrained(I_model_path)
    I_model = AutoModelForSequenceClassification.from_pretrained(I_model_path).to(device)

    CI_model_path=f"models/roberta_{dataset_name}_conditionalCI_classifier"
    CI_tokenizer = AutoTokenizer.from_pretrained(CI_model_path)
    CI_model = AutoModelForSequenceClassification.from_pretrained(CI_model_path).to(device)
    
    compute_causal_metric_score_2(read_path=f"datasets/human_evaluation_dataset/pairwise_comparison/{dataset_name}-round1-dependentScore.json",
                                CI_model=CI_model, I_model=I_model, CI_tokenizer=CI_tokenizer, I_tokenizer=I_tokenizer,
                                save_path=f"datasets/human_evaluation_dataset/pairwise_comparison/{dataset_name}-round1-dependentScore.json",
                                  score_name="causal_score")
    data = json.load(open(f"datasets/human_evaluation_dataset/pairwise_comparison/{dataset_name}-round1-dependentScore.json", "r"))


