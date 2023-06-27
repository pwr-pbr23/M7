from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaModel, AutoTokenizer, DataCollatorWithPadding, \
    AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from torch import nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch import cuda
from sklearn import metrics
import numpy as np
from transformers import AdamW
from transformers import get_scheduler
from patch_entities import VulFixMinerFileDataset
from model import VulFixMinerFineTuneClassifier
from tqdm import tqdm
import pandas as pd
from utils import get_code_version
import config
import argparse
import utils

# dataset_name = 'sap_patch_dataset.csv'
# FINE_TUNED_MODEL_PATH = 'model/patch_variant_2_finetuned_model.sav'

dataset_name = None
FINE_TUNED_MODEL_PATH = None

directory = os.path.dirname(os.path.abspath(__file__))

commit_code_folder_path = os.path.join(directory, 'commit_code')

model_folder_path = os.path.join(directory, 'model')

# rerun with 5 finetune epoch
FINETUNE_EPOCH = 5

LIMIT_FILE_COUNT = 5

NUMBER_OF_EPOCHS = 5
TRAIN_BATCH_SIZE = 4
VALIDATION_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
EARLY_STOPPING_ROUND = 5

TRAIN_PARAMS = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
VALIDATION_PARAMS = {'batch_size': VALIDATION_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
TEST_PARAMS = {'batch_size': TEST_BATCH_SIZE, 'shuffle': True, 'num_workers': 8}

LEARNING_RATE = 1e-5

use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
random_seed = 109
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

CODE_LENGTH = 256
HIDDEN_DIM = 768
HIDDEN_DIM_DROPOUT_PROB = 0.1
NUMBER_OF_LABELS = 2

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def get_input_and_mask(tokenizer, code):
    inputs = tokenizer(code, padding='max_length', max_length=CODE_LENGTH, truncation=True, return_tensors="pt")

    return inputs.data['input_ids'][0], inputs.data['attention_mask'][0]


def predict_test_data(model, testing_generator, device, need_prob=False):
    print("Testing...")
    y_pred = []
    y_test = []
    urls = []
    probs = []
    model.eval()
    with torch.no_grad():
        for id_batch, url_batch, input_batch, mask_batch, label_batch in tqdm(testing_generator):
            input_batch, mask_batch, label_batch \
                = input_batch.to(device), mask_batch.to(device), label_batch.to(device)

            outs = model(input_batch, mask_batch)
            outs = F.softmax(outs, dim=1)
            y_pred.extend(torch.argmax(outs, dim=1).tolist())
            y_test.extend(label_batch.tolist())
            probs.extend(outs[:, 1].tolist())
            urls.extend(list(url_batch))
        precision = metrics.precision_score(y_pred=y_pred, y_true=y_test)
        recall = metrics.recall_score(y_pred=y_pred, y_true=y_test)
        f1 = metrics.f1_score(y_pred=y_pred, y_true=y_test)
        mcc = metrics.matthews_corrcoef(y_pred=y_pred, y_true=y_test)
        try:
            auc = metrics.roc_auc_score(y_true=y_test, y_score=probs)
        except Exception:
            auc = 0

    print("Finish testing")
    if not need_prob:
        return precision, recall, f1, auc, mcc
    else:
        return precision, recall, f1, auc, urls, probs, mcc


def train(model, learning_rate, number_of_epochs, training_generator, test_generator):
    loss_function = nn.NLLLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = number_of_epochs * len(training_generator)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    train_losses = []

    for epoch in range(number_of_epochs):
        model.train()
        total_loss = 0
        current_batch = 0
        for id_batch, url_batch, input_batch, mask_batch, label_batch in tqdm(training_generator):
            input_batch, mask_batch, label_batch \
                = input_batch.to(device), mask_batch.to(device), label_batch.to(device)
            outs = model(input_batch, mask_batch)
            outs = F.log_softmax(outs, dim=1)
            loss = loss_function(outs, label_batch)
            train_losses.append(loss.item())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.detach().item()

            current_batch += 1
            if current_batch % 50 == 0:
                print("Train commit iter {}, total loss {}, average loss {}".format(current_batch, np.sum(train_losses),
                                                                                    np.average(train_losses)))

        print("epoch {}, training commit loss {}".format(epoch, np.sum(train_losses)))
        train_losses = []

        model.eval()

        print("Result on testing dataset...")
        precision, recall, f1, auc, mcc = predict_test_data(model=model,
                                                            testing_generator=test_generator,
                                                            device=device)

        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))
        print("AUC: {}".format(auc))
        print("MCC: {}".format(mcc))
        print("-" * 32)

        if epoch + 1 == FINETUNE_EPOCH:
            torch.save(model.state_dict(), FINE_TUNED_MODEL_PATH)
            if not isinstance(model, nn.DataParallel):
                model.freeze_codebert()
            else:
                model.module.freeze_codebert()
    return model


def retrieve_patch_data(all_data, all_label, all_url):
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    print("Preparing tokenizer data...")

    id_to_label = {}
    id_to_url = {}
    id_to_input = {}
    id_to_mask = {}
    for i, diff in tqdm(enumerate(all_data)):
        added_code = get_code_version(diff=diff, added_version=True)
        deleted_code = get_code_version(diff=diff, added_version=False)

        code = added_code + tokenizer.sep_token + deleted_code

        input_ids, mask = get_input_and_mask(tokenizer, [code])
        id_to_input[i] = input_ids
        id_to_mask[i] = mask
        id_to_label[i] = all_label[i]
        id_to_url[i] = all_url[i]

    return id_to_input, id_to_mask, id_to_label, id_to_url


def read_tensor_flow_dataset(dataset_name, need_url_data=False):
    print("Reading dataset...")
    df = pd.read_csv(dataset_name)

    df = df[['commit_id', 'repo', 'msg', 'filename', 'diff', 'label', 'partition']]

    url_data, label_data = utils.get_data(dataset_name)

    items = df.to_numpy().tolist()

    url_to_msg, url_to_partition, url_to_label = {}, {}, {}

    for item in items:
        commit_id = item[0]
        repo = item[1]
        url = repo + '/commit/' + commit_id
        partition = item[6]
        message = item[2]

        if pd.isnull(message):
            message = ' '

        label = item[5]
        pl = 'UNKNOWN'

        url_to_msg[url] = message
        url_to_label[url] = label
        url_to_partition[url] = partition

    message_train, message_test, label_train, label_test, url_train, url_test = [], [], [], [], [], []

    for i, url in enumerate(url_data['train']):
        message_train.append(url_to_msg[url])
        label_train.append(label_data['train'][i])
        url_train.append(url)

    for i, url in enumerate(url_data['test']):
        message_test.append(url_to_msg[url])
        label_test.append(label_data['test'][i])
        url_test.append(url)

    if not need_url_data:
        return message_train, message_test, label_train, label_test
    else:
        return message_train, message_test, label_train, label_test, url_train, url_test


def preprocess_function(examples):
    return tokenizer(examples, truncation=True)


import numpy as np
from datasets import load_metric, load_dataset


def compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}


def do_train(args):
    global dataset_name, FINE_TUNED_MODEL_PATH

    dataset_name = args.dataset_path

    FINE_TUNED_MODEL_PATH = args.finetune_model_path

    print("Dataset name: {}".format(dataset_name))
    print("Saving model to: {}".format(FINE_TUNED_MODEL_PATH))

    # patch_data, label_data, url_data = get_data(dataset_name)

    message_train, message_test, label_train, label_test = read_tensor_flow_dataset(dataset_name)

    # tokenized_train = message_train.map(preprocess_function, batched=True)
    # tokenized_test = message_test.map(preprocess_function, batched=True)

    message_train = tokenizer(message_train, truncation=True)
    message_test = tokenizer(message_test, truncation=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir="M7",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=NUMBER_OF_EPOCHS,
        weight_decay=0.01,
        save_strategy="epoch",
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=message_train,
        eval_dataset=message_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_path',
                        type=str,
                        required=True,
                        help='name of dataset')
    parser.add_argument('--finetune_model_path',
                        type=str,
                        required=True,
                        help='select path to save model')

    args = parser.parse_args()

    do_train(args)
