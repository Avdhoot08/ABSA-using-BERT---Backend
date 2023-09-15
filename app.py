from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import torch

from absa import ABSA
from glue_utils import ABSAProcessor, InputExample, convert_examples_to_seq_features
from parameters import Parameters
from transformers import BertConfig, BertTokenizer

from seq_utils import ot2bieos_ts, tag2ts, bio2ot_ts
from train import load_and_cache_examples


args = Parameters()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device

_, model_class, tokenizer_class = BertConfig, ABSA, BertTokenizer

model = model_class.from_pretrained(args.output_directory)
tokenizer = tokenizer_class.from_pretrained(args.output_directory)
model.to(args.device)
model.eval()


def predict(pdevice, pmodel, ptokenizer, tagging_format, sentence):
    processor = ABSAProcessor()
    label_list = processor.get_labels(tagging_format)
    words = sentence.split()
    placeholder_labels = ["O" for i in range(len(words))]
    input_sentence = InputExample(
        guid=1, text_a=sentence, text_b=None, label=placeholder_labels
    )
    features = convert_examples_to_seq_features(
        examples=[input_sentence],
        label_list=label_list,
        tokenizer=ptokenizer,
        cls_token_at_end=False,
        cls_token=ptokenizer.cls_token,
        sep_token=ptokenizer.sep_token,
        cls_token_segment_id=0,
        pad_on_left=False,
        pad_token_segment_id=0,
    )
    total_preds, gold_labels = None, None
    idx = 0

    if tagging_format == "BIEOS":
        absa_label_vocab = {
            "O": 0,
            "EQ": 1,
            "B-POS": 2,
            "I-POS": 3,
            "E-POS": 4,
            "S-POS": 5,
            "B-NEG": 6,
            "I-NEG": 7,
            "E-NEG": 8,
            "S-NEG": 9,
            "B-NEU": 10,
            "I-NEU": 11,
            "E-NEU": 12,
            "S-NEU": 13,
        }
    elif tagging_format == "BIO":
        absa_label_vocab = {
            "O": 0,
            "EQ": 1,
            "B-POS": 2,
            "I-POS": 3,
            "B-NEG": 4,
            "I-NEG": 5,
            "B-NEU": 6,
            "I-NEU": 7,
        }
    elif tagging_format == "OT":
        absa_label_vocab = {"O": 0, "EQ": 1, "T-POS": 2, "T-NEG": 3, "T-NEU": 4}
    else:
        raise Exception("Invalid tagging schema %s..." % tagging_format)
    absa_id2tag = {}
    for k in absa_label_vocab:
        v = absa_label_vocab[k]
        absa_id2tag[v] = k
    input_tensors = [
        torch.tensor([features[0].input_ids], dtype=torch.long),
        torch.tensor([features[0].input_mask], dtype=torch.long),
        torch.tensor([features[0].segment_ids], dtype=torch.long),
        torch.tensor([features[0].label_ids], dtype=torch.long),
    ]
    single_input = tuple(t.to(pdevice) for t in input_tensors)
    with torch.no_grad():
        inputs = {
            "input_ids": single_input[0],
            "attention_mask": single_input[1],
            "token_type_ids": single_input[2],
            "labels": single_input[3],
        }
        evaluate_label_ids = [features[0].evaluate_label_ids]
        outputs = pmodel(**inputs)
        # logits: (1, seq_len, label_size)
        logits = outputs[1]
        # predictions: (1, seq_len)
        predictions = np.argmax(logits.detach().cpu().numpy(), axis=-1)
        label_indices = evaluate_label_ids[idx]
        # words = total_words[idx]
        predicted_labels = predictions[0][label_indices]
        assert len(words) == len(predicted_labels)
        predicted_tags = [absa_id2tag[label] for label in predicted_labels]

        if tagging_format == "OT":
            predicted_tags = ot2bieos_ts(predicted_tags)
        elif tagging_format == "BIO":
            predicted_tags = ot2bieos_ts(bio2ot_ts(predicted_tags))
        else:
            # current tagging schema is BIEOS, do nothing
            pass
        p_ts_sequence = tag2ts(ts_tag_sequence=predicted_tags)
        output_ts = []
        output = []
        for t in p_ts_sequence:
            beg, end, sentiment = t
            aspect = words[beg : end + 1]
            output_ts.append("%s: %s" % (aspect, sentiment))
            output.append({"aspect": " ".join(aspect), "sentiment": sentiment})
            # output[" ".join(aspect)] = sentiment
        if inputs["labels"] is not None:
            # for the unseen data, there is no ``labels''
            if gold_labels is None:
                gold_labels = inputs["labels"].detach().cpu().numpy()
            else:
                gold_labels = np.append(
                    gold_labels, inputs["labels"].detach().cpu().numpy(), axis=0
                )
    idx += 1
    return output


def evaluate(review):
    return predict(device, model, tokenizer, "BIO", review)


app = Flask(__name__)
CORS(app)


@app.post("/review")
def review():
    review = request.json["review"]
    return jsonify({"text": review, "result": evaluate(review)})


@app.post("/")
def analyze():
    payload = request.json
    aggregation = {}
    response = []
    for review in payload:
        results = evaluate(review["text"])
        for result in results:
            clean_aspect = result["aspect"].strip(" .,").lower()
            if aggregation.get(clean_aspect, None) is None:
                aggregation[clean_aspect] = {"POS": 0, "NEG": 0, "NEU": 0}
            aggregation[clean_aspect][result["sentiment"]] = (
                aggregation.get(clean_aspect, {}).get(result["sentiment"], 0) + 1
            )
        response.append({"id": review["id"], "text": review["text"], "result": results})
    return jsonify({"reviews": response, "aggregation": aggregation})


if __name__ == "__main__":
    os.environ["FLASK_ENV"] = "development"
    app.run(debug=True)
