import os
import itertools
from seqeval.metrics import classification_report


def docwise_reshape(predict_dataset, predictions, labels):
    n_docs = len(predict_dataset)
    per_example_predictions = predictions.reshape(n_docs, -1)
    per_example_labels = labels.reshape(n_docs, -1)
    print(f"2D shaped prediction have size: {per_example_predictions.shape}")
    print(f"2D shaped labels have size: {per_example_labels.shape}")

    return per_example_predictions, per_example_labels


def ids2strs(seq_of_preds, seq_of_labels, label_list):
    true_preds_and_labels = [[(label_list[p], label_list[l]) for (p, l) in zip(doc_preds, doc_labels) if l != -100] for doc_preds, doc_labels in zip(seq_of_preds, seq_of_labels)]

    doc_level_preds, doc_level_labels = [], []
    for doc_preds_and_labels in true_preds_and_labels:
        docwise_preds, docwise_labels = list(zip(*doc_preds_and_labels))
        doc_level_preds.append(list(docwise_preds))
        doc_level_labels.append(list(docwise_labels))

    return doc_level_preds, doc_level_labels


def chunkwise_output(predict_dataset, docwise_preds, docwise_labels, label_list):
    n_docs = len(predict_dataset)
    chunkwise_input_output = []
    for i in range(n_docs):
        doc_tokens = predict_dataset[i]["tokens"]
        doc_id = predict_dataset[i]["document_id"]
        chunk_id = predict_dataset[i]["chunk_id"]
        true_preds_and_labels = [(label_list[p], label_list[l]) for (p, l) in
                                 zip(docwise_preds[i], docwise_labels[i]) if l != -100]
        doc_level_preds, doc_level_labels = list(zip(*true_preds_and_labels))
        chunkwise_input_output.append((doc_tokens, doc_level_preds, doc_level_labels, doc_id, chunk_id))

    return chunkwise_input_output


def wrap_tag(content, tag, options=''):
    return f"<{tag}{options}>{content}</{tag}>"


def visualize_predictions(chunkwise_input_output, training_args):
    html_dir = os.path.join(training_args.output_dir, "predictions")
    os.makedirs(html_dir)

    fn_color, fp_color, tp_color, fpfn_color = "#572cff", "#ff6a00", "#20b86b", "#ff0000"
    trunc_color = "#808080"
    opacity = "0.9"

    for doc_id, chunks in itertools.groupby(chunkwise_input_output, key=lambda k: k[-2]):  # k[-2] is doc id
        # for i, doc in enumerate(chunk_level_input_output):
        html_output_predictions_file = os.path.join(html_dir, f"{doc_id}_output.html")
        content = ''
        with open(html_output_predictions_file, "w") as writer:
            doc_content = wrap_tag(doc_id, "h1")
            content += doc_content
            for chunk in chunks:
                tokens, str_predictions, str_labels, _, chunk_id = chunk
                content += wrap_tag(f"Chunk {chunk_id}", "h2")
                content += wrap_tag("Tokens", "h3")

                text_content = ''
                for tk in range(len(tokens)):
                    token = tokens[tk]
                    prediction = str_predictions[tk] if len(str_predictions) > tk else "-"
                    label = str_labels[tk] if len(str_labels) > tk else "-"
                    if prediction == label == "O":
                        style = ''
                    elif prediction == "-" or label == "-":
                        style = f' style="background-color: {trunc_color}; opacity: {opacity};"'
                    elif prediction == label:
                        # TP
                        style = f' style="background-color: {tp_color}; opacity: {opacity};"'
                    elif prediction == "O" and label != "O":
                        # FN
                        style = f' style="background-color: {fn_color}; opacity: {opacity};"'
                    elif prediction != "O" and label == "O":
                        # FP
                        style = f' style="background-color: {fp_color}; opacity: {opacity};"'
                    else:
                        # both FP and FN: i.e. MISPREDICTION
                        style = f' style="background-color: {fpfn_color}; opacity: {opacity};"'

                    text_content += wrap_tag(f" {token}", "span", style)

                content += wrap_tag(text_content, "p")

                content += wrap_tag("Raw Targets", "h3")
                truncated_tokens = [wrap_tag("-", "span",
                                             f' style="background-color: {trunc_color}; opacity: {opacity};"')] * (
                                               len(tokens) - len(str_labels))
                extended_str_labels = list(str_labels) + truncated_tokens
                target_content = wrap_tag(" ".join(extended_str_labels), "p")
                content += target_content

                content += wrap_tag("Raw Predictions", "h3")
                truncated_tokens = [wrap_tag("-", "span",
                                             f' style="background-color: {trunc_color}; opacity: {opacity};"')] * (
                                               len(tokens) - len(str_predictions))
                extended_str_preds = list(str_predictions) + truncated_tokens
                preds_content = wrap_tag(" ".join(extended_str_preds), "p")
                content += preds_content

            content += wrap_tag("Legend", "h4")
            content += wrap_tag("The '-' symbol means truncated", "p",
                                f' style="background-color: {trunc_color}; opacity: {opacity};"')
            content += wrap_tag("TP", "p", f' style="background-color: {tp_color}; opacity: {opacity};"')
            content += wrap_tag("FP", "p", f' style="background-color: {fp_color}; opacity: {opacity};"')
            content += wrap_tag("FN", "p", f' style="background-color: {fn_color}; opacity: {opacity};"')
            content += wrap_tag("both FP and FN: Misprediction", "p",
                                f' style="background-color: {fpfn_color}; opacity: {opacity};"')

            body = wrap_tag(content, "body")
            writer.write(body)


def seqeval_metrics(preds, truths):
    print("Seqeval Report")
    return classification_report(truths, preds, digits=3)

