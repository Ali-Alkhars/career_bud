import evaluate

"""
This script is used to evaluate the outputs of
models automatically on three metrics:
- BLEU
- ROUGE
"""

def eval_outputs(model_outputs, ground_truths):
    metrics = ["bleu", "rouge"]

    for m in metrics:
        metric = evaluate.load(m)
        results = metric.compute(predictions=model_outputs, references=ground_truths)

        print(f"\n{m.upper()}: {results}")


predictions = ["hello there general kenobi", "foo bar foobar"]
references = [["hello there general kenobi", "hello there !"],["foo bar foobar"]]
eval_outputs(predictions, references)