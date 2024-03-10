import evaluate

"""
This script is used to evaluate the outputs of
models automatically on three metrics:
- BLEU
- ROUGE
- METEOR
"""

def eval_outputs(model_outputs, ground_truths):
    metrics = ["bleu", "rouge", "meteor"]

    for m in metrics:
        metric = evaluate.load(m)
        results = metric.compute(predictions=model_outputs, references=ground_truths)

        print(f"\n{m.upper()}: {results}")


predictions = ["hello there general kenobi", "foo bar foobar"]
references = [["hello there general kenobi"],["foo bar foobar"]]
eval_outputs(predictions, references)