"""
This script is used to evaluate the fine-tuned models after the
hyper-parameter search is done to find the best fine-tune parameters. 
It contains functions that evaluate the models based on their BLEU 
score (higher is better) and validation loss (less is better).

Because we need to maximise BLEU score and minimise validation loss,
the scores are normalised and a combined score is calculated (higher is better).
"""

def calculate_model_score(bleu_score, validation_loss):
    """
    Calculate a combined score for a model based on its BLEU score and validation loss.
    Return a score representing the model's performance.
    """

    # Max values for the metrics
    max_bleu=1.0
    max_val_loss=2.0
    
    # Normalise the BLEU score and validation loss
    normalised_bleu = bleu_score / max_bleu
    normalised_val_loss = validation_loss / max_val_loss
    
    # Calculate the combined score
    combined_score = normalised_bleu - normalised_val_loss
    
    return combined_score

def evaluate_model():
    """
    An interactive function to evaluate fine-tuned models based 
    on their BLEU score and validation loss. Prints the results.
    """
    model_num = 0
    more = 'yes'
    models_info = []

    while more != 'no':
        more = input("\nMore models? ('p' for pruned model. 'no' to quit): ").lower()
        if more == 'p':
            models_info.append(f"Model [{model_num}] PRUNED")
            print("pruned!")
        elif more != 'no':
            val_loss = float(input("Enter validation loss: "))
            bleu_score = float(input("Enter BLEU score: "))
            models_info.append(f"Model [{model_num}] combined score: {calculate_model_score(bleu_score, val_loss)}")

        model_num += 1

    # Print results
    for model in models_info:
        print(model)

evaluate_model()