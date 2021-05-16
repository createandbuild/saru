from tqdm import tqdm

import transformers
import datasets

import textattack
from textattack.attack_results.skipped_attack_result import SkippedAttackResult
from textattack.attack_results.failed_attack_result import FailedAttackResult

from textattack.models.tokenizers import AutoTokenizer
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import HuggingFaceDataset

from build_attack import build_attack


def generate_adv_examples(model, dataset, max_examples=None):
    # Create attack
    attack = build_attack(model)

    # Run attack
    indices = range(max_examples) if max_examples else range(len(dataset))
    results_iterable = attack.attack_dataset(dataset, indices)

    # Print attack results
    attack_log_manager = textattack.loggers.AttackLogManager()
    attack_log_manager.enable_stdout()
    adv_examples = []

    for result in results_iterable:
        # print(result.__str__(color_method='ansi'))
        attack_log_manager.log_result(result)

        if isinstance(result, SkippedAttackResult):
            adv_examples.append(None)
            continue
        elif isinstance(result, FailedAttackResult):
            adv_examples.append(None)
            continue
        adv_examples.append(result.perturbed_text(color_method=None))

    attack_log_manager.log_summary()
    attack_log_manager.flush()
    print()

    return adv_examples

if __name__ == "__main__":
    # create model & dataset
    model = transformers.AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    tokenizer = AutoTokenizer("cardiffnlp/twitter-roberta-base-sentiment")
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    dataset = HuggingFaceDataset('tweet_eval', 'sentiment', 'test')
    custom_dataset = [
        ('The time has come to fight back. Whether you hold a weapon, pen, keyboard or donate money to the pro-democracy movement, everyone must do their bit for the revolution to succeed', 0),
        ('Lu Siwei, lawyer who assisted one of the 12 Hong Kongers detained in the mainland for attempting to flee to Taiwan and had his legal license revoked, is barred from leaving China to attend fellowship in U.S.', 0),
        ('Taiwan’s democratic credentials are undisputable. Liberal democracies can, and should, develop their own bilateral and multilateral exchanges with Taiwan, even while pushing for its inclusion in the WHO and other UN bodies.', 0)
    ]
    
    adv_examples = generate_adv_examples(model_wrapper, custom_dataset)
