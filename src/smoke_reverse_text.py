import random
import string

from datasets import Dataset

import verifiers as vf

SYSTEM_PROMPT = (
    "You are a precise string reversal engine. Reverse the provided string character by character "
    "and reply with only the reversed string."
)
ALPHABET = string.ascii_lowercase + string.digits


def _make_example(rng: random.Random, min_length: int, max_length: int) -> dict[str, str]:
    source = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(min_length, max_length)))
    return {
        "question": f"Reverse this string exactly:\n{source}",
        "answer": source[::-1],
    }


def _build_dataset(num_examples: int, min_length: int, max_length: int, seed: int) -> Dataset:
    rng = random.Random(seed)
    return Dataset.from_list([_make_example(rng, min_length, max_length) for _ in range(num_examples)])


class SmokeReverseTextEnv(vf.SingleTurnEnv):
    def __init__(
        self,
        num_train_examples: int = 512,
        num_eval_examples: int = 64,
        min_length: int = 4,
        max_length: int = 12,
        seed: int = 0,
    ):
        parser = vf.Parser(lambda text: text.strip())
        rubric = vf.Rubric(parser=parser)

        async def exact_reverse_match(completion: vf.Messages, answer: str, **kwargs) -> float:
            prediction = parser.parse_answer(completion)
            if prediction is None:
                return 0.0
            return float(prediction.strip() == answer.strip())

        rubric.add_reward_func(exact_reverse_match)
        rubric.add_metric(parser.get_format_reward_func(), weight=0.0)

        super().__init__(
            dataset=_build_dataset(num_train_examples, min_length, max_length, seed),
            eval_dataset=_build_dataset(num_eval_examples, min_length, max_length, seed + 1),
            system_prompt=SYSTEM_PROMPT,
            parser=parser,
            rubric=rubric,
            env_id="smoke-reverse-text",
            env_args={
                "num_train_examples": num_train_examples,
                "num_eval_examples": num_eval_examples,
                "min_length": min_length,
                "max_length": max_length,
                "seed": seed,
            },
        )


def load_environment(**kwargs) -> SmokeReverseTextEnv:
    return SmokeReverseTextEnv(**kwargs)
