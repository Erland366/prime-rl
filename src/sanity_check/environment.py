import verifiers as vf
from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are a stop talker engine. When given any question, you will return EOS immediately without saying anything. Do not say anything else, just return EOS."
)

def _build_dataset(
    dataset_path: str,
    dataset_name: str | None = None,
    max_num_examples: int | None = None,
    streaming: bool = False,
    question_column_name: str = "question",
    split: str = "train",
):
    ds = load_dataset(dataset_path, name=dataset_name, split=split, streaming=streaming)

    if streaming:
        ds = ds.take(max_num_examples) if max_num_examples is not None else ds
    else:
        ds = ds.shuffle(seed=0).select(range(max_num_examples)) if max_num_examples is not None else ds

    def _format_example(example: dict) -> dict[str, str]:
        return {
            "question": example[question_column_name],
            "answer": "",
        }

    return ds.map(_format_example)

class StopTalkerEnv(vf.SingleTurnEnv):
    def __init__(
        self,
        dataset_path: str,
        dataset_name: str | None = None,
        max_num_train_examples: int | None = None,
        max_num_eval_examples: int | None = None,
        train_split_name: str = "train",
        eval_split_name: str = "validation",
        streaming: bool = False,
        question_column_name: str = "question",
    ):
        parser = vf.Parser(lambda text: text.strip())
        rubric = vf.Rubric(parser=parser)

        async def stop_talker_reward_func(completion: vf.Messages, answer: str, **kwargs) -> float:
            prediction = parser.parse_answer(completion)
            if prediction is None:
                return 0.0

            return -1 * len(prediction.strip())

        rubric.add_reward_func(stop_talker_reward_func)
        rubric.add_metric(parser.get_format_reward_func(), weight=0.0)

        super().__init__(
            dataset=_build_dataset(
                dataset_path,
                dataset_name,
                max_num_train_examples,
                streaming,
                question_column_name,
                split=train_split_name,
            ),
            eval_dataset=_build_dataset(
                dataset_path,
                dataset_name,
                max_num_eval_examples,
                streaming,
                question_column_name,
                split=eval_split_name,
            ),
            system_prompt=SYSTEM_PROMPT,
            parser=parser,
            rubric=rubric,
            env_id="sanity-check",
            env_args={
                "dataset_path": dataset_path,
                "dataset_name": dataset_name,
                "train_split_name": train_split_name,
                "eval_split_name": eval_split_name,
                "max_num_train_examples": max_num_train_examples,
                "max_num_eval_examples": max_num_eval_examples,
                "streaming": streaming,
                "question_column_name": question_column_name,
            },
        )

def load_environment(**kwargs) -> StopTalkerEnv:
    return StopTalkerEnv(**kwargs)
