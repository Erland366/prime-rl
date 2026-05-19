import verifiers as vf


def test_stop_talker_loads_from_package():
    env = vf.load_environment(
        "sanity-check",
        dataset_path="openai/gsm8k",
        dataset_name="main",
        max_num_train_examples=3,
        max_num_eval_examples=2,
        train_split_name="train",
        eval_split_name="test",
        streaming=False,
        question_column_name="question",
    )

    assert env.env_id == "sanity-check"
    assert env.env_args["dataset_path"] == "openai/gsm8k"
    assert env.env_args["dataset_name"] == "main"
    assert env.env_args["max_num_train_examples"] == 3
    assert env.env_args["max_num_eval_examples"] == 2
    assert env.env_args["train_split_name"] == "train"
    assert env.env_args["eval_split_name"] == "test"
    assert env.env_args["streaming"] is False
    assert env.env_args["question_column_name"] == "question"
