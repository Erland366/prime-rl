import verifiers as vf


def test_smoke_reverse_text_loads_from_package():
    env = vf.load_environment(
        "smoke-reverse-text",
        num_train_examples=3,
        num_eval_examples=2,
        min_length=5,
        max_length=7,
        seed=123,
    )

    assert env.env_id == "smoke-reverse-text"
    assert env.env_args["num_train_examples"] == 3
    assert env.env_args["num_eval_examples"] == 2
    assert env.env_args["min_length"] == 5
    assert env.env_args["max_length"] == 7
    assert env.env_args["seed"] == 123
