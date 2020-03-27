import train
import pre_input
import eval


def auto_main():
    models = ["inceptionresnetv1", "mobilenetv2"]
    seeds = [31, 42]
    batch_sizes = [4, 8, 16, 32]
    max_steps_array = [2000, 4000]
    dropouts = [0.85]
    learning_rates = [0.01, 0.001, 0.0001]
    success_boundaries = [0.7, 0.75, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999]

    round_id = 0
    for model in models:
        # PreInputs processes
        pre_input.main(model)
        for seed in seeds:
            for batch_size in batch_sizes:
                for max_steps in max_steps_array:
                    for dropout in dropouts:
                        for learning_rate in learning_rates:
                            """Train"""
                            train.deploy(model_name=model, seed=seed, batch_size=batch_size, max_steps=max_steps,
                                         dropout=dropout, learning_rate=learning_rate)
                            for success_boundary in success_boundaries:
                                """Eval"""
                                train.deploy(model_name=model, seed=seed, batch_size=batch_size, max_steps=max_steps,
                                             dropout=dropout, learning_rate=learning_rate)
                                eval.deploy(model_name=model, seed=seed, batch_size=batch_size, max_steps=max_steps,
                                            dropout=dropout, learning_rate=learning_rate, success_boundary=success_boundary)

                                """Update Id number."""
                                round_id = round_id + 1

                                """Save data in the array"""



    # TODO: save data in a csv
if __name__ == "__main__":
    auto_main()
