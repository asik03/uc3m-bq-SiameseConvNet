import os

import numpy as np

import train
import pre_input
import eval
import time

root_dir = "D:/PycharmProjects/uc3m-bq-SiameseConvNet/data/"
is_pre_input = False
is_train = True
is_eval = True

csv_header = "Id, Model, Seed, BatchSize, MaxSteps, DropoutKeepProb, LearningRate, DiscriminationThreshold, TP, TN, FP, FN, TrainTime, EvalTime"


def auto_main(_id, _models, _seeds, _batch_sizes, _max_steps_array, _learning_rates, _success_boundaries,
              _dropouts=[0.85]):

    if not os.path.exists(root_dir + "results/" + str(_id) + "/"):
        os.makedirs(root_dir + "results/" + str(_id) + "/")

    results_dir = root_dir + "results/" + str(_id) + "/"
    result_list = []
    round_id = 0
    for model in _models:
        # PreInputs processes
        if is_pre_input:
            pre_input.main(model)
        for seed in _seeds:
            for batch_size in _batch_sizes:
                for max_steps in _max_steps_array:
                    for dropout in _dropouts:
                        for learning_rate in _learning_rates:
                            """Train"""
                            train_time = 0
                            if is_train:
                                train_time = time.time()
                                train.deploy(model_name=model, seed=seed, batch_size=batch_size, max_steps=max_steps,
                                             dropout=dropout, learning_rate=learning_rate)
                                train_time = time.time() - train_time
                            if is_eval:
                                for success_boundary in _success_boundaries:
                                    """Eval"""
                                    start_eval_time = time.time()
                                    eval_result = eval.deploy(model_name=model, seed=seed, batch_size=batch_size,
                                                              max_steps=max_steps,
                                                              dropout=dropout, learning_rate=learning_rate,
                                                              success_boundary=success_boundary)
                                    """Update Id number."""
                                    round_id = round_id + 1

                                    """Save data in the array"""
                                    eval_result = np.concatenate((eval_result, [train_time]))
                                    eval_result = np.concatenate((eval_result, [time.time() - start_eval_time]))
                                    eval_result = np.concatenate(([round_id], eval_result))
                                    result_list.append(eval_result)
                            # save in cvs:
                            np.savetxt(
                                results_dir + "results_" + str(round_id) +
                                ".csv", np.array(result_list), delimiter=',', fmt='%s',
                                header=csv_header, comments="")
    # save in cvs:
    print(result_list)
    np.savetxt(results_dir + 'final_results.csv', np.array(result_list),
               delimiter=',',
               fmt='%s', header=csv_header, comments="")


'''
    models = ["inceptionresnetv2"]
    batch_sizes = [8]
    max_steps_array = [250]
    dropouts = [0.85]
    learning_rates = [0.001]


'''
if __name__ == "__main__":
    seeds = [13, 25, 29, 31, 42, 51, 67, 80, 90]
    success_boundaries = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.995, 0.999]
    dropout_keep_probs = [0.85]

    auto_main(
        _id=10,
        _models=["inceptionresnetv1"],
        _seeds=seeds,
        _batch_sizes=[8],
        _max_steps_array=[250],
        _learning_rates=[0.001],
        _success_boundaries=success_boundaries,
        _dropouts=dropout_keep_probs)

    auto_main(
        _id=11,
        _models=["inceptionresnetv2"],
        _seeds=seeds,
        _batch_sizes=[16],
        _max_steps_array=[500],
        _learning_rates=[0.001],
        _success_boundaries=success_boundaries,
        _dropouts=dropout_keep_probs)

    auto_main(
        _id=12,
        _models=["mobilenetv2"],
        _seeds=seeds,
        _batch_sizes=[8],
        _max_steps_array=[250],
        _learning_rates=[0.001],
        _success_boundaries=success_boundaries,
        _dropouts=dropout_keep_probs)

    # auto_main(
    #     _id=9,
    #     _models=["mobilenetv3"],
    #     _seeds=seeds,
    #     _batch_sizes=[16],
    #     _max_steps_array=[250],
    #     _learning_rates=[0.001],
    #     _success_boundaries=success_boundaries,
    #     _dropouts=dropout_keep_probs)

    ##pre_input.main("inceptionresnetv2")
