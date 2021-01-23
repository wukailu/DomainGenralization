from utils.foundation_tools import submit_jobs, random_params


def params_for_single_train():
    params = {
        'project_name': 'appendix_seed',
        'total_batch_size': 256,  # real batch size will be gpus * batch size
        'gpus': 1,
        'num_epochs': 200,
        'weight_decay': 5e-4,
        'max_lr': 0.01,
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'SGD',
        'backbone': ['vgg16_imagenet'],
        "dataset": "ImageNet100_ordered_partial",
        "dataset_params": {
            "total": [500, 250, 100, 50, 25, 10, 5, 2, 1],
            "selected": 0,
        },
        "seed": [0, 1, 2, 3],
    }
    return random_params(params)


if __name__ == "__main__":
    submit_jobs(params_for_single_train, 'train_single_model.py', number_jobs=1000)
