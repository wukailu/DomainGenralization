import sys
sys.path.append('/job/job_source/')

from model.cifar_lightning_model import CIFAR_Module
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.foundation_tools import parse_params, get_trainer_params

###
# python -m torch.distributed.launch --nproc_per_node=8 ../main.py
# nvidia-smi         发现内存泄露问题，即没有进程时，内存被占用
# kill $(ps aux | grep "main.py" | grep -v grep | awk '{print $2}')
###
USE_FOUNDATIONS = True


from foundations import load_parameters, log_params
print("using atlas framework")
params = load_parameters()
seed_everything(params['seed'])
log_params(params)

params = parse_params(params)
print(params)


model = CIFAR_Module(params).cuda()
lr_logger = LearningRateLogger()
logger = TensorBoardLogger("../logs", name=params["backbone"])
if USE_FOUNDATIONS:
    from foundations import set_tensorboard_logdir
    set_tensorboard_logdir(f'../logs/{params["backbone"]}')

checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor='acc', prefix=str(params["seed"]))
t_params = get_trainer_params(params)
trainer = Trainer(callbacks=[lr_logger], logger=logger, checkpoint_callback=checkpoint_callback, **t_params)
trainer.fit(model)

if USE_FOUNDATIONS and checkpoint_callback.best_model_path != "":
    from foundations import log_metric, save_artifact
    save_artifact(checkpoint_callback.best_model_path, key='best_model_checkpoint')
    log_metric("val_acc", float(checkpoint_callback.best_model_score))

print("Training finished")