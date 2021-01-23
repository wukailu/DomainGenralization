from model.ensemble_models import Proposed_Ensemble
from pytorch_lightning import Trainer

ckpt_path = f"logs/proposed/version_{8}/checkpoints/epoch={904}.ckpt"

# Test model
trainer = Trainer(gpus=1, deterministic=True)
trainer.model = Proposed_Ensemble.load_from_checkpoint(checkpoint_path=ckpt_path)
trainer.test(model=trainer.model)
