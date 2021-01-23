from model.imagenet_models import model_dict

if __name__ == '__main__':
    import torch

    x_test = torch.randn(2, 3, 224, 224)
    net = model_dict["densenet121"](num_classes=100)
    feats, logit = net(x_test, with_feature=True)

    for f in feats:
        print(f.shape, f.min().item())
    print(logit.shape)
    print(type(net))
    print(dir(net))
