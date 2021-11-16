from torchvision import transforms

from segmentation.data_loader.segmentation_dataset import SegmentationDataset
from segmentation.data_loader.transform import Rescale, ToTensor
from segmentation.trainer import Trainer
from segmentation.predict import *
from segmentation.models import all_models
from util.logger import Logger
import argparse

train_images = r'../datasets/images/train'
test_images = r'../datasets/images/test'
train_labled = r'../datasets/labeled/train'
test_labeled = r'../datasets/labeled/test'

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help="Choose model", default="unet_resnet152")
parser.add_argument("--batch_size", help="training batch size", default=4)
parser.add_argument("--num_class", help="how many class you segment", default=4)
parser.add_argument("--epoch", help="how many epoch you want for training", default=100)
parser.add_argument("--image_size", help="minimum of image width", default=200)
args = parser.parse_args()

if __name__ == '__main__':
    model_name = args.model_name
    device = 'cuda'
    batch_size = args.batch_size
    n_classes = args.num_class
    num_epochs = args.epoch
    image_axis_minimum_size = args.image_size
    pretrained = True
    fixed_feature = False

    logger = Logger(model_name=model_name, data_name='dataset')

    ### Loader
    compose = transforms.Compose([
        Rescale(image_axis_minimum_size),
        ToTensor()
         ])

    train_datasets = SegmentationDataset(train_images, train_labled, n_classes, compose)
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, drop_last=True)

    test_datasets = SegmentationDataset(test_images, test_labeled, n_classes, compose)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True, drop_last=True)

    ### Model
    model = all_models.model_from_name[model_name](n_classes, batch_size,
                                                   pretrained=pretrained,
                                                   fixed_feature=fixed_feature)
    model.to(device)

    ###Load model
    ###please check the foloder: (.segmentation/test/runs/models)
    #logger.load_model(model, 'epoch_15')

    ### Optimizers
    if pretrained and fixed_feature: #fine tunning
        params_to_update = model.parameters()
        print("Params to learn:")
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
        optimizer = torch.optim.Adadelta(params_to_update)
    else:
        optimizer = torch.optim.Adadelta(model.parameters())

    ### Train
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    check_point_stride = 90
    trainer = Trainer(model, optimizer, logger, num_epochs, train_loader, test_loader, check_point_epoch_stride=check_point_stride)
    trainer.train()


    #### Writing the predict result.
    predict(model, r'../datasets/predict/input.png',
             r'../datasets/predict/output.png')