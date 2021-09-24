import torch.utils.data
import torchvision.models

from data_visualization.visualize import ImageView
from models.resnet import Resnet, Resnet2
from preprocessing.create_dataset import CreateDataset
from preprocessing.data_augmentation import DataAugmentation

if __name__ == '__main__':
    print(f'\nCopper Beaten Appearance Classification from Skull Radiographs')
    print(f'=====================================')
    print(f'Using PyTorch version {torch.__version__}')

    torch.manual_seed(0)
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'PyTorch is using: {torch_device}\n')

    '''Image transformations'''
    train_transform = DataAugmentation.train_transform()
    val_transform = DataAugmentation.val_transform()
    test_transform = DataAugmentation.test_transform()

    '''Preparing the dataset'''
    # TRAINING DATASET
    train_dirs = {
        'normal': 'datasets/train/normal',
        'cba': 'datasets/train/cba'
    }

    train_dataset = CreateDataset(train_dirs, train_transform)

    # VALIDATION DATASET
    val_dirs = {
        'normal': 'datasets/validation/normal',
        'cba': 'datasets/validation/cba'
    }

    val_dataset = CreateDataset(val_dirs, val_transform)

    # TESTING DATASET
    test_dirs = {
        'normal': 'datasets/test/normal',
        'cba': 'datasets/test/cba'
    }

    test_dataset = CreateDataset(test_dirs, test_transform)

    # DataLoaders
    batch_size = 8
    dl_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dl_val = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    dl_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print(f'\nNumber of train batches {len(dl_train)}')
    print(f'Number of validation batches {len(dl_val)}')
    print(f'Number of test batches {len(dl_test)}\n')

    '''Data visualization'''
    class_names = train_dataset.class_names

    images, labels = next(iter(dl_train))
    ImageView.display_images(images, labels, labels, class_names, n_samples=batch_size)

    images, labels = next(iter(dl_val))
    ImageView.display_images(images, labels, labels, class_names, n_samples=batch_size)

    images, labels = next(iter(dl_test))
    ImageView.display_images(images, labels, labels, class_names, n_samples=batch_size)

    '''Training the model'''
    train = True
    experiment = 'experiment_1'
    resnet_50 = Resnet(model=torchvision.models.resnet50(pretrained=True), name_experiment=experiment
                       , num_classes=2)
    if train == True:
        resnet_50.train_model(epochs=5, dl_train=dl_train, dl_val=dl_val, batch_size=batch_size)
    else:
        resnet_50.load_model('./experiments/' + experiment + '/' + experiment + '.pt')
        resnet_50.model.cuda()

    resnet_50.model.eval()

    '''Predictions'''
    images_pred, labels_pred = next(iter(dl_test))
    images_pred = images_pred.to(torch_device)
    labels_pred = labels_pred.to(torch_device)
    outputs = resnet_50.model(images_pred)
    _, predictions = torch.max(outputs, 1)
    ImageView.display_images(images_pred, labels_pred, predictions, class_names, n_samples=batch_size)

    # resnet_50 = Resnet.resnet_model(model = torchvision.models.resnet50(pretrained=True),
    #                             epochs=10,
    #                             dl_train=dl_train,
    #                             dl_val=dl_val,
    #                             num_classes=2,
    #                             save_model_path='model_50.pt')

    # resnet_18 = Resnet.resnet_18(epochs=2,
    #                              dl_train=dl_train,
    #                              dl_val=dl_val,
    #                              val_len=len(val_dataset),
    #                              torch_device=torch_device)
    # resnet_18.eval()

    #
    # resnet_50 = Resnet2.resnet_50(1, [dl_train, dl_val], 'model_50.pt')
