import torch

from torch import nn
from torch.autograd import Variable
from torch.optim import Adam, SGD
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from torchvision.utils import make_grid
from tqdm import tqdm
import torchnet as tnt
from torchsummary import summary
import os
from Data_reader import*
from Models import*
import sys

sys.setrecursionlimit(15000)
os.environ['CUDA_VISIBLE_DEVICES']= '0'
path = os.getcwd()
data_path = os.path.dirname(os.getcwd()) + "/data/weed/"

NUM_CLASS = 21
INPUT_CHANNEL = 3
BATCH_SIZE = 64
EPOCHS = 500




if __name__ == "__main__":

    dataset_train = Weedread(data_path + "train96_21.h5")
    dataset_test = Weedread(data_path + "val96_21.h5")
    # dataset_train = Mnistread(True, data_path)
    # dataset_test = Mnistread(False, data_path)

    def get_iterator(mode):
        if mode is True:
            dataset = dataset_train
        elif mode is False:
            dataset = dataset_test
        loader = DataLoader(dataset, batch_size = BATCH_SIZE, num_workers=4, shuffle=True)
        return loader
 
    _model = SpatialTransformerPretrained(input_channel=INPUT_CHANNEL, num_class=NUM_CLASS)

    _model.cuda()
    summary(_model, input_size=(3, 96, 96))
    print("# parameters:", sum(param.numel() for param in _model.parameters()))
    ##------------------init------------------------##
    log = []
    optimizer = SGD(_model.parameters(), lr=0.01)
    engine = Engine()#training loop
  
    ##------------------log visualization------------------------##
    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
    train_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Train Accuracy'})
    test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'})
    test_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Test Accuracy'})
    ground_truth_logger = VisdomLogger('image', opts={'title': 'Ground Truth'})
    transformed_logger = VisdomLogger('image', opts={'title': 'Transformed'})
     
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def processor(epoch):
        for istraining in [True, False]:
            if istraining:
                _model.train()
            else:
                _model.eval()

            running_loss = 0
            running_corrects = 0
            for data, target in tqdm(get_iterator(istraining)):
                data, target = data.to(device), target.to(device)
                data = data.float() / 255.0

                optimizer.zero_grad()
                with torch.set_grad_enabled(istraining):
                    output = _model(data)
                    loss = F.nll_loss(output, target)
                    pred = output.max(1, keepdim=True)[1]
                
                    if istraining:
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() *  BATCH_SIZE
                running_corrects += pred.eq(target.view_as(pred)).sum().item()
                
            if istraining:
                n = dataset_train.__len__()
                print('train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'
                .format(running_loss / n, running_corrects, n, 100. * running_corrects / n))
                torch.save(_model.state_dict(), 'epochs/epoch_0.pt')
                train_loss_logger.log(epoch, running_loss / n)
                train_accuracy_logger.log(epoch, 100. * running_corrects / n)
            else:
                n = dataset_test.__len__()
                print('test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'
                .format( running_loss / n, running_corrects, n, 100. * running_corrects / n))
                test_loss_logger.log(epoch,  running_loss / n)
                test_accuracy_logger.log(epoch, 100. * running_corrects / n)

            if not istraining:
                #visualize
                test_loader = get_iterator(False)
                test_sample = next(iter(test_loader))[0].to(device)
                ground_truth = test_sample.float() / 255.0
                transformed, grid = _model.stn(ground_truth)
                regular_grid = torch.cuda.FloatTensor(BATCH_SIZE, 3, 96, 96).fill_(1.)
                sample_grid = F.grid_sample(regular_grid, grid, padding_mode="zeros")

                regular_grid = regular_grid.cpu().view_as(regular_grid).data
                ground_truth = ground_truth.cpu().view_as(ground_truth).data
                sample_grid = sample_grid.cpu().view_as(sample_grid).data
                ground_truth_logger.log(make_grid(255*(ground_truth), nrow=int(BATCH_SIZE ** 0.5), pad_value=255))

                transformed = transformed.cpu().view_as(transformed).data
                transformed_logger.log(make_grid(255*(transformed), nrow=int(BATCH_SIZE ** 0.5), pad_value=255))
                imgs = make_grid(255*(transformed), nrow=int(BATCH_SIZE ** 0.5), pad_value=1)
                imgs = np.array(imgs, dtype=np.uint8)
                imgs = np.transpose(imgs, (1, 2, 0))
                cv2.imwrite(path + "/imgs/" + str(epoch) + ".jpg", imgs)

    for epoch in range(1,  EPOCHS + 1):
        print('Epoch: %d / %d'%(epoch, EPOCHS))
        processor(epoch)

