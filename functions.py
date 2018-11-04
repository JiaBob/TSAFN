import time
from torchvision import utils
from torch.utils.data import DataLoader, sampler
from tensorboardX import SummaryWriter

from models import *
from dataloader import *


log_folder = './runs'  # for comparing several training result
log_path = 'log1'
log = '{}/{}'.format(log_folder, log_path)

# check if 'log_folder' exist, if not create it, if yes but 'log' already exists
# remove 'log', final create 'log' in 'log_folder'
if not os.path.exists(log_folder):
    os.makedirs(log)  # recursive folder making

writer = SummaryWriter(log)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class trainer:
    def __init__(self, model_name, lr, optimizer, pretrained=False, separate=False, epochs=100, numloader=0,
                 batchsize=(2, 1), sample_amount=None, random_crop_size=(64, 64)):
        self.model_name = model_name
        self.model_dict = {'spn': SPN, 'tpn': TPN, 'tsafn': TSAFN, 'all': Combination}
        self.iter_methods = {'spn': self.spn_one_iter, 'tpn': self.tpn_one_iter,
                           'tsafn': self.tsafn_one_iter, 'all': self.combination_one_iter}

        self.pretrained = pretrained
        if model_name == 'all' and pretrained:
            self.separate = separate  # only make sense when model is all and pretrained is true.
            self.model = Combination(pretrained, separate=separate)
        else:
            self.separate = None
            self.model = self.model_dict[model_name](pretrained)
        self.model = nn.DataParallel(self.model).to(device)
        self.model_iter = self.iter_methods[model_name]

        if pretrained:
            self.index = self.model.module.index
        else:
            self.index = None

        self.data = None
        self.sampler = None
        self.sample_amount = sample_amount
        self.random_crop_size = random_crop_size
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.epochs = epochs
        self.batchsize = batchsize
        self.numloader = numloader

        self.lr = lr
        self.optimizer = optimizer(self.model.parameters(), lr)

        self.is_train = True
        self.minimum_val_loss = float('inf')
        self.output_list = torch.empty([0]).to(device)  # for temporarily store output image batch
        self.current_epoch = 0
        self.test_loss = None

    def train(self, path, verbose=True):
        self.data = SplitData(path, self.index)
        self.index = self.data.index_dict  # new index from non-pretrained version

        train_set = self.data('train', random_crop_size=self.random_crop_size)
        if self.sample_amount:
            self.sampler = sampler.WeightedRandomSampler(torch.ones(len(train_set)), self.sample_amount)
        self.train_loader = DataLoader(train_set, batch_size=self.batchsize[0], sampler=self.sampler,
                                       num_workers=self.numloader)
        self.val_loader = DataLoader(self.data('val'), batch_size=self.batchsize[1], num_workers=self.numloader)
        self.test_loader = DataLoader(self.data('test'), batch_size=1, num_workers=self.numloader)

        start_time = time.time()
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            epoch_time = time.time()

            train_loss = self.model_iter(self.train_loader)
            self.output_list = torch.empty([0])  # reset
            val_loss = self.model_iter(self.val_loader)

            epoch_time = time.time() - epoch_time

            # codes below are for printing
            if (epoch + 1) % 1 == 0 and verbose:
                print("Epoch {} finished, current validation loss is {:1.5f}, current train loss is {:1.5f}".format(
                    epoch + 1,
                    val_loss, train_loss, timeformat(epoch_time)))
                print('It takes {} from beginning'.format(timeformat(time.time() - start_time)))

            # codes below are for sustainability
            if val_loss < self.minimum_val_loss:
                self.minimum_val_loss = val_loss
                save_model(self.model, epoch + 1, train_loss, val_loss, self.index,
                           self.optimizer.param_groups[0]['lr'], time.strftime('%dd-%Hh-%Mm', time.localtime(start_time)))

            # codes below are for visualization
            loss_dict = {'{}_loss'.format(self.model_name): {'train': train_loss, 'validation': val_loss}}
            visualize(loss_dict, epoch, mode='scalar_dict')

            img_dict = {'{} results'.format(self.model_name): self.output_list}
            visualize(img_dict, epoch, mode='image')

        self.test()

    def tpn_one_iter(self, data_loader):
        loss_sum = 0
        for x in data_loader:
            if self.is_train:
                self.optimizer.zero_grad()
            with torch.set_grad_enabled(self.is_train):
                output = self.model(x['input'].to(device))
                target = x['TPN'].to(device)

                loss = F.mse_loss(output, target).to(device)
                if self.is_train:
                    loss.backward()
                    self.optimizer.step()
                else:
                    # only display the first one from each minibatch
                    show = 1
                    self.output_list = torch.cat((self.output_list, target[:show], output[:show]), 0)

            loss_sum += loss.item()
        self.is_train = not self.is_train  # reverse mode
        epoch_loss = loss_sum / len(data_loader)
        return epoch_loss

    def spn_one_iter(self, data_loader):
        loss_sum = 0
        for x in data_loader:
            if self.is_train:
                self.optimizer.zero_grad()
            with torch.set_grad_enabled(self.is_train):
                output = self.model(x['input'].to(device))
                target = x['SPN'].to(device)

                loss = spn_loss(*output, target).to(device)
                if self.is_train:
                    loss.backward()
                    self.optimizer.step()
                else:
                    show = 1
                    self.output_list = torch.cat((self.output_list, target[:show], output[-1][:show]), 0)

            loss_sum += loss.item()
        self.is_train = not self.is_train  # reverse mode
        epoch_loss = loss_sum / len(data_loader)
        return epoch_loss

    def tsafn_one_iter(self, data_loader):
        loss_sum = 0
        for x in data_loader:
            if self.is_train:
                self.optimizer.zero_grad()
            with torch.set_grad_enabled(self.is_train):
                inpu = torch.cat((x['input'].to(device), x['TPN'].to(device), x['SPN'].to(device)), 1)
                output = self.model(inpu)
                target = x['TSAFN'].to(device)

                loss = F.mse_loss(output, target).to(device)
                if self.is_train:
                    loss.backward()
                    self.optimizer.step()
                else:
                    show = 1
                    self.output_list = torch.cat((self.output_list, target[:show], output[:show]), 0)

            loss_sum += loss.item()
        self.is_train = not self.is_train  # reverse mode
        epoch_loss = loss_sum / len(data_loader)
        return epoch_loss

    def combination_one_iter(self, data_loader):
        loss_sum = 0
        loss_spn_sum = 0
        loss_tpn_sum = 0
        loss_tsafn_sum = 0
        for x in data_loader:
            if self.is_train:
                self.optimizer.zero_grad()
            with torch.set_grad_enabled(self.is_train):
                inpu = x['input'].to(device)
                target_tpn = x['TPN'].to(device)
                target_spn = x['SPN'].to(device)
                target_tsafn = x['TSAFN'].to(device)

                output_spn, output_tpn, output_tsafn = self.model(inpu)

                loss_spn = spn_loss(*output_spn, target_spn)
                loss_tpn = F.mse_loss(output_tpn, target_tpn)
                loss_tsafn = F.mse_loss(output_tsafn, target_tsafn)

                loss = 0.6 * loss_tsafn + 0.2 * (loss_spn + loss_tpn)
                loss = loss.to(device)

                if self.is_train:
                    loss.backward()
                    self.optimizer.step()
                else:  # only visualize validation data
                    show = 1  # show the first one of each mini-batch
                    output_spn = output_spn[-1].expand(-1, 3, -1, -1)
                    output_tpn = output_tpn.expand(-1, 3, -1, -1)
                    self.output_list = torch.cat((self.output_list, target_tsafn[:show], output_tsafn[:show],
                                                  output_spn[:show], output_tpn[:show]), 0)

            loss_spn_sum += loss_spn.item()
            loss_tpn_sum += loss_tpn.item()
            loss_tsafn_sum += loss_tsafn.item()
            loss_sum += loss.item()

        self.is_train = not self.is_train  # reverse mode
        epoch_loss = loss_sum / len(data_loader)
        return epoch_loss

    def test(self):
        self.is_train = False
        self.test_loss = self.model_iter(self.test_loader)

    def test_unknown(self, path):
        self.data = SplitData(path, unknown=True)
        self.unknown_loader = DataLoader(self.data('unknown'), batch_size=1, num_workers=self.numloader)
        amount = len(self.unknown_loader)
        for i, inpu in enumerate(self.unknown_loader):
            output = self.model(inpu)

            if  self.model_name == 'tsafn':
                utils.save_image(output, './unknown/{}.jpg'.format(i))
            elif self.model_name == 'all' or self.model_name == 'spn':
                utils.save_image(output[-1], './unknown/{}.jpg'.format(i))  # only need the final output
            else:
                utils.save_image(output, './unknown/{}.jpg'.format(i))
            print('Finished {}/{}'.format(i + 1, amount))


def weighted_cross_entropy(pred, target):
    total = target.numel()  # it is int
    negative = (target == 0).sum().item()  # it is tensor before call .item()

    mask = torch.empty_like(target, dtype=torch.float)
    mask[target == 0] = 1 - negative / total
    mask[target != 0] = negative / total

    # If rigorously follow what the HED paper did, argument reduction='sum' should be added.
    # However, to make all losses under small scale, we remove it. Same for BCEWithLogitsLoss below.
    return F.binary_cross_entropy(pred, target, mask)


def spn_loss(so1, so2, so3, so4, so5, fusion, target):
    # non-weighted BCE
    criterion = nn.BCELoss()
    loss = criterion(fusion, target)

    # there are weights called alpha mentioned in paper which is used to combine
    # the side output loss summation. However, no further description about alpha
    # thus here just simply sum them together without weights.

    loss += weighted_cross_entropy(so1, target)
    loss += weighted_cross_entropy(so2, target)
    loss += weighted_cross_entropy(so3, target)
    loss += weighted_cross_entropy(so4, target)
    loss += weighted_cross_entropy(so5, target)

    return loss


def peek_signal_to_noise_ratio(mse):
    return 10 * ((255 ** 2) / mse).log10()


def structure_similarity(x, y):
    meanx = x.mean()
    meany = y.mean()
    sigmax = (x - meanx).pow(2).mean().sqrt()
    sigmay = (y - meany).pow(2).mean().sqrt()
    cov = ((x - meanx) * (y - meany)).mean().sqrt()

    luminance = (2 * meanx * meany + 1) / (meanx + meany + 1)
    contrast = (2 * sigmax * sigmay + 1) / (sigmax + sigmay + 1)
    structure = (cov + 1) / (sigmax + sigmay + 1)

    return luminance * contrast * structure


def timeformat(s):
    s = int(s)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    t = '{:>02d}:{:>02d}:{:>02d}'.format(h, m, s) if h else '{:>02d}:{:>02d}'.format(m, s) if m else '{:2d}s'.format(s)
    return t


def save_model(model, epoch, train_loss, val_loss, index, lr, start_time):
    model_name = model.module.__class__.__name__
    pretrained = model.module.pretrained
    encap = {'lr': lr,  # Note this lr for the first optimizer model
             'state_dict': model.module.state_dict(),
             'train_loss': train_loss,
             'val_loss': val_loss,
             'epoch': epoch,
             'pretrained': pretrained,
             'index': index}

    # special attribute for SPN
    if model_name == 'SPN' or model_name == 'Combination':
        encap['vgg'] = model.module.vgg

    torch.save(encap, './temp_models/{}_{}_{}.pth'.format(model_name, start_time, pretrained))


def visualize(var_dict, epoch, mode='scalar'):
    '''
    Description: visualize variables (single scalar, bunch of scalars or images) on tensorboard

    :param var_dict: {variable name: value}. If mode=scalar, the value should be a scalar. If mode=scalar_dict,
                    the value should be a dict whose key is name of each variable. If mode=image, the value should be
                    a list of torch.Tensor, whose shape is like (N, C, H, W).
    :param mode: can be 'scalar', 'scalar_dict', 'image'
    :param epoch:
    :return: None
    '''

    for label, var in var_dict.items():
        # bunch of variables
        if mode == 'scalar_dict':
            writer.add_scalars(label, var, epoch)

        # single variable
        elif mode == 'scalar':
            writer.add_scalar(label, var, epoch)

        elif mode == 'image':  # combine into grid
            writer.add_image(label, utils.make_grid(var, nrow=2, padding=20))

        else:
            raise Exception('invalid argument for \'mode\'. Please use either scalar, scalar_dict or image')


