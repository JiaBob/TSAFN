import time
from torchvision import utils
from tensorboardX import SummaryWriter

from models import *

__all__ = ['tpn_one_iter', 'spn_one_iter', 'tsafn_one_iter', 'combination_one_iter', 'train', 'test', 'test_unknown']



log_folder = './runs'  # for comparing several training result
log_path = 'log1'
log = '{}/{}'.format(log_folder, log_path)

# check if 'log_folder' exist, if not create it, if yes but 'log' already exists
# remove 'log', final create 'log' in 'log_folder'
if not os.path.exists(log_folder):
    os.makedirs(log)  # recursive folder making

writer = SummaryWriter(log)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tpn_one_iter(model, data_loader, epoch, optimizer=None):

    if optimizer:
        is_train = True
        output_list = None
    else:
        is_train = False
        output_list = torch.empty([0])

    loss_sum = 0
    criterion = nn.MSELoss()
    for x in data_loader:
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            output = model(x['input'].to(device))
            target = x['TPN'].to(device)

            loss = criterion(output, target).to(device)
            if is_train:
                loss.backward()
                optimizer.step()
            else:
                show = 1
                output_list = torch.cat((output_list, target[:show], output[:show]), 0)

        loss_sum += loss.item()

    epoch_loss = loss_sum / len(data_loader)
    return epoch_loss, output_list


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


def spn_one_iter(model, data_loader, epoch, optimizer=None):
    if optimizer:
        is_train = True
        output_list = None
    else:
        is_train = False
        output_list = torch.empty([0])

    loss_sum = 0
    for x in data_loader:
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            output = model(x['input'].to(device))
            target = x['SPN'].to(device)

            loss = spn_loss(*output, target).to(device)

            if is_train:
                loss.backward()
                optimizer.step()
            else:
                show = 1
                output_list = torch.cat((output_list, target[:show], output[-1][:show]), 0)

        loss_sum += loss.item()

    epoch_loss = loss_sum / len(data_loader)
    return epoch_loss, output_list


def tsafn_one_iter(model, data_loader, epoch, optimizer=None):
    if optimizer:
        is_train = True
        output_list = None
    else:
        is_train = False
        output_list = torch.empty([0])

    criterion = nn.MSELoss()
    loss_sum = 0
    for x in data_loader:
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            inpu = torch.cat((x['input'].to(device), x['TPN'].to(device), x['SPN'].to(device)), 1)
            output = model(inpu)
            target = x['TSAFN'].to(device)

            loss = criterion(output, target).to(device)
            if is_train:
                loss.backward()
                optimizer.step()
            else:
                show = 1
                output_list = torch.cat((output_list, target[:show], output[:show]), 0)

        loss_sum += loss.item()
    epoch_loss = loss_sum / len(data_loader)
    return epoch_loss, output_list


def combination_one_iter(model, data_loader, epoch, optimizer=None):
    if optimizer:
        is_train = True
        output_list = None
    else:
        is_train = False
        output_list = torch.empty([0])

    loss_sum = 0
    loss_spn_sum = 0
    loss_tpn_sum = 0
    loss_tsafn_sum = 0

    for x in data_loader:
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            inpu = x['input'].to(device)
            target_tpn = x['TPN'].to(device)
            target_spn = x['SPN'].to(device)
            target_tsafn = x['TSAFN'].to(device)

            output_spn, output_tpn, output_tsafn = model(inpu)

            loss_spn = spn_loss(*output_spn, target_spn)
            loss_tpn = F.mse_loss(output_tpn, target_tpn)
            loss_tsafn = F.mse_loss(output_tsafn, target_tsafn)

            loss = 0.6 * loss_tsafn + 0.2 * (loss_spn + loss_tpn)
            loss = loss.to(device)

            if is_train:
                loss.backward()
                optimizer.step()
            else:  # only visualize validation data
                show = 1  # show the first one of each mini-batch
                output_spn = output_spn[-1].expand(-1, 3, -1, -1)
                output_tpn = output_tpn.expand(-1, 3, -1, -1)
                output_list = torch.cat((output_list, target_tsafn[:show], output_tsafn[:show],
                                         output_spn[:show], output_tpn[:show]), 0)

        loss_spn_sum += loss_spn.item()
        loss_tpn_sum += loss_tpn.item()
        loss_tsafn_sum += loss_tsafn.item()
        loss_sum += loss.item()

    epoch_loss = loss_sum / len(data_loader)
    return epoch_loss, output_list


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


def train(model, model_iter, train_loader, val_loader, optimizer=None, epochs=500, verbose=True):
    start_time = time.time()

    train_losses, val_losses = [], []
    minimum_val_loss = float('inf')
    for epoch in range(epochs):
        epoch_time = time.time()

        train_loss, _ = model_iter(model, train_loader, epoch, optimizer)
        val_loss, output = model_iter(model, val_loader, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        epoch_time = time.time() - epoch_time

        if (epoch + 1) % 1 == 0 and verbose:
            print("Epoch {} finished, current validation loss is {:1.5f}, current train loss is {:1.5f}".format(epoch + 1,
                                                                            val_loss, train_loss, timeformat(epoch_time)))
            print('It takes {} from beginning'.format(timeformat(time.time() - start_time)))

        if val_loss < minimum_val_loss:
            minimum_val_loss = val_loss
            save_model(model, epoch + 1, train_loss, val_loss, optimizer.param_groups[0]['lr'],
                       time.strftime('%dd-%Hh-%Mm', time.localtime(start_time)))

        # codes below are for visualization
        loss_dict = {'loss': {'train': train_loss, 'validation': val_loss}}
        visualize(loss_dict, epoch, mode='scalar_dict')

        img_dict = {'{} results'.format(model.module.__class__.__name__): output}  # only gather the first 10 samples
        visualize(img_dict, epoch, mode='image')
    return train_losses, val_losses


def test(model, model_iter, data_loader):
    loss, output = model_iter(model, data_loader, epoch=1)
    return loss, output


def test_unknown(model, data_loader):
    for i, inpu in enumerate(data_loader):
        output = model(inpu)

        if output.shape[0] != 3:
            output = output.expand(3, -1, -1)
        utils.save_image(output, './unknown/{}.jpg'.format(i))


def save_model(model, epoch, train_loss, val_loss, lr,  start_time):
    model_name = model.module.__class__.__name__
    pretrained = model.module.pretrained
    encap = {'lr': lr,  # Note this lr for the first optimizer model
             'state_dict': model.state_dict(),
             'train_loss': train_loss,
             'val_loss': val_loss,
             'epoch': epoch,
             'pretrained': pretrained}

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


