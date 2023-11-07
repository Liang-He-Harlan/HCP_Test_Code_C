# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from datasets.dataloader import MyDataset
from torch.utils.data import DataLoader, ConcatDataset
import os
import torch
import utils.timm
from utils.timm import create_model
from utils.timm.scheduler.cosine_lr import CosineLRScheduler
import h5py
import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda:0')

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    data_dir = str(os.getcwd()+'/Data/')
    T0 = 'REST1/'
    L0 = 0
    T1 = 'WM/'
    L1 = 1
    batch_size = 16

    Ttag = 0 # 0: train; 1: test

    NT0_dataset = MyDataset(data_dir + T0, L0, 0)
    TT0_dataset = MyDataset(data_dir + T0, L0, 1)

    NT1_dataset = MyDataset(data_dir + T1, L1, 0)
    TT1_dataset = MyDataset(data_dir + T1, L1, 1)

    NT01_train_dataloader = ConcatDataset([NT0_dataset, NT1_dataset])
    NT01_test_dataloader = ConcatDataset([TT0_dataset, TT1_dataset])
    train_dataloader = DataLoader(NT01_train_dataloader, batch_size=batch_size)
    test_dataloader = DataLoader(NT01_test_dataloader, batch_size=batch_size)

    model = create_model('vit_small_patch16_224', pretrained=True, num_classes=2)
    model.patch_embed = torch.nn.Linear(136, model.pos_embed.shape[2])
    model.pos_embed = torch.nn.Parameter(torch.zeros(1, 136 + 1, model.pos_embed.shape[2]))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_schedule = CosineLRScheduler(optimizer=optimizer, t_initial=10, lr_min=1e-5, warmup_t=5)
    loss_fn = torch.nn.CrossEntropyLoss()

    epochs = 300
    loss_fn = loss_fn.to(device)
    model = model.to(device)
    end_flag = 0

    for epoch in range(epochs):
        model.train()
        train_loss, test_acc, test_loss = .0, .0, .0
        for image, label in train_dataloader:
            image = image.to(device)
            image = image.to(torch.float32)

            label = label.to(device)
            pred = model(image)
            loss = loss_fn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        lr_schedule.step(epoch)
        with torch.no_grad():
            model.eval()
            for image, label in test_dataloader:
                image = image.to(device)
                image = image.to(torch.float32)

                label = label.to(device)
                pred = model(image)
                loss = loss_fn(pred, label)
                acc = (pred.argmax(dim=1) == label).float().mean()
                test_acc += acc.item()
                test_loss += loss.item()
        print('Epoch: {:2d}  Train Loss: {:.4f}  Test Loss: {:.4f}  Test Acc: {:.4f}' \
              .format(epoch, train_loss / len(train_dataloader), test_loss / len(test_dataloader),
                      test_acc / len(test_dataloader)))
        if test_acc / len(test_dataloader) > 0.9:
            end_flag += 1
            if end_flag > 20:
                torch.save(model.state_dict(),
                           './outputs/model_para/' + 'HCP_Transformer_motor_res.pth')
                break



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
