import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# resnet architecture
class ResNet(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()

        self.conv1 = self.ConvBlock(in_channels, 64)
        self.conv2 = self.ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(self.ConvBlock(128, 128), self.ConvBlock(128, 128))

        self.conv3 = self.ConvBlock(128, 256, pool=True)
        self.conv4 = self.ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(self.ConvBlock(512, 512), self.ConvBlock(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(2),
                                        nn.Flatten(),
                                        nn.Linear(512, num_diseases))

    @staticmethod
    def ConvBlock(in_channels: int, out_channels: int, pool: bool = False) -> nn.Sequential:
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(4))
        return nn.Sequential(*layers)

    def forward(self, batch):
        out = self.conv1(batch)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate validateions
        loss = F.cross_entropy(out, labels)  # Calculate los
        acc = accuracy(out, labels)
        return loss, acc

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate validateion
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {"val_loss": loss.detach(), "val_accuracy": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine loss
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy}  # Combine accuracies

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['train_acc'], result['val_loss'], result['val_accuracy']))

    @torch.no_grad()
    def validate(self, validation_data):
        self.eval()
        outputs = [self.validation_step(batch) for batch in tqdm(validation_data)]
        return self.validation_epoch_end(outputs)

    def fit(self, epochs, max_lr, train_loader, val_loader, weight_decay,
            grad_clip, opt_func):
        torch.cuda.empty_cache()
        history = []

        optimizer = opt_func(self.parameters(), max_lr, weight_decay=weight_decay)
        # scheduler for one cycle learning rate
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))

        for epoch in range(epochs):
            # Training
            self.train()
            train_losses = []
            train_acc = []
            for batch in tqdm(train_loader):
                loss, acc = self.training_step(batch)
                train_losses.append(loss)
                train_acc.append(acc)
                loss.backward()

                # gradient clipping
                if grad_clip:
                    nn.utils.clip_grad_value_(self.parameters(), grad_clip)

                optimizer.step()
                optimizer.zero_grad()

                # recording and updating learning rates
                sched.step()

            # validation
            result = self.validate(val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            result['train_acc'] = torch.stack(train_acc).mean().item()
            self.epoch_end(epoch, result)
            history.append(result)

        return history
