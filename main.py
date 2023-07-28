from torchsummary import summary
from preprocess import *
from resnet import *
from matplotlib import pyplot as plt


num_classes = 33
training_data, validation_data = preprocess()

resnet = ResNet(3, num_classes)
model = to_device(resnet, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
print(model)
print(summary(model.cuda(), (3, 128, 128)))



history = model.fit(epochs=3,
                    max_lr=0.01,
                    train_loader=training_data,
                    val_loader=validation_data,
                    grad_clip=0.1,
                    weight_decay=1e-5,
                    opt_func=torch.optim.Adam)




history += model.fit(epochs=3,
                    max_lr=0.001,
                    train_loader=training_data,
                    val_loader=validation_data,
                    grad_clip=0.1,
                    weight_decay=1e-5,
                    opt_func=torch.optim.Adam)

val_accuracy = []
val_loss = []
train_accuracy = []
train_loss = []
for i in range(len(history)):
    val_accuracy.append(history[i]['val_accuracy'].cpu() * 100)
    val_loss.append(history[i]['val_loss'].cpu())
    train_accuracy.append(history[i]['train_acc'] * 100)
    train_loss.append(history[i]['train_loss'])


fig, ax = plt.subplots(figsize=(5, 3), layout='constrained')
fig2, ax2 = plt.subplots(figsize=(5, 3), layout='constrained')
ax.plot([i for i in range(6)], val_accuracy, label='val acc')
ax2.plot([i for i in range(6)], val_loss, label='val loss')
ax.plot([i for i in range(6)], train_accuracy, label='train acc')
ax2.plot([i for i in range(6)], train_loss, label='train loss')
ax.set_xlabel('epochs')
ax2.set_xlabel('epochs')
ax.set_title('accuracy (percentage)')
ax2.set_title('loss')
ax.legend()
ax2.legend()
plt.show()

torch.save(model.state_dict(), 'm.pth')