import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_metric_curves(train_losses, train_acc, test_losses, test_acc):
    t = [t_items.item() for t_items in train_losses]

    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(t)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")

def plot_misclassified(model, test_dataloader, device):
    pred_list, image_list, label_list = [], [], []
    with torch.inference_mode():
        model.eval()
        for images, labels in test_dataloader:
            images = images.to(device)
            outputs = model(images)

            preds = torch.argmax(outputs, dim=1).cpu()
            wrong_indices = torch.argwhere(preds != labels).flatten()
            pred_list += preds[wrong_indices].tolist()
            image_list += [img for img in images[wrong_indices].cpu()]
            label_list += labels[wrong_indices].tolist()

            if len(label_list) >= 10:
                break
    pred_list = pred_list[:10]
    label_list = label_list[:10]
    image_list = image_list[:10]

    fig, ax = plt.subplots(5,2, figsize=(10,10))
    for i in range(5):
        for j in range(2):
            img = image_list.pop()
            img = img / 2 + 0.5     # unnormalize
            npimg = img.numpy()

            ax[i,j].imshow(np.transpose(npimg, (1, 2, 0)))
            ax[i,j].set_title(f"Label: {test_dataloader.dataset.classes[label_list.pop()]} | Predicted: {test_dataloader.dataset.classes[pred_list.pop()]}")
            ax[i,j].grid(False)
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
    