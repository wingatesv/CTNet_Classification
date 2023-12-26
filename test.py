from setting import parse_opts 
from datasets.brains18 import BrainS18Dataset
from model import generate_model
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy import ndimage
import nibabel as nib
import sys
import os
from utils.file_process import load_lines
import numpy as np
from sklearn.metrics import f1_score

# def seg_eval(pred, label, clss):
#     """
#     calculate the accuracy between prediction and ground truth
#     input:
#         pred: predicted mask
#         label: ground truth
#         clss: eg. [0, 1] for binary class
#     """
#     Ncls = len(clss)
#     accuracies = np.zeros(Ncls)
#     [depth, height, width] = label.shape  # Assuming label has the correct shape

#     for idx, cls in enumerate(clss):
#         # binary map
#         pred_cls = np.zeros([depth, height, width])
#         pred_cls[np.where(pred == cls)] = 1
#         label_cls = np.zeros([depth, height, width])
#         label_cls[np.where(label == cls)] = 1

#         # calculate accuracy
#         correct_pixels = np.sum(np.logical_and(pred_cls == 1, label_cls == 1))
#         total_pixels = np.sum(label_cls == 1)
        
#         try:
#             accuracy = correct_pixels / total_pixels
#         except ZeroDivisionError:
#             print("Total pixels is zero when calculating accuracy.")
#             accuracy = -1

#         accuracies[idx] = accuracy

#     return accuracies

def test(data_loader, model, labels, sets):
    predictions = []
    model.eval()  # for testing
    for batch_id, batch_data in enumerate(data_loader):
        # forward
        volume = batch_data
        if not sets.no_cuda:
            volume = volume.cuda()
        with torch.no_grad():
            probs = model(volume)
            probs = F.softmax(probs, dim=1)

            # Extract predicted classes
            prediction = torch.argmax(probs, dim=1)

            print('prediction: ', prediction)

        predictions.append(prediction.cpu().numpy())
    
    # Calculate accuracy
    correct_predictions = sum(p == t for p, t in zip(predictions, labels))
    accuracy = correct_predictions / len(labels)

    # Calculate F1 score
    flat_predictions = [item for sublist in predictions for item in sublist]
    f1 = f1_score(labels, flat_predictions, average='binary')  # Assuming binary classification

    return accuracy, f1


if __name__ == '__main__':
    # settting
    sets = parse_opts()
    sets.target_type = "normal"
    sets.phase = 'test'

    # getting model
    checkpoint = torch.load(sets.resume_path)
    net, _ = generate_model(sets)
    net.load_state_dict(checkpoint['state_dict'])

    # data tensor
    testing_data =BrainS18Dataset(sets.data_root, sets.img_list, sets)
    data_loader = DataLoader(testing_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

    # testing
    # img_names = [info.split(" ")[0] for info in load_lines(sets.img_list)]
    # masks = test(data_loader, net, img_names, sets)
    
    # evaluation: calculate dice 
    labels = [int(info.split(" ")[2]) for info in load_lines(sets.img_list)]

    accuracy, f1 = test(data_loader, net, labels, sets)

    print('Accuracy: {:.4%}'.format(accuracy.item()))
    print('F1-Score: {:.4%}'.format(f1))
    # Nimg = len(label_names)
    # dices = np.zeros([Nimg, sets.n_seg_classes])
    # for idx in range(Nimg):
    #     label = nib.load(os.path.join(sets.data_root, label_names[idx]))
    #     label = label.get_fdata()
    #     dices[idx, :] = seg_eval(masks[idx], label, range(sets.n_seg_classes))
    
    # # print result
    # for idx in range(1, sets.n_seg_classes):
    #     mean_dice_per_task = np.mean(dices[:, idx])
    #     print('mean dice for class-{} is {}'.format(idx, mean_dice_per_task))   
