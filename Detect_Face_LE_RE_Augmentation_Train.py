import dlib
import cv2
import imutils
import torchvision
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from imutils import face_utils
from torch.utils.data import DataLoader, Dataset
import torchvision
import argparse
import os
import shutil
import time, math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import torch.utils.model_zoo as model_zoo
from torch.autograd.variable import Variable
from skimage import io, transform
import PIL
import glob


# define GPU if exist, if not, use CPU
selected_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on", selected_device)

print("Exp start,,, ")
##### load data Train, CrossVal, Test
##### Import data
import os, os.path



file = '/home/luthffi/PycharmProjects/PhDLuthffiDeepLearningDell/P4_Train_Test_eyeTracker/200Train50Val50Test/eye_tracker_train_and_val.npz'



training_images_root_dir = '/home/luthffi/PycharmProjects/PhDLuthffiDeepLearningDell/CroppedData_dummy/Train_dummy_cropped'
training_images = glob.glob(os.path.join(training_images_root_dir, '.jpg'))

validation_images_root_dir = '/home/luthffi/PycharmProjects/PhDLuthffiDeepLearningDell/CroppedData_dummy/CV_dummy_cropped'
validation_images = glob.glob(os.path.join(validation_images_root_dir, '.jpg'))

testing_images_root_dir = '/home/luthffi/PycharmProjects/PhDLuthffiDeepLearningDell/CroppedData_dummy/Test_dummy_cropped'
testing_images = glob.glob(os.path.join(testing_images_root_dir, '.jpg'))


dllib_dat_file = '/home/luthffi/PycharmProjects/PhDLuthffiDeepLearningDell/dlib-models-master/shape_predictor_68_face_landmarks.dat'


class EyeDetectionDataSet(Dataset):

    def __init__(self, image_dir, landmarks_file, transforms=None):

        self.predictor = dlib.shape_predictor(landmarks_file)
        self.detector = dlib.get_frontal_face_detector()
        self.image_dir = image_dir
        self.transform = transforms
        self.images_files = [name for name in os.listdir(self.image_dir) if os.path.isfile(name)]

        self.FACIAL_LANDMARKS_IDXS = OrderedDict([
            ("right_eye", (37, 42)),
            ("left_eye", (42, 48)),
            ("face", (1, 68)),
        ])

    def __len__(self):
        return len(self.images_files)

    def __getitem__(self, idx):
        image = io.imread(os.path.join(self.image_dir, self.images_files[idx]))
        # landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        # print(image)
        faces_bb = self.detector(image, 1)
        for face_bb in faces_bb:


            landmarks = face_utils.shape_to_np(faces_bb)
            sample = {'image': image, 'landmarks': landmarks}

            if self.transform:
                sample = self.transform(sample)

            # we can get the label from the file name of the image, e.g. Cropped.01.s01.KinectFrame5011.0.jpg
            label = self.images_files[idx].split('.')[-2]
            return sample, label

from torchvision import transforms, datasets


class StretchTransform(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        stretched_landmarks = self.stretch_landmarks(landmarks)

        return {'image': image, 'landmarks': stretched_landmarks}

    def stretch_landmarks(self, landmarks):
        # loop over the subset of facial landmarks, drawing the specific face parts

        visual_check = self.image.copy()
        extracted_data = self.image.copy()

        for (x, y) in landmarks[i:j]:
            cv2.circle(visual_check, (x, y), 1, (0, 0, 255), -1)  # red facial mark

            # extract the ROI of the face region as a separate image
            (x, y, w, h) = cv2.boundingRect(np.array([landmarks[i:j]]))
            offset = (abs(w - h)) / (2)
            extraspace = 10  # random.randrange(10, 15)   # add min max here and time 5 for every detected eyeL, eyeR, and face
            if w > h:
                roi = extracted_data[y - math.floor(offset) - extraspace: y + h + math.ceil(offset) + extraspace,
                      x - extraspace: x + w + extraspace]
            else:
                roi = extracted_data[y - extraspace: y + h + extraspace,
                      x - math.floor(offset) - extraspace: x + w + math.ceil(offset) + extraspace]

    def CropEyesAndFaceTransform(self, leftEye, rightEye, Face, labels):


    def LightVariation(self, brightness, contrast):

        brightness =
        contrast =



transformation_pipeline = transforms.Compose([
        CropEyesAndFaceTransform(EyeDetectionDataSet),
        StretchTransform(10, 15),
        transforms.Resize(64),
        LightVariation()
    ])



training_dataset = EyeDetectionDataSet(training_images_root_dir, dllib_dat_file, transforms=transformation_pipeline)
val_dataset = EyeDetectionDataSet(validation_images_root_dir, dllib_dat_file, transforms=transformation_pipeline)

dataset_loader = torch.utils.data.DataLoader(training_dataset,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)


# count data that primarily loaded
# count data that successfully loaded to the network
# count data that augmented

def load_data(file):
    npzfile = np.load(file)
    train_eye_left = npzfile["train_eye_left"]
    train_eye_right = npzfile["train_eye_right"]
    train_face = npzfile["train_face"]
    train_y = npzfile["train_y"]

    val_eye_left = npzfile["val_eye_left"]
    val_eye_right = npzfile["val_eye_right"]
    val_face = npzfile["val_face"]
    val_y = npzfile["val_y"]

    # Normalize all data
    train_eye_left = normalize(train_eye_left)
    train_eye_right = normalize(train_eye_right)
    train_face = normalize(train_face)
    train_y = train_y.astype('float32')

    val_eye_left = normalize(val_eye_left)
    val_eye_right = normalize(val_eye_right)
    val_face = normalize(val_face)
    val_y = val_y.astype('float32')
    # Also normalize validation
    # Change y to float32

    return [train_eye_left, train_eye_right, train_face, train_y], [val_eye_left, val_eye_right, val_face, val_y]


def normalize(Train_data):
    shape = Train_data.shape
    Train_data = np.reshape(Train_data, (shape[0], -1))
    Train_data = Train_data.astype('float32') / 255.  # scaling
    Train_data = Train_data - np.mean(Train_data, axis=0)  # normalizing
    return np.reshape(Train_data, shape)


def normalize(Val_data):
    shape = Val_data.shape
    Val_data = np.reshape(Val_data, (shape[0], -1))
    Val_data = Val_data.astype('float32') / 255.  # scaling
    Val_data = Val_data - np.mean(Val_data, axis=0)  # normalizing
    return np.reshape(Val_data, shape)


batch_size = 10


class MyTrainData(Dataset):
    def __init__(self, is_train):
        train_data = load_data(file)
        index = 0
        if not is_train:
            index = 1
        self.train_eye_left, self.train_eye_right, self.train_face, self.train_y = train_data[index]

    def __getitem__(self, index):
        train_eye_left = np.transpose(self.train_eye_left[index], (2, 0, 1))
        train_eye_right = np.transpose(self.train_eye_right[index], (2, 0, 1))
        train_face = np.transpose(self.train_face[index], (2, 0, 1))
        return train_eye_left, train_eye_right, train_face, self.train_y[index]

    def __len__(self):
        return self.train_eye_left.shape[0]


class MyValData(Dataset):
    def __init__(self, is_val):
        Val_data = load_data(file)
        index = 0
        if not is_val:
            index = 1
        self.val_eye_left, self.val_eye_right, self.val_face, self.val_y = Val_data[index]

    def __getitem__(self, index):
        val_eye_left = np.transpose(self.val_eye_left[index], (2, 0, 1))
        val_eye_right = np.transpose(self.val_eye_right[index], (2, 0, 1))
        val_face = np.transpose(self.val_face[index], (2, 0, 1))
        return val_eye_left, val_eye_right, val_face, self.val_y[index]

    def __len__(self):
        return self.val_eye_left.shape[0]


train_loader = torch.utils.data.DataLoader(
    MyTrainData(is_train=True),
    batch_size=batch_size, shuffle=True)


val_loader = torch.utils.data.DataLoader(
    MyValData(is_val=True),
    batch_size=batch_size, shuffle=True)

print("checkfile dataset path:", file)
print("check...load data")

##### define convolutional neural network
##### Network Parameters
img_size = 64
n_channel = 3


#####Model
class ItrackerImageModel(nn.Module):
    # Used for both eyes (with shared weights) and the face (with unqiue weights)
    def __init__(self):
        super(ItrackerImageModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


class FaceImageModel(nn.Module):
    def __init__(self):
        super(FaceImageModel, self).__init__()
        self.conv = ItrackerImageModel()
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class ITrackerModel(nn.Module):
    def __init__(self):
        super(ITrackerModel, self).__init__()
        self.eyeModel = ItrackerImageModel()
        self.faceModel = FaceImageModel()
        # self.gridModel = FaceGridModel()
        # Joining both eyes
        self.eyesFC = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
        )
        # Joining everything
        self.fc = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, eyesLeft, eyesRight, faces):  # , remove faceGrids
        # Eye nets
        xEyeL = self.eyeModel(eyesLeft)
        xEyeR = self.eyeModel(eyesRight)
        # Cat and FC
        xEyes = torch.cat((xEyeL, xEyeR), 1)
        xEyes = self.eyesFC(xEyes)
        # Face net
        xFace = self.faceModel(faces)
        # xGrid = self.gridModel(faceGrids)
        # Cat all
        x = torch.cat((xEyes, xFace), 1)  # remove xGrid
        x = self.fc(x)
        return x


print("check....initialize Network...")
network = ITrackerModel()

##### define loss function and optimizer
base_lr = 0.0001
momentum = 0.9
weight_decay = 1e-4
print_freq = 10
prec1 = 0
best_prec1 = 1e20
lr = base_lr
num_epochs = 10

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(network.parameters(), lr=base_lr, weight_decay=weight_decay)

##### Train the Network on train and cross validation dataset
##### evaluate loss function and optimizer
##### evaluate error accuracy, recall, precission, f1-score
print('Final Check Samples...')
print('Number of training samples =', len(MyTrainData(is_train=True)))
print('Number of validation samples =', len(MyValData(is_val=False)))  # need to check for validation sample
print('Enter loop...')
# since = time.time()

train_losses_plot = []
val_losses_plot = []

train_accuracy_plot = []
val_accuracy_plot = []

for epoch in range(0, num_epochs):
    for phase in ['train', 'validate']:
        if phase == 'train':
            network.train()

            correct_train = 0
            total_train = 0
            total_train_running_loss = 0
            total_train_iterations = 0
            for i, (train_eye_left, train_eye_right, train_face, train_y) in enumerate(train_loader):

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                output_train = network(train_eye_left, train_eye_right, train_face)
                loss_train = criterion(output_train, train_y)
                loss_train.backward()
                optimizer.step()

                total_train_running_loss += loss_train.item()
                total_train_iterations += 1

                # Rescale network output to [0, 1] from sigmoid
                network_out_train = output_train.data
                network_out_train = (network_out_train + 1) / 2
                # Create array for predictions
                prediction_train = torch.zeros_like(network_out_train)
                for sample_index in range(network_out_train.shape[0]):
                    # Predict 1 if larger than 0.5 otherwise 0
                    prediction_train[sample_index] = 1 if network_out_train[sample_index] > 0.5 else 0
                total_train += train_y.size(0)
                # Count how many times prediction is equal to target
                correct_train += (prediction_train.long() == train_y.long()).sum().item()

                loss_train = (total_train_running_loss / total_train_iterations)
                accuracy_train = (100 * correct_train) / float(total_train)

            print('Epoch:{}/{} -- Train_Loss: {} -- Train_Accuracy: {}%'.format(epoch, num_epochs, loss_train,
                                                                                accuracy_train))  # recall (TP)
            train_losses_plot.append(loss_train)
            train_accuracy_plot.append(accuracy_train)

        if phase == 'validate':

            network.eval()
            correct_val = 0
            total_val = 0
            total_val_running_loss = 0
            total_val_iterations = 0
            for i, (val_eye_left, val_eye_right, val_face, val_y) in enumerate(val_loader):
                # zero the parameter gradients
                # optimizer.zero_grad()

                # forward + backward + optimize
                output_val = network(val_eye_left, val_eye_right, val_face)
                loss_val = criterion(output_val, val_y)
                # loss.backward()
                # optimizer.step()

                total_val_running_loss += loss_val.item()
                total_val_iterations += 1

                # Rescale network output to [0, 1] from sigmoid
                network_out_val = output_val.data
                network_out_val = (network_out_val + 1) / 2

                # Create array for predictions
                prediction_val = torch.zeros_like(network_out_val)
                for sample_index in range(network_out_val.shape[0]):
                    # Predict 1 if larger than 0.5 otherwise 0
                    prediction_val[sample_index] = 1 if network_out_val[sample_index] > 0.5 else 0
                total_val += val_y.size(0)

                # Count how many times prediction is equal to target
                correct_val += (prediction_val.long() == val_y.long()).sum().item()

                loss_val = (total_val_running_loss / total_val_iterations)
                accuracy_val = (100 * correct_val) / float(total_val)

            print('Epoch:{}/{} -- Val_Loss: {} -- Val_Accuracy: {}%'.format(epoch, num_epochs, loss_val, accuracy_val))
            val_losses_plot.append(loss_val)
            val_accuracy_plot.append(accuracy_val)

print("Trained network successfully saved...")
print("Training checked...")
##### save the trained networl
torch.save(network.state_dict(), 'Model_eyeContactTracker250.pth')

print("Let's plot the result: ")

# Train_Loss vs Val_Loss for all Epoch
print("Train_Loss vs Val_Loss for All Epoch")
fig = plt.figure()
plt.title('Train Loss and Validation Loss')
plt.plot(train_losses_plot, label='Train Loss')
plt.plot(val_losses_plot, "--", label='Val. Loss')
plt.legend()
plt.savefig('./Plot_loss_Train_Val.png')
plt.show()

# Val_loss vs Val_acc for all epoch
print("Val_Loss vs Val_Accuracy for All Epoch")
fig = plt.figure()
plt.title('Train Accuracy and Validation Accuracy')
plt.plot(train_accuracy_plot, label='Train Loss')
plt.plot(val_accuracy_plot, "--", label='Val. Loss')
plt.legend()
plt.savefig('./Plot_Accuracy_Train_Val.png')
plt.show()
