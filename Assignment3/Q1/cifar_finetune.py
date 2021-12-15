'''
This is starter code for Assignment 2 Problem 1 of CMPT 726 Fall 2020.
The file is adapted from the repo https://github.com/chenyaofo/CIFAR-pretrained-models
'''
import os
import ssl
import pickle
import numpy as np
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

NUM_EPOCH_PER_LAMBDA = 10
VAL_AFTER = 3
TRAIN_MINI_BATCHES = 20
TEST_MINI_BATCHES = 20
PATH = "models/best_model.pt"
TRAIN_LOSSES_PATH = "models/train_losses.pkl"
VAL_LOSSES_PATH = "models/val_losses.pkl"
IMAGE_SAVE_PATH = "images/"
lambdas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]

cifar_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

ssl._create_default_https_context = ssl._create_unverified_context

def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
	expansion = 1
	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out

class CifarResNet(nn.Module):
	def __init__(self, block, layers, num_classes=100):
		super(CifarResNet, self).__init__()
		self.inplanes = 16
		self.conv1 = conv3x3(3, 16)
		self.bn1 = nn.BatchNorm2d(16)
		self.relu = nn.ReLU(inplace=True)

		self.layer1 = self._make_layer(block, 16, layers[0])
		self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(64 * block.expansion, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x

######################################################
####### Do not modify the code above this line #######
######################################################

class cifar_resnet20(nn.Module):
	def __init__(self):
		super(cifar_resnet20, self).__init__()
		ResNet20 = CifarResNet(BasicBlock, [3, 3, 3])
		url = "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar100_resnet20-23dac2f1.pt"
		ResNet20.load_state_dict(model_zoo.load_url(url))
		modules = list(ResNet20.children())[:-1]
		backbone = nn.Sequential(*modules)
		self.backbone = nn.Sequential(*modules)
		self.fc = nn.Linear(64, 10)

	def forward(self, x):
		out = self.backbone(x)
		out = out.view(out.shape[0], -1)
		return self.fc(out)

def loadAndPlot():
	train_loss_hist = pickle.load(open(TRAIN_LOSSES_PATH, "rb"))
	val_loss_hist = pickle.load(open(VAL_LOSSES_PATH, "rb"))
	mean_train_loss, stdev_train_loss = [], []
	mean_val_loss, stdev_val_loss = [], []

	# Un comment to print for Markdown
	# train_df = pd.DataFrame.from_dict(train_loss_hist)
	# val_df = pd.DataFrame.from_dict(val_loss_hist)
	# print(train_df.to_markdown())
	# print(val_df.to_markdown())

	if not os.path.exists(IMAGE_SAVE_PATH):
		print("Creating a directory ("+ IMAGE_SAVE_PATH +") to save graphs ...")
		os.makedirs(IMAGE_SAVE_PATH)
	
	plt.plot([], [], " ", label="Lambda")
	for lambda2, loss in train_loss_hist.items():
		mean_train_loss.append(np.mean(np.array(loss)))
		stdev_train_loss.append(np.std(np.array(loss)))
		plt.plot(list(range(1, NUM_EPOCH_PER_LAMBDA+1)), loss, label=lambda2)
	plt.legend()
	plt.title("Training Losses for different lambdas")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.savefig(os.path.join(IMAGE_SAVE_PATH, "train.png"))
	plt.show()

	plt.plot([], [], " ", label="Lambda")
	for lambda2, loss in val_loss_hist.items():
		mean_val_loss.append(np.mean(np.array(loss)))
		stdev_val_loss.append(np.std(np.array(loss)))
		plt.plot(list(range(VAL_AFTER, NUM_EPOCH_PER_LAMBDA, VAL_AFTER)), loss, label=lambda2)
	plt.legend()
	plt.title("Validation Losses for different lambdas")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.savefig(os.path.join(IMAGE_SAVE_PATH, "test.png"))
	plt.show()

	plt.errorbar(list(train_loss_hist.keys()), mean_train_loss, stdev_train_loss, label="Train Loss")
	plt.errorbar(list(val_loss_hist.keys()), mean_val_loss, stdev_val_loss, label="Validation Loss")
	# plt.xscale("log", nonposx="clip")
	# plt.yscale("log", nonposy="clip")
	plt.legend()
	plt.title("Error plots for Train and Val losses wrt Lambda")
	plt.xlabel("Lambda")
	plt.ylabel("Loss")
	plt.savefig(os.path.join(IMAGE_SAVE_PATH, "lambda_errorbar.png"))
	plt.show()

def testModel(model, testloader):
	"""
	Test the best model
	"""
	test_loss = 0.0
	correct, total = 0, 0
	lab_prediction, lab_actual = [], []
	correct_pred = {classname: 0 for classname in cifar_classes}
	total_pred = {classname: 0 for classname in cifar_classes}

	for i, data in enumerate(testloader, 0):
		# No need to update weights during validation phase
		model.eval()

		with torch.no_grad():
			# get the inputs
			# data to GPU
			inputs, labels = data[0].to(device), data[1].to(device)
			# just forward no backward and don't optimize
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			test_loss += loss.item()

			if (i % TEST_MINI_BATCHES == TEST_MINI_BATCHES-1):
				print("Batch %4d | Loss %f"%(i, test_loss/TEST_MINI_BATCHES))
				test_loss = 0.0

			val, predictions = torch.max(nn.functional.softmax(outputs.data, dim=1), 1)
			
			total += labels.size(0)
			correct += (predictions == labels).sum().item()

			for label, prediction in zip(labels, predictions):
				if label == prediction:
					correct_pred[cifar_classes[label]] += 1
				total_pred[cifar_classes[label]] += 1

			y_pred = predictions.cpu().numpy()
			y_true = labels.cpu().numpy()

			lab_prediction.append(y_pred.reshape(len(y_pred), 1))
			lab_actual.append(y_true.reshape(len(y_true), 1))

			# print for only the last 
			if (i == len(testloader)-1):
				probs = nn.functional.softmax(outputs.data, dim=1).cpu().numpy()
				preds = [cifar_classes[p] for p in predictions.cpu().numpy()]
				labs = [cifar_classes[l] for l in labels.cpu().numpy()]

				df = pd.DataFrame(probs, index=labs, columns=cifar_classes)
				with open(os.path.join("models", "test.html"), 'w') as f:
					f.write(df.reset_index().style.highlight_max(color="yellow", axis=1).render())
				print(df.to_markdown())
				grid = make_grid(inputs, nrow=1).cpu()
				plt.imshow(grid.permute(1, 2, 0), interpolation='nearest')
				plt.show()
	
	lab_prediction, lab_actual = np.vstack(lab_prediction), np.vstack(lab_actual)
	accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
	conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)

	print("Accuracy:", accuracy*100, "%")
	print(classification_report(y_true=y_true, y_pred=y_pred))
	plt.imshow(conf_mat)
	plt.title("Confusion Matrix")
	plt.savefig(os.path.join(IMAGE_SAVE_PATH, "confusion_matrix.png"))
	plt.show()

	for classname, correct_count in correct_pred.items():
		accuracy = 100 * float(correct_count) / total_pred[classname]
		print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))

if __name__ == "__main__":
	model = cifar_resnet20()
	transform = transforms.Compose([transforms.ToTensor(),
									transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
									std=(0.2023, 0.1994, 0.2010))])
	trainset = datasets.CIFAR10("./data", download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

	valset = datasets.CIFAR10("./data", download=False, transform=transform, train=False)
	valloader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=True, num_workers=2)

	criterion = nn.CrossEntropyLoss()

	if not os.path.exists(PATH.split("/")[0]):
		print("Creating a directory ("+ PATH.split("/")[0] +"/) to save models ...")
		os.makedirs(PATH.split("/")[0])
	else:
		if os.path.isfile(PATH):
			print("Loading the best model...")
			model.load_state_dict(torch.load(PATH))

	# to take advantage of CUDA enabled GPU
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.to(device)
	train_loss_hist, val_loss_hist = {}, {}
	print("The model is running on:", device)

	NUM_TRAIN_BATCHES, NUM_VAL_BATCHES = len(trainloader), len(valloader)
	lowest_val_loss = float("inf")

	# Do the training
	for lambda2 in lambdas:
		print("Using Lambda = %f" % (lambda2))
		optimizer = optim.SGD(list(model.fc.parameters()), lr=0.001, momentum=0.9, weight_decay=lambda2)

		for epoch in range(NUM_EPOCH_PER_LAMBDA):  # loop over the dataset multiple times
			running_loss, val_loss = 0.0, 0.0
			
			# Train Loop
			for i, data in enumerate(trainloader, 0):
				model.train()
				# get the inputs
				# data to GPU
				inputs, labels = data[0].to(device), data[1].to(device)
				# zero the parameter gradients
				optimizer.zero_grad()
				
				# Manual update
				# l2=0
				# for param in model.parameters():
				# 	l2 += (param**2).sum()

				# forward + backward + optimize
				outputs = model(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()
				# running_loss += loss.item() + (lambda2 * l2/ (2 * list(labels.shape)[0]))
				running_loss += loss.item()
				if (i % TRAIN_MINI_BATCHES == TRAIN_MINI_BATCHES-1):    # print every 20 mini-batches
					print("[%d, %5d] loss: %.3f" %(epoch + 1, i + 1, running_loss/TRAIN_MINI_BATCHES))
					# round to the nearest tens (when the previous if condition becomes true)
					# to run it only on the last batch of the epoch (save the loss in dict - to be saved in a file)
					if (i % round(NUM_TRAIN_BATCHES, -1) == round(NUM_TRAIN_BATCHES, -1) - 1):
						if lambda2 not in train_loss_hist.keys():
							train_loss_hist[lambda2] = [running_loss/TRAIN_MINI_BATCHES]
						else:
							train_loss_hist[lambda2].append(running_loss/TRAIN_MINI_BATCHES)
					running_loss = 0.0
			
			# Validate after every 3 epochs
			if epoch % VAL_AFTER == VAL_AFTER-1:
				# Validation Loop
				for i, data in enumerate(valloader, 0):
					# No need to update weights during validation phase
					model.eval()
					with torch.no_grad():
						# get the inputs
						# data to GPU
						inputs, labels = data[0].to(device), data[1].to(device)
						# just forward no backward and don't optimize
						outputs = model(inputs)
						loss = criterion(outputs, labels)

					val_loss += loss.item()
					if (i % NUM_VAL_BATCHES == NUM_VAL_BATCHES-1):
						if lambda2 not in val_loss_hist.keys():
							val_loss_hist[lambda2] = [val_loss/NUM_VAL_BATCHES]
						else:
							val_loss_hist[lambda2].append(val_loss/NUM_VAL_BATCHES)
						
						if (val_loss <= lowest_val_loss):
							print("\nPrevious Validation Loss: %.5f, New Validation Loss: %.5f" %(lowest_val_loss/NUM_VAL_BATCHES, val_loss/NUM_VAL_BATCHES))
							lowest_val_loss = val_loss
							print("Saving model weights to " + PATH + " ...")
							torch.save(model.state_dict(), PATH)

				print("\n[%d] validation loss: %.5f, lambda: %f\n" % ((epoch+1), val_loss/NUM_VAL_BATCHES, lambda2))

		print("Saving Losses to a file (update after each lambda)...")
		with open(TRAIN_LOSSES_PATH, "wb") as train_file:
			pickle.dump(train_loss_hist, train_file)
		with open(VAL_LOSSES_PATH, "wb") as val_file:
			pickle.dump(val_loss_hist, val_file)

	print("Finished Training")
	print("The best validation loss is: " + str(lowest_val_loss/NUM_VAL_BATCHES))
	
	loadAndPlot()
	testModel(model, valloader)