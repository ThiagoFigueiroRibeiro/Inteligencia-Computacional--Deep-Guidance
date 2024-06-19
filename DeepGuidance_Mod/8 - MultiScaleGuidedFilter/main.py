import torch
import torch.nn as nn
import torch.nn.functional as F

import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score

import os
import argparse
import time
from core.utils import calculate_Accuracy, get_img_list, get_model, get_data
from pylab import *
import random
plt.switch_backend('agg')

# --------------------------------------------------------------------------------
models_list = ["DG_Net", "M_Net", "UNet512"]
dataset_list = ['DRIVE', "STARE", "CHASEDB1"]

# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch ASOCT_Demo')

parser.add_argument('--epochs', type=int, default=500,
                    help='the epochs of this run')
parser.add_argument('--n_class', type=int, default=2,
                    help='the channel of out img, decide the num of class, ASOCT_eyes is 2/4 class')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--GroupNorm', type=bool, default=True,
                    help='decide to use the GroupNorm')
parser.add_argument('--BatchNorm', type=bool, default=False,
                    help='decide to use the BatchNorm')

# ---------------------------
# model
# ---------------------------
parser.add_argument('--datasetID', type=int, default=2,
                    help='dir of the all img')
parser.add_argument('--SubImageID', type=int, default=1,
                    help='Only for Stare Dataset')
parser.add_argument('--model_id', type=int, default=0,
                    help='the id of choice_model in models_list')
parser.add_argument('--batch_size', type=int, default=1,
                    help='the num of img in a batch')
parser.add_argument('--img_size', type=int, default=512,
                    help='the train img size')
parser.add_argument('--my_description', type=str, default='CHASEDB1_500DGNet',
                    help='some description define your train')

# ---------------------------
# GPU
# ---------------------------
parser.add_argument('--use_gpu', type=bool, default=False,
                    help='dir of the all ori img')
parser.add_argument('--gpu_avaiable', type=str, default='3',
                    help='the gpu used')

args = parser.parse_args()
print(args)
# --------------------------------------------------------------------------------
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_avaiable

def fast_test(model, args, img_list, model_name):
    softmax_2d = nn.Softmax2d()
    EPS = 1e-12

    Background_IOU = []
    Vessel_IOU = []
    ACC = []
    SE = []
    SP = []
    AUC = []

    for i, path in enumerate(img_list):
        start = time.time()
        img, imageGreys, gt, tmp_gt, img_shape, label_ori = get_data(dataset_list[args.datasetID], [path], img_size=args.img_size, gpu=args.use_gpu, flag='test')
        model.eval()

        out, side_5, side_6, side_7, side_8 = model(img, imageGreys)
        out = torch.log(softmax_2d(side_8) + EPS)

        out = F.upsample(out, size=(img_shape[0][0],img_shape[0][1]), mode='bilinear')
        out = out.cpu().data.numpy()
        y_pred =out[:,1,:,:]
        y_pred = y_pred.reshape([-1])
        ppi = np.argmax(out, 1)

        tmp_out = ppi.reshape([-1])
        tmp_gt=label_ori.reshape([-1])

        my_confusion = metrics.confusion_matrix(tmp_out, tmp_gt).astype(np.float32)
        meanIU, Acc, Se, Sp, IU = calculate_Accuracy(my_confusion)
        Auc = roc_auc_score(tmp_gt, y_pred)
        AUC.append(Auc)

        Background_IOU.append(IU[0])
        Vessel_IOU.append(IU[1])
        ACC.append(Acc)
        SE.append(Se)
        SP.append(Sp)
        end = time.time()

        print(str(i+1)+r'/'+str(len(img_list))+': '+'| Acc: {:.3f} | Se: {:.3f} | Sp: {:.3f} | Auc: {:.3f} |  Background_IOU: {:f}, vessel_IOU: {:f}'.format(Acc,Se,Sp,Auc,IU[0], IU[1])+'  |  time:%s'%(end-start))

    print('Acc: %s  |  Se: %s |  Sp: %s |  Auc: %s |  Background_IOU: %s |  vessel_IOU: %s '%(str(np.mean(np.stack(ACC))),str(np.mean(np.stack(SE))), str(np.mean(np.stack(SP))),str(np.mean(np.stack(AUC))),str(np.mean(np.stack(Background_IOU))),str(np.mean(np.stack(Vessel_IOU)))))

    # store test information
    with open(r'/logs/%s_%s.txt' % (model_name, args.my_description), 'a+') as f:
        f.write('Acc: %s  |  Se: %s |  Sp: %s |  Auc: %s |  Background_IOU: %s |  vessel_IOU: %s '%(str(np.mean(np.stack(ACC))),str(np.mean(np.stack(SE))), str(np.mean(np.stack(SP))),str(np.mean(np.stack(AUC))),str(np.mean(np.stack(Background_IOU))),str(np.mean(np.stack(Vessel_IOU)))))
        f.write('\n\n')

    #return np.mean(np.stack(Vessel_IOU))
    return np.mean(np.stack(ACC))


#RootDir = os.getcwd()
RootDir = "/code"
model_name = models_list[args.model_id]
Dataset = dataset_list[args.datasetID]
SubID = args.SubImageID
model = get_model(model_name)

model = model(n_classes=args.n_class, bn=args.GroupNorm, BatchNorm=args.BatchNorm)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

if args.use_gpu:
    model.cuda()
    print('GPUs used: (%s)' % args.gpu_avaiable)
    print('------- success use GPU --------')

EPS = 1e-12
# define path
img_list = get_img_list  (Dataset, SubID, flag='train')
test_img_list = get_img_list(Dataset, SubID, flag='test')
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
criterion = nn.NLLLoss2d()
#criterion = nn.NLLLoss()
#m = nn.LogSoftmax(dim=1)
softmax_2d = nn.Softmax2d()
IOU_best = 0

print ('This model is %s_%s_%s_%s' % (model_name, args.n_class, args.img_size,args.my_description))
#if not os.path.exists(r'%s/models/%s_%s' % (RootDir, #model_name, args.my_description)):
#    os.mkdir(r'%s/models/%s_%s' % (RootDir, model_name, #args.my_description))

#with open(r'%s/logs/%s_%s.txt' % (RootDir, model_name, #args.my_description), 'w+') as f:
#    f.write('This model is %s_%s: ' % (model_name, #args.my_description)+'\n')
#    f.write('args: '+str(args)+'\n')
#    f.write('train lens: '+str(len(img_list))+' | test #lens: '+str(len(test_img_list)))
#    f.write('\n\n----------------------------------------#-----\n\n')

BestAccuracy = 0
for epoch in range(args.epochs):
    model.train()
    begin_time = time.time()
    print ('This model is %s_%s_%s_%s' % (
        model_name, args.n_class, args.img_size, args.my_description))
    random.shuffle(img_list)

    if (epoch % 100 ==  0) and epoch != 0 and epoch < 500:
        args.lr /= 10
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for i, (start, end) in enumerate(zip(range(0, len(img_list), args.batch_size),
                                         range(args.batch_size, len(img_list) + args.batch_size,
                                               args.batch_size))):
        path = img_list[start:end]
        img, imageGreys, gt, tmp_gt, img_shape,label_ori = get_data(Dataset, path, img_size=args.img_size, gpu=args.use_gpu)
        optimizer.zero_grad()
        out, side_5, side_6, side_7, side_8 = model(img, imageGreys)
        
        #output = criterion(m(out), gt)
        out = torch.log(softmax_2d(out) + EPS)
        loss = criterion(out, gt)
        loss += criterion(torch.log(softmax_2d(side_5) + EPS), gt)
        loss += criterion(torch.log(softmax_2d(side_6) + EPS), gt)
        loss += criterion(torch.log(softmax_2d(side_7) + EPS), gt)
        loss += criterion(torch.log(softmax_2d(side_8) + EPS), gt)
        out = torch.log(softmax_2d(side_8) + EPS)

        loss.backward()
        optimizer.step()
        ppi = np.argmax(out.cpu().data.numpy(), 1)

        tmp_out = ppi.reshape([-1])
        tmp_gt = tmp_gt.reshape([-1])

        my_confusion = metrics.confusion_matrix(tmp_out, tmp_gt).astype(np.float32)
        #print(my_confusion.shape)

        meanIU, Acc, Se, Sp, IU = calculate_Accuracy(my_confusion)

        print(str('model: {:s}_{:s} | epoch_batch: {:d}_{:d} | loss: {:f}  | Acc: {:.3f} | Se: {:.3f} | Sp: {:.3f}'
                  '| Background_IOU: {:f}, vessel_IOU: {:f}').format(model_name, args.my_description,epoch, i, loss.item(), Acc,Se,Sp,
                                                                                  IU[0], IU[1]))

    print('training finish, time: %.1f s' % (time.time() - begin_time))

    if epoch % 2 == 0 and epoch != 0:
        Accuracy = fast_test(model, args, test_img_list, model_name)
        print('BestAccuracy:',BestAccuracy)
        if Accuracy > BestAccuracy:
            BestAccuracy = Accuracy
            #torch.save(model.state_dict(),
            #           '%s/models/%s_%s/%s.pth' % #(RootDir, model_name, args.my_description,str(epoch)))
            # For evaluation
            print('success save Nucleus_best model')
