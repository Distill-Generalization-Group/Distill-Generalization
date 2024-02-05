import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
import tqdm
import matplotlib.pyplot as plt
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug, my_create_network


def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='noise', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--use_ModelPool', action="store_true", help='whether use model pool or not')
    parser.add_argument('--use_Dropout', action="store_true", help='whether use dropout or not')
    parser.add_argument('--use_KD', action="store_true", help='whether use knowledge distillation or not')
    parser.add_argument('--temperature', type=float, default=1.5, help='temperature for knowledge distillation')
    parser.add_argument('--alpha', type=float, default=0.5, help='weight factor for knowledge distillation')
    parser.add_argument('--temperature_decay', type=float, default=0.9, help='temperature decay factor for knowledge distillation')
    parser.add_argument('--use_stored', action="store_true", help='whether use store or not')
    parser.add_argument('--cached_img_path', type=str, default="", help='path to save cached images')
    parser.add_argument('--eval_model', type=str, default="", help='evaluation net type')
    parser.add_argument('--img_path', type=str, required=True, help='path to save images')

    args = parser.parse_args()
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # eval_it_pool = np.arange(0, args.Iteration+1, 500).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    eval_it_pool = np.arange(1000, args.Iteration+1, 1000).tolist()
    print('eval_it_pool: ', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    
    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []
    
    eval_model_list = ["MLP", "ConvNet", "LeNet", "AlexNet", "VGG11", "ResNet18"]
    #eval_model_list = ["MLP", "ConvNet"]
    model_acc_list = []
    if args.use_stored:
        if args.eval_model == "all":
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            plt.xlabel("KD Temperature")
            plt.ylabel("Test Accuracy (%)")
            #plt.title("temperature vs accuracy")
            for index in range(len(eval_model_list)):
                args.eval_model = eval_model_list[index]
                tem_list = [0.0, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]
                #tem_list = [0.0, 0.05]
                #from 1 to len(tem_list)
                x_list = [i for i in range(1, len(tem_list)+1)]
                tem_acc_list = []
                for i in tqdm.tqdm(range(len(tem_list))):
                    args.temperature = tem_list[i]
                    print("use stored data")
                    assert args.cached_img_path != ""
                    assert args.eval_model != ""
                    if args.dsa:
                        args.epoch_eval_train = 1000
                        args.dc_aug_param = None
                        print('DSA augmentation strategy: \n', args.dsa_strategy)
                        print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                    else:
                        args.dc_aug_param = get_daparam(args.dataset, args.model, args.eval_model, args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
                        print('DC augmentation parameters: \n', args.dc_aug_param)
                    acc_list = []
                    raw_data = torch.load(args.cached_img_path)
                    image_syn_eval = raw_data['data'][0][0].to(args.device)
                    label_syn_eval = raw_data['data'][0][1].to(args.device)
                    for _ in tqdm.tqdm(range(args.num_eval)):
                        print("Load synthetic data from: ", args.cached_img_path)
                        net_teacher = get_network("ConvNet", channel, num_classes, im_size).to(args.device)
                        net_eval = get_network(args.eval_model, channel, num_classes, im_size).to(args.device) # get a random model
                        it_eval = int(args.cached_img_path.split("_")[-1].split(".")[0])
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, teacher_model=net_teacher)
                        acc_list.append(acc_test)
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(acc_list), args.eval_model, np.mean(acc_list), np.std(acc_list)))
                    tem_acc_list.append(np.mean(acc_list))
                model_acc_list.append({"model": args.eval_model, "max_acc": max(tem_acc_list), "max_tem": tem_list[tem_acc_list.index(max(tem_acc_list))]})
                ax.plot(x_list, tem_acc_list, label=args.eval_model)
                ax.set_xticks(x_list)
                ax.set_xticklabels(tem_list)
                ax.legend()
            plt.savefig("img/"+args.img_path, format='svg',dpi=150)
            print("final result: \n", model_acc_list)
        return

    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data
        else:
            print('initialize synthetic data from random noise')


        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)
        print('%s training begins'%get_time())

        for it in range(args.Iteration+1):

            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                current_accs = dict()
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                    if args.dsa:
                        args.epoch_eval_train = 1000
                        args.dc_aug_param = None
                        print('DSA augmentation strategy: \n', args.dsa_strategy)
                        print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                    else:
                        args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, args.ipc) # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
                        print('DC augmentation parameters: \n', args.dc_aug_param)

                    if args.dsa or args.dc_aug_param['strategy'] != 'none':
                        args.epoch_eval_train = 1000  # Training with data augmentation needs more epochs.
                    else:
                        args.epoch_eval_train = 300

                    accs = []
                    for it_eval in range(args.num_eval):
                        net_teacher = get_network("ConvNet", channel, num_classes, im_size).to(args.device)
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, teacher_model=net_teacher)
                        accs.append(acc_test)
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))

                    if it == args.Iteration: # record the final results
                        accs_all_exps[model_eval] += accs
                    current_accs[model_eval] = accs
                with open("result.txt", 'a') as f:
                    print("dataset: ", args.dataset)
                    print("ipc: ", args.ipc)
                    print("iteration: ", it)
                    print("use_ModelPool: ", args.use_ModelPool)
                    print("use_Dropout: ", args.use_Dropout)
                    print("use_KD: ", args.use_KD)
                    if args.use_KD:
                        print("temperature: ", args.temperature, "alpha: ", args.alpha)
                    f.write("dataset: " + args.dataset + "\n")
                    f.write("ipc: " + str(args.ipc) + "\n")
                    f.write("iteration: " + str(it) + "\n")
                    f.write("use_ModelPool: " + str(args.use_ModelPool) + "\n")
                    f.write("use_Dropout: " + str(args.use_Dropout) + "\n")
                    f.write("use_KD: " + str(args.use_KD) + "\n")
                    if args.use_KD:
                        f.write("temperature: " + str(args.temperature) + ", alpha: " + str(args.alpha) + "\n")
                    for key in model_eval_pool:
                        accs = current_accs[key]
                        print('evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(len(accs), key, np.mean(accs)*100, np.std(accs)*100))
                        f.write('evaluate %d random %s, mean  = %.2f%%  std = %.2f%%\n'%(len(accs), key, np.mean(accs)*100, np.std(accs)*100))
                    f.write("\n")

                ''' visualize and save '''
                save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, exp, it))
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.


            ''' Train synthetic data '''
            net = get_network(args.model, channel, num_classes, im_size, dropout=args.use_Dropout).to(args.device)
            if args.use_ModelPool:
                if it <= 500:
                    net = get_network(args.model, channel, num_classes, im_size, dropout=args.use_Dropout).to(args.device)
                else:
                    net = get_network("ModelPool", channel, num_classes, im_size, dropout=args.use_Dropout).to(args.device) # randomly select a model from the model pool
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()
            loss_avg = 0
            args.dc_aug_param = None  # Mute the DC augmentation when learning synthetic data (in inner-loop epoch function) in oder to be consistent with DC paper.


            for ol in range(args.outer_loop):

                ''' freeze the running mu and sigma for BatchNorm layers '''
                # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
                # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
                # This would make the training with BatchNorm layers easier.

                BN_flag = False
                BNSizePC = 16  # for batch normalization
                for module in net.modules():
                    if 'BatchNorm' in module._get_name(): #BatchNorm
                        BN_flag = True
                if BN_flag:
                    img_real = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=0)
                    net.train() # for updating the mu, sigma of BatchNorm
                    output_real = net(img_real) # get running mu, sigma
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():  #BatchNorm
                            module.eval() # fix mu and sigma of every BatchNorm layer


                ''' update synthetic data '''
                loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                    lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    output_real = net(img_real)
                    loss_real = criterion(output_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    output_syn = net(img_syn)
                    loss_syn = criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                    loss += match_loss(gw_syn, gw_real, args)

                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()
                loss_avg += loss.item()

                if ol == args.outer_loop - 1:
                    break


                ''' update network '''
                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
                for il in range(args.inner_loop):
                    epoch('train', trainloader, net, optimizer_net, criterion, args, aug = True if args.dsa else False)


            loss_avg /= (num_classes*args.outer_loop)

            if it%10 == 0:
                print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))

            if it % 1000 == 0 and it != 0:
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc_%d.pt'%(args.method, args.dataset, args.model, args.ipc, it)))
            
            # if it == args.Iteration: # only record the final results
            #     data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
            #     torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc.pt'%(args.method, args.dataset, args.model, args.ipc)))


    print('\n==================== Final Results ====================\n')
    print("dataset: ", args.dataset)
    print("ipc: ", args.ipc)
    print("use_ModelPool: ", args.use_ModelPool)
    print("use_Dropout: ", args.use_Dropout)
    print("use_KD: ", args.use_KD)
    if args.use_KD:
        print("temperature: ", args.temperature, ", alpha: ", args.alpha)
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))



if __name__ == '__main__':
    main()


