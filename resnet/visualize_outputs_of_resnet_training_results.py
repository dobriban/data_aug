import matplotlib.pyplot as plt
import numpy as np
import os

# For reading the txt files made from stdout during training
def create_dictionary(filename):
    dictionary = {}
    with open(filename, 'r') as f:
        for line in f.readlines():
            if "Epoch" in line:
                parts = line.split(":")
                epoch = parts[1].strip()
            if "100/100" in line:
                parts = line.split("|")
                acc_line = parts[3]
                acc_parts = acc_line.split()
                acc = acc_parts[1][:-1]
                #print("Epoch: {}, Acc: {}".format(epoch, acc))
                dictionary[int(epoch)] = float(acc)
    return dictionary

# If the number of batches for test and train are the same, need to take every other
def create_dictionary_sbs(filename):
    dictionary = {}
    take = False
    with open(filename, 'r') as f:
        for line in f.readlines():
            if "Epoch" in line:
                parts = line.split(":")
                epoch = parts[1].strip()
            if "100/100" in line:
                if take:
                    parts = line.split("|")
                    acc_line = parts[3]
                    acc_parts = acc_line.split()
                    acc = acc_parts[1][:-1]
                    #print("Epoch: {}, Acc: {}".format(epoch, acc))
                    dictionary[int(epoch)] = float(acc)
                    take = False
                else:
                    take = True
    return dictionary

def create_avg_dictionary(dirname):
    dictionary = {}
    for file in os.listdir(dirname):
        print(file)
        with open(os.path.join(dirname, file)) as f:
            for line in f.readlines():
                if "Epoch" in line:
                    parts = line.split(":")
                    epoch = parts[1].strip()
                if "100/100" in line:
                    parts = line.split("|")
                    acc_line = parts[3]
                    acc_parts = acc_line.split()
                    acc = acc_parts[1][:-1]
                    
                    #print("Epoch: {}, Acc: {}".format(epoch, acc))
                    
                    try:
                        out = dictionary[int(epoch)]
                        old_avg = out[0]
                        old_sum = out[1]
                        old_n = out[2]
                    except KeyError:
                        old_avg = 0
                        old_sum = 0
                        old_n = 0
                    
                    new_n = old_n + 1
                    new_sum = old_sum + float(acc)
                    new_avg = update_running_avg(old_sum, float(acc), new_n)
                    
                    #print("New Avg:{}, New sum: {}, New n: {}".format(new_avg, new_sum, new_n))
                    dictionary[int(epoch)] = [new_avg, new_sum, new_n]
                    
    dictionary_final_update(dictionary)                
    return dictionary
                    
def update_running_avg(old_sum, new_val, n):
    return float(old_sum + new_val)/n

def dictionary_final_update(dictionary):
    for key in dictionary.keys():
        out = dictionary[key]
        dictionary[key] = out[0]

def create_mean_std_dictionary(dirname):
    dictionary = {}
    for file in os.listdir(dirname):
        print(file)
        with open(os.path.join(dirname, file)) as f:
            for line in f.readlines():
                if "Epoch" in line:
                    parts = line.split(":")
                    epoch = parts[1].strip()
                if "100/100" in line:
                    parts = line.split("|")
                    acc_line = parts[3]
                    acc_parts = acc_line.split()
                    acc = acc_parts[1][:-1]
                    
                    try:
                        val = dictionary[int(epoch)]
                    except KeyError:
                        val = []
                    
                    val.append(float(acc))
                    dictionary[int(epoch)] = val
                    
    dictionary_final_update_std(dictionary)
    return dictionary

def dictionary_final_update_std(dictionary):
    for key in dictionary.keys():
        val = dictionary[key]
        mean = np.mean(val)
        stddev = np.std(val)
        dictionary[key] = [mean, stddev]

if __name__ == "__main__" :
    import argparse

    parser = argparse.ArgumentParser(description='Process Output of Repeated Resnet Training Experiments')
    parser.add_argument('--aug_flip_dir', metavar='path', default='./rep_out/full/aug_flip',
                        help='the path to directory with all horizontal flip augmentation results')
    parser.add_argument('--aug_dir', metavar='path', default='./rep_out/full/aug',
                        help='the path to directory with crop and flip augmentation results')
    parser.add_argument('--no_aug_dir', metavar='path', default='./rep_out/full/no_aug',
                        help='the path to directory with no augmentation results')
    parser.add_argument('--plot_title', default='CIFAR10 Training Data', help='the title of the resulting plot')
    parser.add_argument('--save_name', default='compare_aug_results.png', help='the name of the resulting save file (png)')
    parser.add_argument('--errorbar', action='store_true', help='whether to include errors bar or not')
    args = parser.parse_args()

    if args.errorbar:
        aug_flip_dict = create_mean_std_dictionary(args.aug_flip_dir)
        aug_dict = create_mean_std_dictionary(args.aug_dir)
        no_aug_dict = create_mean_std_dictionary(args.no_aug_dir)

        no_aug_data = {
            'x': no_aug_dict.keys(),
            'y': [item[0] for item in no_aug_dict.values()],
            'yerr': [item[1] for item in no_aug_dict.values()]
        }

        aug_flip_data = {
            'x': aug_flip_dict.keys(),
            'y': [item[0] for item in aug_flip_dict.values()],
            'yerr': [item[1] for item in aug_flip_dict.values()]
        }

        aug_data = {
            'x': aug_dict.keys(),
            'y': [item[0] for item in aug_dict.values()],
            'yerr': [item[1] for item in aug_dict.values()]
        }

        for data in [no_aug_data, aug_flip_data, aug_data]:
            plt.errorbar(**data, alpha=.85, elinewidth=0.25, linestyle=':')
            data = {
                'x': data['x'],
                'y1': [y - e for y, e in zip(data['y'], data['yerr'])],
                'y2': [y + e for y, e in zip(data['y'], data['yerr'])]}
            plt.fill_between(**data, alpha=.15)

    else:
        aug_flip_dict = create_avg_dictionary(args.aug_flip_dir)
        aug_dict = create_avg_dictionary(args.aug_dir)
        no_aug_dict = create_avg_dictionary(args.no_aug_dir)

        plt.plot(no_aug_dict.keys(), no_aug_dict.values(), linestyle='-.')
        plt.plot(aug_flip_dict.keys(), aug_flip_dict.values(), linestyle='-.')
        plt.plot(aug_dict.keys(), aug_dict.values(), linestyle='-.')

    plt.xlabel("Training Epoch")
    plt.ylabel("Test Accuracy")
    plt.legend(["No Augmentation", "Horizontal Flip", "Random Crop + Horizontal Flip"])
    # plt.title(args.plot_title)

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 7
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["legend.fontsize"] = 15
    plt.rcParams["legend.handlelength"] = 2.5
    plt.rcParams["font.size"] = 15

    plt.savefig(args.save_name, bbox_inches='tight', dpi=500)
