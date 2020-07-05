# coding:utf-8
import os
import random
'''
    为数据集生成对应的txt文件
'''


def gen_txt(img_dir):
    # f = open(txt_path, 'w')
    line = []
    for root, s_dirs, _ in os.walk(img_dir, topdown=True):  # 获取 train文件下各文件夹名称
        index = 0
        for sub_dir in s_dirs:
            i_dir = os.path.join(root, sub_dir)  # 获取各类的文件夹 绝对路径
            img_list = os.listdir(i_dir)  # 获取类别文件夹下所有png图片的路径
            for i in range(len(img_list)):
                if not img_list[i].endswith('png'):  # 若不是png文件，跳过
                    continue
                label = str(index)
                img_path = os.path.join(i_dir, img_list[i])
                line.append((img_path + ' ' + label))
                # f.write(line)
    # len1 = len(line)
    print(len(line))
    # random.shuffle(line)
    # f = open('train_data.txt', 'w')
    # for lines in line[:int(len1*0.8)]:
    #     line1 = lines + '\n'
    #     f.write(line1)
    # f.close()
    # f1 = open('test_data.txt', 'w')
    # for lines in line[int(len1*0.8):]:
    #     line1 = lines + '\n'
    #     f1.write(line1)
    # f1.close()
    f1 = open('6dbfan2_data.txt', 'w')
    for lines in line[:]:
        line1 = lines + '\n'
        f1.write(line1)
    f1.close()


if __name__ == '__main__':
    gen_txt('E:\\6db2png')
