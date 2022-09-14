import matplotlib.pyplot as plt
import numpy as np


def get_data(log):
    """
    args: log
    return: dict
    description: 传入一个日志文件，返回字典数据
    """
    res = {}
    for i in log.item().keys():
        res[i] = log.item().get(i)
    return res

def try_and_plot(name, title, legend_loc, arr_dict):
    """
    args: name -> 尝试绘制的变量名
          title -> 图像的名称
          legend_loc -> 标签绘制的位置
          arr_dict -> 字典，包括了array的变量和label名
    return: none
    description: 尝试绘制，若报错，则退出
    """
    try:
        plt.grid(True)
        plt.title(title)
        for item in arr_dict:
            plt.plot(arr_dict[item][name], label=item)
        plt.legend(loc=legend_loc)
        # plt.show()
        plt.savefig("{}.svg".format(name), bbox_inches="tight", pad_inches=0.2, dpi=300)
        plt.close()
    except:
        print("Don't have {}".format(name))


if __name__ == "__main__":
    rq = get_data(np.load('./exp-rq/cache/log.npy', allow_pickle=True))
    l1 = get_data(np.load('./exp-l1/cache/log.npy', allow_pickle=True))
    l2 = get_data(np.load('./exp-l2/cache/log.npy', allow_pickle=True))
    none = get_data(np.load('./exp-none/cache/log.npy', allow_pickle=True))

    try_and_plot("loss", "Loss", "upper right", { "rq": rq, "l1": l1, "l2": l2, "none": none })
    try_and_plot("categorical_accuracy", "Accuracy", "lower right", { "rq": rq, "l1": l1, "l2": l2, "none": none })
    try_and_plot("val_loss", "Validation Loss", "upper right", { "rq": rq, "l1": l1, "l2": l2, "none": none })
    try_and_plot("val_categorical_accuracy", "Validation Accuracy", "lower right", { "rq": rq, "l1": l1, "l2": l2, "none": none })

    try_and_plot("psnr_pred", "PSNR", "lower right", { "rq": rq, "l1": l1, "l2": l2, "none": none })
    try_and_plot("ssim_pred", "SSIM", "lower right", { "rq": rq, "l1": l1, "l2": l2, "none": none })
    try_and_plot("val_psnr_pred", "Validation PSNR", "lower right", { "rq": rq, "l1": l1, "l2": l2, "none": none })
    try_and_plot("val_ssim_pred", "Validation SSIM", "lower right", { "rq": rq, "l1": l1, "l2": l2, "none": none })
