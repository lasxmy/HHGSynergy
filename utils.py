import math
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, mean_squared_error, r2_score, roc_curve
from scipy.stats import pearsonr
from rdkit import DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem import AllChem as Chem
import random
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold


def uniform(size, tensor):
    if tensor is not None:
        bound = 1.0 / math.sqrt(size)
        tensor.data.uniform_(-bound, bound)


def kaiming_uniform(tensor, fan, a):
    if tensor is not None:
        bound = math.sqrt(6 / ((1 + a ** 2) * fan))
        tensor.data.uniform_(-bound, bound)


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def glorot_orthogonal(tensor, scale):
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)


def normal(tensor, mean, std):
    if tensor is not None:
        tensor.data.normal_(mean, std)


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


def metrics_graph(yt, yp):
    precision, recall, _, = precision_recall_curve(yt, yp)
    aupr = -np.trapz(precision, recall)
    auc = roc_auc_score(yt, yp)
    # ---f1,acc,recall, specificity, precision
    real_score = np.mat(yt)
    predict_score = np.mat(yp)
    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]
    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN
    tpr = TP / (TP + FN)
    fpr = FP / (FP + TN)
    recall_list = tpr
    precision_list = TP / (TP + FP)
    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)
    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return auc, aupr, f1_score[0, 0], accuracy[0, 0]  # , recall[0, 0], specificity[0, 0], precision[0, 0]


class FP:
    """
    Molecular fingerprint class, useful to pack features in pandas df
    Parameters
    ----------
    fp : np.array
        Features stored in numpy array
    names : list, np.array
        Names of the features
    """

    def __init__(self, fp, names):
        self.fp = fp
        self.names = names

    def __str__(self):
        return "%d bit FP" % len(self.fp)

    def __len__(self):
        return len(self.fp)


def get_cfps(mol, radius=2, nBits=256, useFeatures=False, counts=False, dtype=np.float32):
    """Calculates circural (Morgan) fingerprint.
    http://rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
    radius : float
        Fingerprint radius, default 2
    nBits : int
        Length of hashed fingerprint (without descriptors), default 1024
    useFeatures : bool
        To get feature fingerprints (FCFP) instead of normal ones (ECFP), defaults to False
    counts : bool
        If set to true it returns for each bit number of appearances of each substructure (counts). Defaults to false (fingerprint is binary)
    dtype : np.dtype
        Numpy data type for the array. Defaults to np.float32 because it is the default dtype for scikit-learn
    Returns
    -------
    ML.FP
        Fingerprint (feature) object
    """
    arr = np.zeros((1,), dtype)

    if counts is True:
        info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, useFeatures=useFeatures,
                                                   bitInfo=info)
        DataStructs.ConvertToNumpyArray(fp, arr)
        arr = np.array([len(info[x]) if x in info else 0 for x in range(nBits)], dtype)
    else:
        DataStructs.ConvertToNumpyArray(
            AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, useFeatures=useFeatures), arr)
    return FP(arr, range(nBits))


def get_fingerprint_from_smiles(smiles):
    m = Chem.MolFromSmiles(smiles)
    Finger = get_cfps(m)
    fp = Finger.fp
    fp = fp.tolist()
    return fp


def get_MACCS(smiles):
    m = Chem.MolFromSmiles(smiles)
    arr = np.zeros((1,), np.float32)
    fp = MACCSkeys.GenMACCSKeys(m)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def set_seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def regression_metric(ytrue, ypred):
    rmse = mean_squared_error(y_true=ytrue, y_pred=ypred, squared=False)
    r2 = r2_score(y_true=ytrue, y_pred=ypred)
    r, p = pearsonr(ytrue, ypred)
    return rmse, r2, r

def draw_bar(data):
    count_dict = Counter(data)

    # 按数字升序排序
    sorted_numbers = sorted(count_dict.keys())
    sorted_counts = [count_dict[num] for num in sorted_numbers]

    # 配置图形样式
    # plt.figure(figsize=(12, 6))
    plt.figure(figsize=(12, 6))
    bars = plt.bar(sorted_numbers, sorted_counts, color='skyblue')

    # 添加标注和标签
    # plt.xlabel('number of cell line', fontsize=12, fontweight='bold')
    # plt.ylabel('number of drug combination', fontsize=12, fontweight='bold')
    plt.xlabel('number of drug', fontsize=12, fontweight='bold')
    plt.ylabel('number of drug-cell line combination', fontsize=12, fontweight='bold')
    
    # plt.title('数值分布柱状图', fontsize=14)
    # plt.xticks(sorted_numbers)
    
    plt.xticks([i for i in range(0, max(sorted_numbers)+1, 2)])
    
    # plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 在柱子上方显示具体数值
    # for bar in bars:
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width()/2., height,
    #              f'{height}',
    #              ha='center', va='bottom')

    plt.tight_layout()

    # 保存图片到文件（默认PNG格式）
    output_file = "number_distribution.png"  # 可以自定义文件名
    plt.savefig(output_file, dpi=300, bbox_inches='tight')  # dpi=300提高清晰度，bbox_inches='tight'避免裁剪

    print(f"图表已保存为: {output_file}")
    
def draw(X, y):
    X = manifold.TSNE(n_components=2, init='pca', random_state=42).fit_transform(X)
    
    print(X.shape)
    print(y.shape)
    # X = np.vstack([data_neg, data_mid, data_pos])
    # y = np.array([0]*(n_samples//3) + [1]*(n_samples//3) + [2]*(n_samples//3+1))

    # 2. t-SNE降维（如果原始数据是高维的）
    # tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    # X_tsne = tsne.fit_transform(原始高维数据)
    # 此处直接使用生成的2D数据模拟结果

    # 3. 绘制t-SNE图
    plt.figure(figsize=(8, 8))

    # 自定义标记符号和颜色
    markers = ['o', 's']  # 对应三种synergy范围
    colors = ['#FF6B6B', '#4ECDC4']  # 红/青绿/灰蓝
    labels = ['synergy < 0', 'synergy > 30']

    # 绘制每个类别
    for i in range(2):
        plt.scatter(X[y==i, 0], X[y==i, 1],
                   marker=markers[i],
                   s=50,           # 点大小
                   c=colors[i],
                   alpha=0.7,
                   edgecolors='w',
                   linewidths=0.5,
                   label=labels[i])
        
    plt.tick_params(
        axis='both',          # 同时作用于x和y轴
        which='both',        # 同时控制主副刻度
        bottom=False,        # 隐藏x轴底部刻度
        left=False,          # 隐藏y轴左侧刻度
        labelbottom=False,   # 隐藏x轴底部刻度标签
        labelleft=False      # 隐藏y轴左侧刻度标签
    )

    # 4. 设置坐标轴（与图片一致）
    # plt.ylim(-90, 120)  # 纵轴范围
    # plt.yticks([-90, -60, -30, 0, 30, 60, 90, 120])  # 纵轴刻度
    # plt.xlim(-25, 27)
    # plt.ylim(-20, 25)
    plt.xlim(-24, 25)
    plt.ylim(-25, 26)
    # plt.xlim(-27, 25)
    # plt.ylim(-32, 28)
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)  # 纵轴标签
    plt.title('t-SNE Visualization with Synergy Groups', fontsize=14)

    # 5. 添加图例和网格
    plt.legend(loc='upper right', fontsize=10)
    # plt.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    # 保存图片到文件（默认PNG格式）
    output_file = "number_distribution1.png"  # 可以自定义文件名
    plt.savefig(output_file, dpi=300, bbox_inches='tight')  # dpi=300提高清晰度，bbox_inches='tight'避免裁剪

    print(f"图表已保存为: {output_file}")
    a=n
    
