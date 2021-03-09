from collections import Counter
import pandas as pd

def gini(nums):
    """
    计算gini指数
    :param nums:
    :return: gini指数
    """
    d = Counter(nums)
    prob = [v/len(nums) for k, v in d.items()]
    return 1-sum([p**2 for p in prob])

df = pd.DataFrame()

def split_df(data, col):
    """
    根据特征值划分df数据
    :param data:
    :param col:
    :return:
    """
    unique_values = data[col].unique()
    result_dict = {}
    for value in unique_values:
        result_dict[value] = df[df[col] == value]
    return result_dict

def choose_best_col(df, label):
    """
    选择最佳划分特征
    :param df:
    :param label:
    :return:
    """
    # 计算初始的gini指数
    #gini_d = gini(df[label].tolist())
    # 特征范围
    cols = [col for col in df.columns if col not in [label]]
    min_value = 999
    best_col = None
    min_splited = None

    # 遍历每个特征
    for col in cols:
        # 依据特征划分数据集
        split_set = split_df(df, col)
        gini_da = 0
        # 计算划分后的gini指数
        for keys, values in split_set.items():
            gini_di = gini(values[label].tolist())
            gini_da += len(values)/len(df)*gini_di
        # 更新最优gini指数
        if gini_da < min_value:
            min_value = gini_da
            best_col = col
            min_splited = split_set
    return min_value, best_col, min_splited


#构建CART
class Cart():
    class Node():
        def __init__(self, name):
            self.name = name
            self.connections = {}

        def connect(self, label, node):
            self.connections[label] = node

    def __init__(self, data, label):
        self.columns = data.columns
        self.data = data
        self.label = label
        self.root = self.Node("root")

    def construct_tree(self):
        """
        生成Cart
        :return:
        """
        self.construct(self.root, "", self.data, self.label)

    def construct(self, parent_node, parent_connection_label, input_data, columns):
        """
        递归构建CART
        """
        min_value, best_col, min_splited = choose_best_col(input_data[columns], self.label)
        # 无最优划分(为叶子节点)
        if not best_col:
            node = self.Node(input_data[self.label].iloc[0])
            parent_node.connect(parent_connection_label,node)
            # 结束递归
        # 找到最优划分特征
        node = self.Node(best_col)
        parent_node.connect(parent_connection_label, node)

        # 删除最优的划分特征得到剩下的特征
        new_columns = [col for col in columns if col != best_col]
        # 递归的构建决策树
        for key, value in min_splited.items():
            self.construct(node, key, value, new_columns)




df = pd.read_csv("example_data.csv", dtype={"windy":"str"})


