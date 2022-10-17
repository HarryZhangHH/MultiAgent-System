import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
import pandas as pd


class Node():
    node_count = 0
    def __init__(self, reward=0, depth=0, left=None, right=None, address=[]):
        self.reward = reward
        self.ucb_value = np.inf
        self.counts = 0
        self.depth = depth
        self.right = right
        self.left = left
        self.address = address
        self.r = 0
        Node.node_count += 1

    def __str__(self):
        return f"Node: Depth = {self.depth} - {self.address}"
#         return f"({self.reward}) - UCB: {self.ucb_value_init}"

class Binary_tree():
    def __init__(self, depth=12, root=None, c_value=0.5):
        self.node_count = 0
        self.depth = depth
        self.root = Node(0)
        self.c = c_value
        self.leaf_count = 0
        self.leaf_nodes = []
        self.leaf_values = []
        self.mcts_result = None
        self.A_t = []
#         print(f"node: {self.node_count} ({self.root.reward})")
        
    def tree(self, node, depth=0):
        if depth < self.depth:
            if node.left is None:
                self.node_count += 1
                A = node.address.copy()
                A.append("L")
                node.left = Node(depth=depth+1,address=A)
                if depth + 1 == self.depth:
                    self.leaf_count += 1
                    self.leaf_nodes.append(node.left)
                self.tree(node.left, depth+1)
            if node.right is None:
                A = node.address.copy()
                A.append("R")
                node.right = Node(depth=depth+1,address=A)
                if depth + 1 == self.depth:
                    self.leaf_count += 1
                    self.leaf_nodes.append(node.right)
                self.tree(node.right, depth+1)
        return node
    
    # assign reward to each leaf
    def assign_values(self, A_t, B=23, tao=5):
        self.A_t = A_t
        for leaf in self.leaf_nodes:
            x_i = self.leaf_reward(leaf)
            self.leaf_values.append(x_i)
    
    def leaf_reward(self, node, B=23, tao=5):
        A_i = node.address
        d_i = edit_distance(A_i, self.A_t)
        x_i = B*np.exp(-d_i/tao)
        node.reward = x_i
        return x_i
            
        
    def MCTS(self, root):
        best_node = self.select(root)
        return best_node
    
    def select(self, root):
        iter = 0
        while not self.is_leaf(root):
            iter += 1
            print('iterate:', iter)
            for i in range(23):
                root, _ = self.snow(root)
            
            root.left = self.UCB(root.left, root.counts)
            root.right = self.UCB(root.right, root.counts)
            print(f"Reward: {root.reward}, Count:{root.counts}, Left_UCB:{root.left.ucb_value}, Right_UCB:{root.right.ucb_value}")
            if root.left.ucb_value > root.right.ucb_value:
                print("left")
                root = root.left
            else:
                print("right")
                root = root.right
        return root
    
    def snow(self, node, parent_node=None, expanded=False):
        node.counts += 1
        
        # check whether is leaf node
        if self.is_leaf(node):
            node.r += node.reward.copy()
            if self.mcts_result is None or self.mcts_result.r < node.reward:
                self.mcts_result = node
            print(f"{node} BEST REWARD ---> {node.reward} <--- (all time champion: {self.mcts_result.reward})")
            return node, node.reward.copy()
        
        if node.left.counts != 0:
            node.left = self.UCB(node.left, node.counts)
        if node.right.counts != 0:
            node.right = self.UCB(node.right, node.counts)
                
        if not expanded:
            if node.left.ucb_value > node.right.ucb_value:
                _ , leaf_reward = self.snow(node.left, parent_node=node)
            elif node.left.ucb_value < node.right.ucb_value:
                _ , leaf_reward = self.snow(node.right, parent_node=node)
            # if both are unexplored or equal, random choice
            elif node.left.ucb_value == node.right.ucb_value:
                _ , leaf_reward = self.snow(self.random_select(node), parent_node=node, expanded=True)
        
        if expanded:
            leaf_reward = self.roll_out(node)
        node = self.back_up(node, leaf_reward)
#         self.print_node(node, parent_node)
        return node, leaf_reward
    
    def random_select(self, node):
        if np.random.randint(2) == 0:
            return node.left
        else:
            return node.right
    
    def roll_out(self, node):
        while node.left is not None and node.right is not None:
            if np.random.randint(2) == 0:
                node = node.left
            else:
                node = node.right
        if self.is_leaf(node):
            return node.reward
        else:
            return 0

    def back_up(self, node, leaf_reward):
        node.r += leaf_reward
        return node

    def UCB(self, node, parent_counts):
        c = self.c
        n_i = node.counts
        N = parent_counts
        x_i = node.r / n_i

        # UCB1
        node.ucb_value = x_i + c * (np.sqrt(np.log(N) / n_i))
        return node
    
    def is_leaf(self, node):
        if node.left is None and node.right is None:
            if node.reward == 0:
                _ = self.leaf_reward(node)
            return True
        else:
            return False

    def print_node(self, node, parent_node):
        print(f"{node}, Node reward: {node.r}, Node counts: {node.counts}")
        if parent_node is not None:
            print(f"Parent Node:{parent_node}, Parent Node reward: {parent_node.r}")
    
def edit_distance(s1, s2):
    m=len(s1)+1
    n=len(s2)+1

    tbl = {}
    for i in range(m): tbl[i,0]=i
    for j in range(n): tbl[0,j]=j
    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            tbl[i,j] = min(tbl[i, j-1]+1, tbl[i-1, j]+1, tbl[i-1, j-1]+cost)
    return tbl[i,j]

def average(lst):
    return sum(lst) / len(lst)


def visualizing_rewards(data):
    
    plt.figure(figsize=(16, 10))
    sns.barplot(x='c', y='value', data=data, palette="Blues_d")
    plt.ylim(0, 25)
    plt.ylabel('Reward', fontsize = 23)
    plt.xlabel('c-value', fontsize = 23)
    plt.legend()
    plt.savefig('c_vals.png')
    plt.show()

def plot_hist(c):
    l = len(c)
    pecent = int(l * 0.05)

    plt.figure()
    sns.distplot(c)
#     plt.xlim(0, 100)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('dist_plot.png')


def main():
    depth = 12
    iterations = 100 # number of iterations
    c_list = [0, 0.05, 0.1, 0.5, 1, 1.4, 2, 5, 10]
    results = {}
    for c in c_list:
        results_c = []
        for i in range(iterations):
            print(f"---START! c: {c} --- iteration：{i}")
            Node.node_count = 0
            env = Binary_tree(depth=depth, c_value=c)
            tree = env.tree(env.root)
            print(tree.node_count)
            leaf_len = len(env.leaf_nodes)
            t = np.random.randint(0,leaf_len)
            A_t = env.leaf_nodes[t].address
            print(A_t)
            env.assign_values(A_t)
            # print(env.leaf_values)
            tree = env.MCTS(tree)
            print(tree.reward)
            results_c.append(tree.reward)
        results[c] = results_c
        print(results)

    c = 1
    env = Binary_tree(depth=depth, c_value=c)
    tree = env.tree(env.root)
    print(tree.node_count)
    leaf_len = len(env.leaf_nodes)
    t = np.random.randint(0,leaf_len)
    A_t = env.leaf_nodes[t].address
    print(A_t)
    env.assign_values(A_t)
    print(env.leaf_values)

    avg_list = []
    max_list = []
    min_list = []
    for re in results:
        x =  [i for i in range(23)]
        y = results[re]
        avg_list.append(average(y))
        max_list.append(max(y))
        min_list.append(min(y))
    #     plt.plot(x, y)
    print(avg_list)
    print(max_list)
    print(min_list)

    # visualizing_rewards(results[5])
    c_5 = [5]*iterations
    c_2 = [2]*iterations
    c_1 = [1]*iterations
    c_1_4 = [1.4]*iterations
    df = pd.DataFrame(list(zip((results[1]+results[1.4]+results[2]+results[5]),(c_1 + c_1_4 + c_2 + c_5))), columns=['value','c'])
    print(df)
    visualizing_rewards(df)

    plot_hist(results[5])

if __name__ == "__main__":
    main()