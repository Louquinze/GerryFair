from gerryfair.util import creat_big_pareto
import argparse

communities_dataset = "./dataset/communities.csv"  # 0
communities_attributes = "./dataset/communities_protected.csv"
lawschool_dataset = "./dataset/lawschool.csv"  # 1
lawschool_attributes = "./dataset/lawschool_protected.csv"
adult_dataset = "./dataset/adult.csv"  # 3
adult_attributes = "./dataset/adult_protected.csv"
student_dataset = "./dataset/student-mat.csv"  # 5
student_attributes = "./dataset/student_protected.csv"

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset', type=int, default=0)
parser.add_argument('--regressor', type= str, default=["linear", "random-forest", "gradient-boost", "mlp"], nargs='+')
parser.add_argument('--gamma', type=float, default=.01)
parser.add_argument('--max_iters', type=int, default=200)

if __name__ == '__main__':
    args = parser.parse_args()
    tag = f"{args.dataset}_{args.gamma}_{args.max_iters}"  # {args.regressor}_

    creat_big_pareto(tag=tag, include=args.regressor)
