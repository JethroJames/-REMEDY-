echo "Activating conda env..."
# 激活自己的conda,下面这行命令，开始的点不能删掉
. /mnt/data/optimal/chenmulin/miniconda3/etc/profile.d/conda.sh
# 验证自己的conda是否激活
which conda
# 激活需要的环境
cd /mnt/data/optimal/chenmulin/haojianhuang/MNER/RpBERT/
conda activate rpbert

chmod a+r /mnt/data/optimal/chenmulin/miniconda3/envs/rpbert/include/cudnn*.h
export LD_LIBRARY_PATH=/mnt/data/optimal/chenmulin/miniconda3/envs/rpbert/lib:$LD_LIBRARY_PATH
source ~/.bashrc

mkdir -p results

lamda1_values=(0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5)
lamda2_values=(0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5)


for lamda1 in "${lamda1_values[@]}"; do
    echo "正在运行实验，lamda1=$lamda1 和 lamda2=1"
    python main_uncertainty_fusion-lamda.py --lamda1 $lamda1 --lamda2 1 --dataset twitter2017 > "results/report2017_lamda1_${lamda1}_lamda2_1.txt"
done


for lamda2 in "${lamda2_values[@]}"; do
    echo "正在运行实验，lamda1=1 和 lamda2=$lamda2"
    python main_uncertainty_fusion-lamda.py --lamda1 1 --lamda2 $lamda2 --dataset twitter2017 > "results/report_lamda1_1_lamda2_${lamda2}.txt"
done
