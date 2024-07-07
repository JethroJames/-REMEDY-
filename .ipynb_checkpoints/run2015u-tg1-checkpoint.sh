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

python main_uncertainty_fusion-tg1.py