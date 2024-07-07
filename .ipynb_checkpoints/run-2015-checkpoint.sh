echo "Activating conda env..."
# 激活自己的conda,下面这行命令，开始的点不能删掉
. /mnt/data/optimal/chenmulin/miniconda3/etc/profile.d/conda.sh
# 验证自己的conda是否激活
which conda
# 激活需要的环境
conda activate rpbert
cd /mnt/data/optimal/chenmulin/haojianhuang/MNER/RpBERT/
python main.py