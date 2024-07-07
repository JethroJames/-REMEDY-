echo "Activating conda env..."
# 激活自己的conda,下面这行命令，开始的点不能删掉
. /mnt/data/optimal/chenmulin/miniconda3/etc/profile.d/conda.sh
# 验证自己的conda是否激活
which conda
# 激活需要的环境
cd /mnt/data/optimal/chenmulin/haojianhuang/MNER/RpBERT/
conda activate rpbert

cp cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive/include/cudnn*.h /mnt/data/optimal/chenmulin/miniconda3/envs/rpbert/include
cp cudnn-linux-x86_64-8.3.2.44_cuda11.5-archive/lib/libcudnn* /mnt/data/optimal/chenmulin/miniconda3/envs/rpbert/lib
chmod a+r /mnt/data/optimal/chenmulin/miniconda3/envs/rpbert/include/cudnn*.h
export LD_LIBRARY_PATH=/mnt/data/optimal/chenmulin/miniconda3/envs/rpbert/lib:$LD_LIBRARY_PATH
source ~/.bashrc

python main2017tg1.py --dataset twitter2017