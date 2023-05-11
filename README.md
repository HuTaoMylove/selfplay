# selfplay
蒟蒻的毕业设计
基于gfootball的观测层级化自博弈，根据测试的结果不同的层级数分别取得了接近和超越预训练的结果。
## 安装
```shell
# create python env
conda create -n sp python=3.8
# install dependency
pip install Gfootball
```
2vs2修改了Gfootball原有的5vs5场景球员配置和观察空间，设置了5个测试场景。
需要：
- `将eval_scen中的wrappers.py替换conda路径下的gfootball\env`
- `将eval_scen中其他的py文件放置到conda路径下的gfootball\scenarios`
##使用方法
```shell
python main.py --selfplay-algorithm ["hsp_2","hsp_3","fsp", "sp", 'rsp']
```
实验结果见log文件夹
```shell
tensorboard --logdir=log/
```