export ALL_PROXY=socks5://127.0.0.1:1080
export http_proxy=socks5://127.0.0.1:1080
export https_proxy=socks5://127.0.0.1:1080

cd /data/clx/pycharm_projs/assignment1-basics/assignment1-basics/cs336_basics || exit
nohup python train.py > train.log &
