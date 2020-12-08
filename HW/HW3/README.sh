# DIP Homework Assignment #3
# Name: 曾俊為
# ID >: r07922106
# email:champion8599@gmail.com
echo " 開始執行.. "
get_char()
{
    SAVEDSTTY=`stty -g`
    stty -echo
    stty cbreak
    dd if=/dev/tty bs=1 count=1 2> /dev/null
    stty -raw
    stty echo
    stty $SAVEDSTTY
}
python prob1.py
python prob2.py
echo "Press any key to continue 。。。"
echo " CTRL+C break command bash ..." # 组合键 CTRL+C 终止命令!
char=`get_char`
echo " 操作完成 .... "
