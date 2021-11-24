#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

typedef double DB;
typedef long long LL;

const DB eps = 1e-6;
const DB errorRate = 0.0001;
const DB INF = 1000000000000000000;

//随机[l,r]的数
DB rand_number(DB l,DB r)
{
    if(fabs(r - l) < eps)
        return l;
    int len = 1e9;
    int t = rand() * rand() % len;
    return t * 1.0 / 1e9 * (r - l) + l;
}

//神经元
struct  neuron{
    DB in, out;                             //输入输出
};

//神经层
struct layer{
    vector<neuron> node;                    //每层由神经元组成
    int actType;                            //激活函数类型，每层认为是一样的
};

typedef vector<layer> netWork;              //神经网络
typedef vector<vector<DB>> weightMatrix;    //单层权重矩阵
typedef vector<weightMatrix> netWeight;     //神经网络权重矩阵
typedef vector<DB> layerBias;               //单层偏置
typedef vector<layerBias> netbias;          //神经网络偏置

struct bpAlgorithm {
    DB alpha = 0.01;                        //leaky Relu 参数

    DB learnRate;                           //学习率
    DB sumError;                            //总误差
    int maxIter;                            //最大迭代次数
    int cntLayer;                           //隐含层数量
    netWork bpNet;                          //网络对象
    netWeight bpWeight;                     //当前版本
    netbias bpBias;
    netWeight newWeight;                    //下一版本
    netbias newBias;                        
    netWeight oldWeight;                    //上一版本，用于回滚
    netbias oldBias;

    /*
        n               输入维数
        m               数据组数
        setMaxIter      最大迭代次数
        setCntLayer     隐含层数，一般为1,2
        actType         激活函数类型
        setLearnRate    学习率
    */
    bpAlgorithm(int n, int m, int setMaxIter = 100, int setCntLayer = 1, int actType = 1, DB setLearnRate = 0.1){
        maxIter = setMaxIter;
        cntLayer = setCntLayer;
        learnRate = setLearnRate;

        //输入层
        {
            layer tmplayer;
            for (int i = 1; i <= n; ++i){
                neuron tn;
                tn.in = tn.out = 0;
                tmplayer.node.push_back(tn);
            }
            bpNet.push_back(tmplayer);
        }

        //隐含层
        int nd = sqrt(n + m) + rand_number(0, 8) + 2;       //每层节点个数设定
        for (int i = 1; i <= cntLayer; ++i){
            layer tmplayer;
            for (int j = 1; j <= nd; ++j){
                neuron tn;
                tmplayer.node.push_back(tn);
            }
            tmplayer.actType = actType;
            bpNet.push_back(tmplayer);
        }

        //输出层
        {
            layer tmplayer;
            neuron tn;
            tmplayer.actType = 0;
            tmplayer.node.push_back(tn);
            bpNet.push_back(tmplayer);
        }

        //构造随机W矩阵
        for (int i = 0; i < bpNet.size() - 1; ++i){
            weightMatrix tmpw;
            for (int j = 0; j < bpNet[i].node.size(); ++j){
                vector<DB> tmpv;
                for (int k = 0; k < bpNet[i + 1].node.size(); ++k)
                    tmpv.push_back(rand_number(0, 1));
                tmpw.push_back(tmpv);
            }
            bpWeight.push_back(tmpw);
        }

        //构造随机偏置
        for (int i = 0; i < bpNet.size(); ++i){
            layerBias tmpb;
            for (int j = 0; j < bpNet[i].node.size(); ++j)
                tmpb.push_back(rand_number(0, 1));
            bpBias.push_back(tmpb);
        }
    }

    /*
        type:
            0   f(x) = x
            1   sigmoid
            2   tanh
            3   relu
            4   leaky relu
        flag = 1, 返回f(x)
        flag = 0, 返回偏导值
    */
    DB get_value(DB x, int type, int flag)
    {
        if(flag){
            if(type == 0)
                return x;
            if(type == 1)
                return 1.0 / (1 + exp(-x));
            if(type == 2)
                return (exp(2 * x) - 1) / (exp(2 * x) + 1);
            if(type == 3)
                return max(x, 0.0);
            if(type == 4)
                return max(alpha * x, x);
        }   
        else{
            if(type == 0)
                return 1;
            if(type == 1)
                return x * (1 - x);
            if(type == 2)
                return 1 - x * x;
            if(type == 3)
                return x <= 0 ? 0 : 1;
            if(type == 4)
                return x <= 0 ? alpha : 1;
        }
    }

    //前向传播
    void forward_propagation()
    {
        for (int i = 0; i < bpNet.size() - 1; ++i){
            for (int k = 0; k < bpNet[i + 1].node.size(); ++k){
                bpNet[i + 1].node[k].in = bpBias[i + 1][k];         //bpbias[i][j]表示第i层第j个节点从上层获取的偏置
                for (int j = 0; j < bpNet[i].node.size(); ++j){
                    bpNet[i + 1].node[k].in += bpNet[i].node[j].out * bpWeight[i][j][k];
                }

                bpNet[i + 1].node[k].out = get_value(bpNet[i + 1].node[k].in, bpNet[i + 1].actType, 1);
                
            }
        }

    }

    //反向传播
    void back_propagation(DB y)
    {
        vector<DB> resItem;

        //输出层残差
        {
            DB t = bpNet.back().node[0].out - y;
            resItem.push_back(t * get_value(bpNet.back().node[0].out, bpNet.back().actType, 0));
        }
        
        //计算weight梯度和残差项
        for (int i = bpNet.size() - 1; i >= 1; --i){
            //计算i - 1 -> i新的weight
            for (int j = 0; j < bpNet[i - 1].node.size(); ++j){
                for (int k = 0; k < bpNet[i].node.size(); ++k)
                    newWeight[i - 1][j][k] -= learnRate * resItem[k] * bpNet[i - 1].node[j].out;   
            }

            //计算i - 1 -> i新的bias
            for (int j = 0; j < bpNet[i].node.size(); ++j){
                newBias[i][j] -= learnRate * resItem[j];
            }

            //计算i - 1层的残差项
            vector<DB> newresItem;
            for (int j = 0; j < bpNet[i - 1].node.size(); ++j){
                DB t = 0;
                for (int k = 0; k < bpNet[i].node.size(); ++k)
                    t += bpWeight[i - 1][j][k] * resItem[k];

                newresItem.push_back(t * get_value(bpNet[i - 1].node[j].out, bpNet[i - 1].actType, 0));
            }
            resItem = newresItem;

        }
    }

    void train(vector<vector<DB>>& X_train, vector<DB> y_train)
    {
        oldWeight = bpWeight;
        oldBias = bpBias;
        DB lastSumError = INF;
        for (int round = 1; round <= maxIter; ++round){
            //每次迭代的初始化，复制相关数组
            sumError = 0.0;
            newWeight = bpWeight;
            newBias = bpBias;
            for (int id = 0; id < X_train.size(); ++id){
                
                //数据有误
                if(bpNet[0].node.size() != X_train[id].size()){
                    cout << "error data format";
                    exit(1);
                }
                
                //放入信息到输入神经元
                for (int i = 0; i < bpNet[0].node.size(); ++i)
                    bpNet[0].node[i].out = X_train[id][i];

                //前向传播
                forward_propagation();

                //计算误差和
                DB t = bpNet.back().node[0].out - y_train[id];
                sumError += fabs(t);
                //反向传播
                back_propagation(y_train[id]);
            }

            //已经收敛，退出迭代
            sumError /= X_train.size();
            if(sumError < errorRate)
                break;

            //学习率过大
            if(sumError >= lastSumError){
                learnRate *= 0.1;
                bpWeight = oldWeight;
                bpBias = oldBias;
            }
            else{
                //更新W矩阵和bias
                oldWeight = bpWeight;
                oldBias = bpBias;
                bpWeight = newWeight;
                bpBias = newBias;
                lastSumError = sumError;
            }
            
        }
    }

    void print_result()
    {
        printf("error sum : %.10lf\n", sumError);
        for (int i = 0; i < bpNet.size() - 1;++i){
            cout << "layer : " << i << endl;
            cout << "weightMatrix : " << endl;
            for (int j = 0; j < bpNet[i].node.size(); ++j){
                for (int k = 0; k < bpNet[i + 1].node.size(); ++k)
                    printf("%.10lf ", bpWeight[i][j][k]);
                cout << endl;
            }

            cout << "bias : " << endl;
            for (int j = 0; j < bpBias[i + 1].size(); ++j)
                printf("%.10lf ", bpBias[i + 1][j]);
            cout << endl
                    << endl
                    << endl;
        }
    }
};

int main()
{
    //freopen("ls.csv", "r", stdin);
    int T = 0, n = 0;
    cout << "number of test" << endl;
    cin >> T;
    cout << "dimension of X" << endl;
    cin >> n;

    bpAlgorithm bp(n, T, 200, 1, 1, 0.01);

    vector<vector<DB>> X_train;
    vector<DB> y_train;
    for(int rd = 0;rd < T; ++rd){
        vector<DB> tmpx;
        for (int i = 0; i < n;++i){
            tmpx.push_back(0);
            cin >> tmpx[i];
        }
        X_train.push_back(tmpx);
        y_train.push_back(0);
        cin >> y_train[rd];
    }

    bp.train(X_train, y_train);
    bp.print_result();
    return 0;
}