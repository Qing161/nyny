import sys, os, time
import random
sys.path.append("code")
from code.data_preprocessing1 import *
from models import *
from fastapi import FastAPI, Request


app = FastAPI()

def get_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
get_random_seed(1412)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@app.post('/闻诊/算法/特征参数')
async def chuli(request: Request):
    dictc = await request.json()  # 使用 await

    s = [["少吃高脂肪以及高盐的食物，及时医院就诊",
          "少吃刺激性食物和兴奋类药物，及时医院就诊",
          "选择比较温和的运动，及时医院就诊"],
         ["检测正常，若确有不适，及时医院就诊",
          "检测正常，可进一步参考中医四诊软件",
          "检测正常，可正常工作生活，加强锻炼"],
         ["平时不能劳累，注意休息，及时医院就诊",
          "低盐饮食，适当活动，及时医院就诊",
          "注意保持心情平静，及时医院就诊"],
         ["检测正常，若确有不适，及时医院就诊",
          "检测正常，可进一步参考中医四诊软件",
          "检测正常，可正常工作生活，加强锻炼"],
         ["尽量不要饮酒，抽烟，及时医院就诊",
          "避免过度的体力劳动和剧烈运动，及时医院就诊",
          "若无症状，可正常生活工作，定期检查，及时医院就诊"]]

    test_inputs, t1, draw_test_dict = testdata_preprocessing(dictc)
    try:
        test_inputs_list = test_inputs.tolist()
    except:
        test_inputs_list = None


    test_inputs = torch.tensor(test_inputs)
    t1 = torch.tensor(t1)
    # print(type(test_inputs), type(t1))

    if (test_inputs is None) or (t1 is None):
        print('waiting: 数据采集未完成')
        time.sleep(5)
    else:
        device = torch.device('cpu')
        in_channels = 1
        n_classes = 5
        net = FCNNet(in_channels, n_classes)
        PATH = "./net.pt"
        net.load_state_dict(torch.load(PATH, map_location='cpu'))
        net = net.to(device)
        net.eval()
        test_pred = net(test_inputs.to(device)).argmax(dim=1).cpu()
        test_pred = np.array(test_pred.detach()).tolist()
        label_dict = {0: "aortic stenosis(主动脉狭窄)", 1: "normal(正常)", 2: "normal(正常)", 3: "normal(正常)",
                      4: "murmur exist in the systole interval(收缩期存在杂音)"}
        # number_to_label = [label_dict[x] if x in label_dict else x for x in test_pred]
        # print(type(test_pred),test_pred)
        number_to_label=label_dict[test_pred[0]]

        # print(type(draw_test_dict))

        draw_test_dict['predictions']=number_to_label

        print("本次检测完成")

        return draw_test_dict


@app.post('/闻诊/算法/预处理')
async def shujv(request: Request):
    dictc = await request.json()
    # print(dictc)
    return dictc



if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=5001)