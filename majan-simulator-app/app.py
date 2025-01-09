from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import numpy as np
import cv2
import os
import mss
import pygetwindow as gw
from pathlib import Path
import base64
import asyncio

import json
from pprint import pprint

import requests

from mahjong import *

#######################
        # 関数
#######################

# 手牌、自模牌、捨て牌、ドラ牌の画像の切り抜き関数
def cropMyHandImage(jantamaMainImage):
    height, width = jantamaMainImage.shape[:2]
    myHandLeft = int(width*203/1665)
    myHandRight = int(width*1255/1665)
    myHandTop = int(height*810/938)
    myHandBottom = int(height*920/938)
    tsumoLeft = int(width*1278/1665)
    tsumoRight = int(width*1359/1665)

    # sutehaiLeft = int(width*659/1665)
    # sutehaiRight = int(width*1028/1665)
    # sutehaiTop = int(height*481/938)
    # sutehaiBottom = int(height*640/938)

    doraLeft = int(width*34/1665)
    doraRight = int(width*273/1665)
    doraTop = int(height*72/938)
    doraBottom = int(height*127/938)

    myHandImage = jantamaMainImage[myHandTop:myHandBottom, myHandLeft:myHandRight]
    myTsumoImage = jantamaMainImage[myHandTop:myHandBottom, tsumoLeft:tsumoRight]
    doraImage = jantamaMainImage[doraTop:doraBottom, doraLeft:doraRight]
    # sutehaiImage =  jantamaMainImage[sutehaiTop:sutehaiBottom, sutehaiLeft:sutehaiRight]

    myHandImage = cv2.resize(myHandImage, dsize = (1068,131))
    myTsumoImage = cv2.resize(myTsumoImage, dsize = (81,131))
    # sutehaiImage = cv2.resize(myTsumoImage, dsize = (486,393))

    return [myHandImage, myTsumoImage, doraImage] # , sutehaiImage

# Base64エンコード化関数
def encode_image_to_base64(image):
    # 画像をJPEG形式でエンコード
    _, buffer = cv2.imencode('.jpg', image)
    # バイナリデータをBase64エンコード
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return encoded_image

# 手牌の切り分け関数
def divideMyHandImage(myHandImage):
    myHandImageList = []
    for i in range(2,1068,82):
        myHandImageList.append(myHandImage[:,i:i+81])
    return myHandImageList

# ドラ牌の切り分け関数
def divideDoraImage(doraImage):
    dora_w = 64
    doraImageList = []
    for i in range(4):
        doraImage_resize = cv2.resize(doraImage[:,i + dora_w * i:dora_w * (i + 1)], dsize = (81,131))
        doraImageList.append(doraImage_resize)
    return doraImageList

# 捨て牌の切り分け関数 vol.2で実装予定
# def recog_sutehai(sutehai_img, pai_list_image):

#     # 移動点を指定して射影変換用の変換行列を作成
#     h, w = sutehai_img.shape[:2]
#     src_pts = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float32)
#     dst_pts = np.array([[-22, -30], [0, h], [w, h], [w+45, -30]], dtype=np.float32)

#     mat = cv2.getPerspectiveTransform(src_pts, dst_pts)

#     # 射影変換で鳴き牌画像を台形補正
#     sutehai_img = cv2.warpPerspective(sutehai_img, mat, (w, h))

#     # 切り出す捨て牌の大きさ
#     pai_h, pai_w = [82, 75]

#     # 右隣、下の牌との距離
#     left_diff = 4
#     bottom_diff = 2
    
#     # 捨て牌情報用リスト
#     sutehai_info_list = []
    
#     # 3×6+1に区切って認識する
#     for i in range(3):
#         for j in range(7):
#             if i < 2 and j == 6:
#                 continue
#             # 捨て牌1枚部分を切り出し
#             sutehai_image = sutehai_img[(pai_h + bottom_diff)*i:(pai_h + bottom_diff)*i + pai_h, (pai_w + left_diff)*j:(pai_w + left_diff)*j + pai_w]
#             # テンプレートマッチングに合うようにリサイズ
#             sutehai_image = cv2.resize(sutehai_image, dsize = (66, 99))
#             cv2.imwrite("./static/data/jantama_capture/sutehai_{}_{}.png".format(i,j),sutehai_image)
#             # 牌の種類を認識
#             sutehai_info = recogPaiImage(sutehai_image, pai_list_image, 0.2)
#             if sutehai_info:
#                 sutehai = sutehai_info
#             else:
#                 sutehai = "Unknown"

#             # リストに追加
#             sutehai_info_list.append(sutehai) 

#     return [sutehai_info_list, sutehai_img]

# テンプレートマッチング処理関数
def recogPaiImage(paiImage, paiListImage, threshold = 0.6):
    # 雀牌表画像のグレースケール化
    paiListImage_gray = cv2.cvtColor(paiListImage, cv2.COLOR_BGR2GRAY)
    
    # 識別する雀牌画像のグレースケール化
    paiImage_gray = cv2.cvtColor(paiImage, cv2.COLOR_BGR2GRAY)

    # キャプチャ画像に対して、テンプレート画像との類似度を算出する
    res = cv2.matchTemplate(paiListImage_gray, paiImage_gray, cv2.TM_CCOEFF_NORMED)

    # 類似度の高い部分を検出する
    loc_candidate = np.where(res >= threshold)

    if len(loc_candidate[0]) == 0:
        return None

    # マッチング座標の中で最頻値座標を求める
    mode = []
    for loc_it in loc_candidate:
        unique, freq = np.unique(loc_it, return_counts=True)
        mode.append(unique[np.argmax(freq)])

    # 座標を元に牌の種類を識別する
    paiList = (
        ('Manzu1','Manzu2','Manzu3','Manzu4','Manzu5','Manzu6','Manzu7','Manzu8','Manzu9'),
        ('Pinzu1','Pinzu2','Pinzu3','Pinzu4','Pinzu5','Pinzu6','Pinzu7','Pinzu8','Pinzu9'),
        ('Sozu1','Sozu2','Sozu3','Sozu4','Sozu5','Sozu6','Sozu7','Sozu8','Sozu9'),
        ('Ton','Nan','Sya','Pe','Haku','Hatu','Tyun')
    )
    listHeight, listWidth = paiListImage.shape[:2]
    paiKind = int((mode[0]+listHeight/8)/(listHeight/4))
    paiNum = int((mode[1]+listWidth/18)/(listWidth/9))
    return paiList[paiKind][paiNum]

def calc_remaining_tiles(hand_tiles, dora_indicators, melded_blocks):
    # counts[0] ~ counts[34]: 各牌の残り枚数、counts[34] ~ counts[36]: 赤牌が残っているかどうか
    counts = [4 for _ in range(34)] + [1, 1, 1]
    meld_tiles = [tile for meld in melded_blocks for tile in meld["tiles"]]
    visible_tiles = hand_tiles + dora_indicators + meld_tiles

    for tile in visible_tiles:
        counts[tile] -= 1
        if tile == Tile.AkaManzu5:
            counts[Tile.Manzu5] -= 1
        elif tile == Tile.AkaPinzu5:
            counts[Tile.Pinzu5] -= 1
        elif tile == Tile.AkaSozu5:
            counts[Tile.Sozu5] -= 1

    return counts

def print_result(result):

    result_emit = []

    result_type = result["result_type"]  # 結果の種類
    syanten = result["syanten"]  # 向聴数
    time_us = result["time"]  # 計算時間 (マイクロ秒)

    # print(
    #     f"向聴数: {syanten['syanten']}"
    #     f" (通常手: {syanten['normal']}, 七対子手: {syanten['tiitoi']}, 国士無双手: {syanten['kokusi']})"
    # )
    # print(f"計算時間: {time_us / 1e6}秒")
    result_emit.append(f"向聴数: {syanten['syanten']}")
    result_emit.append(f" (通常手: {syanten['normal']}, 七対子手: {syanten['tiitoi']}, 国士無双手: {syanten['kokusi']})")
    result_emit.append(f"計算時間: {time_us / 1e6}秒")

    if result_type == 0:
        #
        # 手牌の枚数が13枚の場合、有効牌、期待値、和了確率、聴牌確率が得られる。
        #
        required_tiles = result["required_tiles"]  # 有効牌
        exp_values = result["exp_values"]  # 期待値 (1~17巡目)
        win_probs = result["win_probs"]  # 和了確率 (1~17巡目)
        tenpai_probs = result["tenpai_probs"]  # 聴牌確率 (1~17巡目)

        tiles = [f"{tile['tile']}: {tile['count']}枚" for tile in required_tiles]
        # print(f"  有効牌: {', '.join(tiles)}")
        result_emit.append(f"  有効牌: {', '.join(tiles)}")

        for turn, (exp, win_prop, tenpai_prop) in enumerate(
            zip(exp_values, win_probs, tenpai_probs), 1
        ):
            # print(
            #     f"  {turn}巡目 期待値: {exp:.0f}点, 和了確率: {win_prop:.1%}, 聴牌確率: {tenpai_prop:.1%}"
            # )
            result_emit.append(f"  {turn}巡目 期待値: {exp:.0f}点, 和了確率: {win_prop:.1%}, 聴牌確率: {tenpai_prop:.1%}")

    elif result_type == 1:
        #
        # 手牌の枚数が14枚の場合、打牌候補ごとに有効牌、期待値、和了確率、聴牌確率が得られる。
        #
        for candidate in result["candidates"]:
            tile = candidate["tile"]  # 打牌候補
            syanten_down = candidate["syanten_down"]  # 向聴戻しとなる打牌かどうか
            required_tiles = candidate["required_tiles"]  # 有効牌
            exp_values = candidate["exp_values"]  # 期待値 (1~17巡目)
            win_probs = candidate["win_probs"]  # 和了確率 (1~17巡目)
            tenpai_probs = candidate["tenpai_probs"]  # 聴牌確率 (1~17巡目)

            # print(f"打牌候補: {Tile.Name[tile]} (向聴落とし: {syanten_down})")
            result_emit.append(f"打牌候補: {Tile.Name[tile]} (向聴落とし: {syanten_down})")

            tiles = [f"{tile['tile']}: {tile['count']}枚" for tile in required_tiles]
            # print(f"  有効牌: {', '.join(tiles)}")
            result_emit.append(f"  有効牌: {', '.join(tiles)}")

            for turn, (exp, win_prop, tenpai_prop) in enumerate(
                zip(exp_values, win_probs, tenpai_probs), 1
            ):
                # print(
                #     f"  {turn}巡目 期待値: {exp:.0f}点, 和了確率: {win_prop:.1%}, 聴牌確率: {tenpai_prop:.1%}"
                # )
                result_emit.append(f"  {turn}巡目 期待値: {exp:.0f}点, 和了確率: {win_prop:.1%}, 聴牌確率: {tenpai_prop:.1%}")

    return result_emit

# 不要になったら消す
def create_sample_request1():
    ###########################
    # サンプル: 手牌が14枚で面前の場合
    # 例: 222567m34p33667s北
    ###########################
    # 手牌
    hand_tiles = [
        Tile.Manzu2,
        Tile.Manzu2,
        Tile.Manzu2,
        Tile.Manzu5,
        Tile.Manzu6,
        Tile.Manzu7,
        Tile.Pinzu3,
        Tile.Pinzu4,
        Tile.Sozu3,
        Tile.Sozu3,
        Tile.Sozu6,
        Tile.Sozu6,
        Tile.Sozu7,
        Tile.Pe,
    ]
    # 副露牌 (4個まで指定可能)
    melded_blocks = []

    # ドラ表示牌 (4枚まで指定可能)
    dora_indicators = [Tile.Ton]
    # 場風 (東: Tile.Ton, 南: Tile.Nan, 西: Tile.Sya, 北: Tile.Pe)
    bakaze = Tile.Ton
    # 自風 (東: Tile.Ton, 南: Tile.Nan, 西: Tile.Sya, 北: Tile.Pe)
    zikaze = Tile.Ton
    # 計算する向聴数の種類 (通常手: SyantenType.Normal, 七対子手: SyantenType.Tiitoi, 国士無双手: SyantenType.Kokusi)
    syanten_type = SyantenType.Normal
    # 現在の巡目 (1~17巡目の間で指定可能)
    turn = 3
    # 場に見えていない牌の枚数を計算する。
    counts = calc_remaining_tiles(hand_tiles, dora_indicators, melded_blocks)
    # その他、手牌とドラ表示牌以外に場に見えている牌がある場合、それらを引いておけば、山にないものとして計算できる。

    # 期待値を計算する際の設定 (有効にする設定を指定)
    exp_option = (
        ExpOption.CalcSyantenDown  # 向聴落とし考慮
        | ExpOption.CalcTegawari  # 手変わり考慮
        | ExpOption.CalcDoubleReach  # ダブル立直考慮
        | ExpOption.CalcIppatu  # 一発考慮
        | ExpOption.CalcHaiteitumo  # 海底撈月考慮
        | ExpOption.CalcUradora  # 裏ドラ考慮
        | ExpOption.CalcAkaTileTumo  # 赤牌自摸考慮
    )

    # リクエストデータを作成する。
    req_data = {
        "version": "0.9.0",
        "zikaze": bakaze,
        "bakaze": zikaze,
        "turn": turn,
        "syanten_type": syanten_type,
        "dora_indicators": dora_indicators,
        "flag": exp_option,
        "hand_tiles": hand_tiles,
        "melded_blocks": melded_blocks,
        "counts": counts,
    }

    return req_data

def remakeDataForJson(li):
    remakeData = []
    pai_dict = {
        'Manzu1':Tile.Manzu1,'Manzu2':Tile.Manzu2,'Manzu3':Tile.Manzu3,'Manzu4':Tile.Manzu4,'Manzu5':Tile.Manzu5,'Manzu6':Tile.Manzu6,'Manzu7':Tile.Manzu7,'Manzu8':Tile.Manzu8,'Manzu9':Tile.Manzu9,'AkaManzu5':Tile.AkaManzu5,
        'Pinzu1':Tile.Pinzu1,'Pinzu2':Tile.Pinzu2,'Pinzu3':Tile.Pinzu3,'Pinzu4':Tile.Pinzu4,'Pinzu5':Tile.Pinzu5,'Pinzu6':Tile.Pinzu6,'Pinzu7':Tile.Pinzu7,'Pinzu8':Tile.Pinzu8,'Pinzu9':Tile.Pinzu9,'AkaPinzu5':Tile.AkaPinzu5,
        'Sozu1':Tile.Sozu1,'Sozu2':Tile.Sozu2,'Sozu3':Tile.Sozu3,'Sozu4':Tile.Sozu4,'Sozu5':Tile.Sozu5,'Sozu6':Tile.Sozu6,'Sozu7':Tile.Sozu7,'Sozu8':Tile.Sozu8,'Sozu9':Tile.Sozu9,'AkaSozu5':Tile.AkaSozu5,
        'Ton':Tile.Ton,'Nan':Tile.Nan,'Sya':Tile.Sya,'Pe':Tile.Pe,'Haku':Tile.Haku,'Hatu':Tile.Hatu,'Tyun':Tile.Tyun
    }

    for i in li:
        remakeData.append(pai_dict[i])

    return remakeData

async def async_score_clac(doraList,tehai):
    emit('error_calc', {'error_calc': ""})
    ########################################
    # 計算実行
    ########################################

    doraData = [item for item in doraList if item != "Unknown"]
    dora_indicators = remakeDataForJson(doraData)

    hand_tiles = remakeDataForJson(tehai)

    melded_blocks = []

    counts = calc_remaining_tiles(hand_tiles, dora_indicators, melded_blocks)

    req_data = {
        "version": "0.9.0",
        "zikaze": Tile.Ton, 
        "bakaze": Tile.Ton, 
        "turn": turn,
        "syanten_type": syanten_Type,
        "dora_indicators": dora_indicators,
        "flag": consideration,
        "hand_tiles": hand_tiles,
        "melded_blocks": melded_blocks,
        "counts": counts,
    }
    # req_data = create_sample_request1()

    # dict -> json
    payload = json.dumps(req_data)
    # リクエストを送信する。
    res = requests.post(
        "http://localhost:8888", payload, headers={"Content-Type": "application/json"}
    )
    res_data = res.json()

    ########################################
    # 結果出力
    ########################################
    if not res_data["success"]:
        emit('error', {'error': f"計算の実行に失敗しました。(理由: {res_data['err_msg']})"})
        raise RuntimeError(f"計算の実行に失敗しました。(理由: {res_data['err_msg']})")
    
    result = res_data["response"]
    # result_emit = print_result(result)

    # emit('result', {'result': result_emit})
    emit('result', {'result': result})

##################################
        # 実行プログラム #
##################################

app = Flask(__name__, instance_relative_config=True)
socketio = SocketIO(app)

paiListPath = './static/data/template_images/paiList.png'
img_dir_name = "./static/data/jantama_capture"
dir_path = Path(img_dir_name)
dir_path.mkdir(parents=True, exist_ok=True)
os.makedirs(img_dir_name, exist_ok=True)

@app.route("/")
def index():
    return render_template('index.html')

@socketio.on('start_capture')
def window_capture():
    img_No = 0
    FPS = 14
    #繰り返しスクリーンショットを撮る
    with mss.mss() as sct:
        windows = gw.getWindowsWithTitle("雀魂-じゃんたま-")
        if not windows:
            emit('error', {'error': "雀魂を先に開いてください"})
            return
        else:
            emit('error', {'error': ""})
        
        #キャプチャスタート
        global capturing
        capturing = True

        global calc
        calc = False

        global syanten_Type
        global consideration
        global turn

        try:
            paiListImage = cv2.imread(paiListPath)
            emit('error', {'error': ""})
        except Exception as e:
            emit('error', {'error': "雀牌表画像の読み込みエラーです。"})
            print(e)

        window = windows[0]
        left, top, width, height = window.left, window.top, window.width, window.height
        monitor = {"top": top, "left": left, "width": width, "height": height}
        while capturing:
            emit('msg', {'msg': "Count:{}".format(img_No)})
            try:
                img_No = img_No + 1
                img = sct.grab(monitor)
                img = np.asarray(img)
                encoded_image = encode_image_to_base64(img)
                emit('new_image', {'img_path': f'data:image/jpeg;base64,{encoded_image}'})

                myHandImage, myTsumoImage, doraImage = cropMyHandImage(img)
                myHandImageList = divideMyHandImage(myHandImage)

                # for j in range(len(myHandImageList)):
                #     cv2.imwrite("{}/hand{}_{}.png".format(img_dir_name, img_No, j),myHandImageList[j])
                cv2.imwrite("{}/{}.png".format(img_dir_name, img_No),img)

                paiList = []
                for count in range(13):
                    response = recogPaiImage(myHandImageList[count], paiListImage)
                    if response:
                        paiList.append(response)
                    else:
                        paiList.append("Unknown")

                tsumo_response = recogPaiImage(myTsumoImage, paiListImage)
                if tsumo_response:
                    tsumopai = tsumo_response
                else:
                    tsumopai = "Unknown"

                # encoded_image2 = encode_image_to_base64(doraImage)
                # emit('new_image2', {'img_path2': f'data:image/jpeg;base64,{encoded_image2}'})
                doraImageList = divideDoraImage(doraImage)

                doraList = []
                for dora in doraImageList:
                    response = recogPaiImage(dora, paiListImage)
                    if response:
                        doraList.append(response)
                    else:
                        doraList.append("Unknown")

                tehai = list(paiList)
                if tsumopai != "Unknown":
                    tehai.append(tsumopai)

                # 非同期計算処理
                if calc and "Unknown" not in paiList :
                    asyncio.run(async_score_clac(doraList,tehai))
                else:
                    emit('error_calc', {'error_calc': "手牌が上手く読み込まれていません"})
                calc = False

                emit('tehaiList', {'tehaiList': paiList})
                emit('tsumohai', {'tsumohai': tsumopai})
                emit('doraList', {'doraList': doraList})

                socketio.sleep(1 / FPS)
            except Exception as e:
                emit('error', {'error': f"キャプチャエラー: {e}"})
                continue

@socketio.on('checkbox_change')
def handle_checkbox_change(data):
    global consideration
    consideration = sum(data["values"])

@socketio.on('number_input')
def handle_number_input(data):
    global turn
    turn = data['value']

@socketio.on('radio_change')
def handle_radio_change(data):
    global syanten_Type
    syanten_Type = data['value']

@socketio.on('stop_capture')
def capture_stop():
    global capturing
    capturing = False

@socketio.on('calc')
def calc_start():
    global calc
    calc = True

if __name__ == "__main__":
    socketio.run(app, host="127.0.0.1", port=5000, debug=True, allow_unsafe_werkzeug=True)
