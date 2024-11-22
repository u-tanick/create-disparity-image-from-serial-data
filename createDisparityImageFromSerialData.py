import serial
import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor, Resize

# ---------------------------------------------------------------------
# for Stereo Image

# MiDaSモデルのロード
def load_midas_model():
    print("Loading MiDaS model...")
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.eval()
    transform = Compose([ToTensor()])
    return model, transform

# 入力画像をMiDaSに適したサイズに変換する
def prepare_image(img, midas_transform):
    target_size = (256, 256)  # MiDaS推奨サイズ
    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    input_tensor = midas_transform(resized_img).unsqueeze(0)
    return input_tensor, resized_img

# 視差画像の生成
def create_disparity_images(img, depth_map):
    h, w = depth_map.shape
    left_image = img.copy()
    right_image = img.copy()

    # 深度に基づいて左右にシフト
    for y in range(h):
        for x in range(w):
            shift = int(depth_map[y, x] * 10)  # 深度に基づくシフト量を計算
            if x + shift < w:
                right_image[y, x] = img[y, x + shift]
            if x - shift >= 0:
                left_image[y, x] = img[y, x - shift]

    # 左右画像を結合 (SBS形式)
    sbs_image = np.concatenate((left_image, right_image), axis=1)
    return sbs_image

# ---------------------------------------------------------------------
# for Serial Communication

# シリアルポートの設定
COM_PORT = 'COM5'
BAUD_RATE = 115200

# JPEGデータの境界
JPEG_START = b'\xff\xd8'  # JPEGヘッダー (開始)
JPEG_END = b'\xff\xd9'    # JPEGフッター (終了)

def read_image_from_serial(ser):
    buffer = bytearray()
    timeout_counter = 0  # タイムアウト用カウンタ
    MAX_TIMEOUT = 100  # タイムアウト制限回数

    while True:
        # 1回分のデータを読み取る
        data = ser.read(256)
        if data:
            buffer.extend(data)
            timeout_counter = 0  # データを受信した場合、タイムアウトカウンタをリセット
        else:
            timeout_counter += 1
            if timeout_counter > MAX_TIMEOUT:
                raise TimeoutError("Timed out waiting for image data.")

        # バッファ内でJPEG開始と終了を探す
        start_idx = buffer.find(JPEG_START)
        end_idx = buffer.find(JPEG_END, start_idx)
        if start_idx != -1 and end_idx != -1:
            # JPEGデータを抽出
            jpg_data = buffer[start_idx:end_idx + len(JPEG_END)]
            buffer = buffer[end_idx + len(JPEG_END):]  # 残りをバッファに保存

            image = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)

            return image

# ---------------------------------------------------------------------
# for mian

# 視差画像を表示するウィンドウ名
WINDOW_NAME = 'Disparity Camera Image'
# デスクトップ上に表示する際の解像度（5インチディスプレイの解像度にセット）
NEW_WIDTH  = 1920
NEW_HEIGHT = 1080

def main():

    # MiDaSモデルのロード
    midas_model, midas_transform = load_midas_model()

    with serial.Serial(COM_PORT, BAUD_RATE, timeout=2) as ser:
        print(f"Connected to {COM_PORT} at {BAUD_RATE} baud.")
        while True:
            try:
                image = read_image_from_serial(ser)
                if image is not None:

                    # 画像をリサイズ
                    input_tensor, resized_image = prepare_image(image, midas_transform)

                    # 深度推定
                    with torch.no_grad():
                        depth_map = midas_model(input_tensor).squeeze().numpy()

                    # 深度マップのサイズを元画像サイズにリサイズ
                    depth_map_resized = cv2.resize(depth_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

                    # 視差画像生成
                    disparity_image = create_disparity_images(image, depth_map_resized)

                    # 画像をリサイズ
                    resized_image = cv2.resize(disparity_image, (NEW_WIDTH, NEW_HEIGHT))

                    # リサイズした画像を表示
                    cv2.imshow(WINDOW_NAME, resized_image)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            except Exception as e:
                print(f"Error: {e}")
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
