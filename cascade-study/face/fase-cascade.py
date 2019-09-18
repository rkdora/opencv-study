import cv2
import matplotlib.pyplot as plt

# 画像読込み
origin_img = cv2.imread("free.png")
# 画像コピー
img = origin_img.copy()

# カスケードファイルのパス
cascade_path = "haarcascade_frontalface_alt.xml"
# カスケード分類器の特徴量取得
cascade = cv2.CascadeClassifier(cascade_path)

# 画像グレースケール化
grayscale_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# 顔検出
# minSize 最小サイズ指定
front_face_list = cascade.detectMultiScale(grayscale_img, minSize = (100, 100))

# 検出判定
print(front_face_list)
if len(front_face_list) == 0:
    print("Failed")
    quit()

# 検出位置描画
for (x,y,w,h) in front_face_list:
    print("[x,y] = %d,%d [w,h] = %d,%d" %(x, y, w, h))
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), thickness=10)

# 顔検出画像表示
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
# 顔検出画像出力
cv2.imwrite("out.jpg", img)

# 検出画像出力
for (x,y,w,h) in front_face_list:
    face_img = origin_img[y:y+h, x:x+w]
    filename = "face_" + str(x) + "-" + str(y) + ".jpg"
    cv2.imwrite(filename, face_img)
