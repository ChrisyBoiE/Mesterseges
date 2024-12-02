import cv2

def video_informacio(video):
    szelesseg = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    magassag = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    osszes_kocka = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    return szelesseg, magassag, fps, osszes_kocka
