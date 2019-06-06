"""
Transpiled into Python 3 by Jungwoo Nam (2019-05-20)
========================Original Tag=========================
작성일          : 2017-04-16
작성자          : 윤종섭
개요            : 자동차를 촬영한 사진으로부터 자동차 번호판을 추출
개발툴          : Visual Studio 2015
외부 라이브러리 : OpenCV 3.2.0
=============================================================
- 프로그램 설명 -
자동차 전면부를 촬영한 사진을 이진화 시키고, 이진화 영상으로부터 Contours를 추출,
각 Contours를 감싸는 사각형을 얻은 후 사각형의 가로,세로 길이 값을 관찰해
번호판 여부를 판단하여 번호판 영상을 얻는다.

번호판 추출의 처리 절차는 다음과 같다.

1. 자동차 촬영 사진 입력
2. BGR to GRAY 변환
3. 3x3 박스필터를 이용한 블러링(잡음이 Contour에 영향을 주기 때문에 시행)
4. 블러된 영상 이진화(임계값은 경험적으로 얻은 150을 사용)
5. Contours 탐색
6. 각 Contours를 감싸는 사각형 탐색(RotatedRect 클래스 객체)
7. 6에서 얻은 회전된 사각형을 Rect클래스로 변환(회전하지 않은 직사각형 정보를 저장)
8. 7에서 얻은 Rect 객체들의 정보를 이용하여 번호판 여부를 판단한다.
"""

import sys

import numpy as np
import cv2 as cv
import os

# 히스토그램에서 최댓값을 갖는 부분을 가져오는 함수


def getHistMax(img):
    hist = cv.calcHist([img], [0], None, [10], [0, 255])
    hist = np.resize(hist, [1, 10])[0]
    return np.argmax(hist)


# 번호판 영억을 검출 및 표시하는 함수
def getNumberPlate(image_path):

    # 이미지 열기
    with open(image_path, 'rb') as f:
        data = bytearray(f.read())
        array = np.asarray(data, np.uint8)
        src = cv.imdecode(array, cv.IMREAD_UNCHANGED)
    if src is None:
        print('src is empty!', file=sys.stderr)
        exit(-1)

    # 이미지 가로 크기 고정
    width = 1000
    height = int(src.shape[0] * width/src.shape[1])
    src = cv.resize(src, (width, height))

    # 2. BGR to GRAY 변환
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # 3. 3x3 박스필터를 이용한 블러링
    src_gray = cv.blur(src_gray, (3, 3))

    # Adaptive threshold를 이용하여 이진화
    threshold_output = cv.adaptiveThreshold(
        src=src_gray,
        maxValue=255,
        adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv.THRESH_BINARY,
        blockSize=7,
        C=2)

    # 5. contours 탐색
    contours, hierarchy = cv.findContours(
        threshold_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]

    # 6. 각 contours를 감싸는 사각형 탐색
    # minAreaRect -> rect : (center(x, y), (width, height), angle of rotation)
    rotated_rects = [cv.minAreaRect(cnt) for cnt in contours]

    # 7. 6에서 얻은 회전된 사각형을 Rect로 변환 (angle = 0으로 변환)
    rects = [cv.boundingRect(cv.boxPoints((center, size, angle)))
             for center, size, angle in rotated_rects]

    # 8. 7에서 얻은 Rect들의 정보를 이용하여 번호판 여부를 판단

    # 비사업용 보통 등록 번호판 규격 (520mm x 110mm)
    numberplate_ratio = 520 / 110

    # 예상 후보영역의 리스트
    expect = []

    # hier : (next, prev, child, parent). (-1) means there's nothing.
    for (x, y, w, h) in rects:
        rect_ratio = w / h

        # 비율 검사 - 3.2, 5.5는 경험적으로 얻은 값
        if 3.2 <= rect_ratio <= 5.5:
            # 너무 크거나 작은 영역 제외
            if w < width*.9 and h > 5:
                # 밝기 검사 - 중간값보다 밝아야 함.
                if getHistMax(src_gray[y:y+h, x:x+w, ]) >= 4:
                    # 크기 검사 - 넓이가 2000px이상 되어야 함.
                    if w*h > 2000:
                        # x,y,w,h,f 배열 저장 f는 적합도이다. 적합도는 클수록(음수이므로 0에 가까울수록)좋다.
                        expect.append(
                            [x, y, w, h, -abs(1-rect_ratio/numberplate_ratio)])

    # 적합도 순으로 정렬한다.
    def fitness(item):
        return item[4]
    expect = sorted(expect, key=fitness)

    # 적합도 순으로 정렬된 번호판 후보 배열 출력
    # 인덱스가 작을수록 적합도가 작다. 즉, 리스트의 가장 끝 요소가 가장 잘 검출된 번호판이다.
    count = 1
    print(image_path)
    for i in expect:
        print(count, i)
        count += 1
    print("\n\n")

    # 번호판 영역 드로우. 적합도가 작은 영역부터 그려진다.
    font = cv.FONT_HERSHEY_SIMPLEX
    count = len(expect)
    for (x, y, w, h, f) in expect:

        # 영역은 초록색으로 그리되, 가장 적합한 세 개의 후보는 붉은색으로 그린다.
        thickness = 3
        color = (0, 255,  0)
        if count < 4:
            thickness = 4
            color = (0, 0, 255)
        cv.rectangle(src, (x, y), (x + w, y + h), color, thickness)
        cv.putText(src, str(count), (x, y), font, 1, color, 2, cv.LINE_AA)
        count -= 1

    # 영역을 출력한다.
    cv.imshow('Detected area', src)

    # 사용자가 키를 누를 때가지 대기하고, 키를 누르면 화면을 없엔다.
    cv.waitKey()
    cv.destroyAllWindows()


# 현재 디렉토리 아래에 있는 car 디렉토리에서 파일을 하나씩 읽는다.
for i in os.listdir('./car'):
    image_path = './car/'+i
    getNumberPlate(image_path)
