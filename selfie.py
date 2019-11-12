import sys
import traceback
import tellopy
import av # av는 환경이 꼬여서 conda prompt에서만 실행됨
import cv2.cv2 as cv  # for avoidance of pylint error
import numpy
import time
import keyboard

# 얼굴 검출용 xml 파일
face_cascade = cv.CascadeClassifier('./xml_file/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('./xml_file/haarcascade_eye.xml')

def face_detection():
    cap = cv.VideoCapture(0)
    while(1):
        ret, img = cap.read()



        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        
        for (x,y,w,h) in faces:
            img = cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
        cv.imshow('face_detection', img)        
        k = cv.waitKey(30)
        if k == ord('s'):
            break
                
    cap.release()
    cv.destroyAllWindows()



    

# 황금 비율 촬영 모드
# 얼굴과 몸의 비율이 가장 좋을 때 촬영! ex) 8등신
def mode_t():       
    #drone = tellopy.Tello()

    try:
        #drone.connect()
        # 15초간 드론 연결을 대기.. 초과되면 프로그램 종료
        #drone.wait_for_connection(15.0)
        print("드론 연결 성공.")
        #drone.takeoff()
        #time.sleep(4)
        print("드론 이륙 성공.")
        
        retry = 3
        container = None
        #while container is None and 0 < retry:
        #    retry -= 1
        #    try:
        #        container = av.open(drone.get_video_stream())
        #    except av.AVError as ave:
        #        print(ave)
        #        print('av.open err retry...')

        # 첫 420 frame은 연결 문제가 있어서 생략
        frame_skip = 420
        while True:
            for frame in container.decode(video=0):
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
                # 촬영 시작 시간
                start_time = time.time()
                # 드론이 촬영한 영상을 opencv형식으로 변환
                img = cv.cvtColor(numpy.array(frame.to_image()), cv.COLOR_RGB2BGR)
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                #cv.imshow('drone cam', img)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                        
                for (x, y, w, h) in faces:
                    img = cv.rectangle(img, (x, y),(x + w, y + h), (255, 0, 0), 2)
                    roi_gray = gray[y : y + h, x : x + w]
                    roi_color = img[y : y + h, x : x + w]
                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    for (ex, ey, ew, eh) in eyes:
                        cv.rectangle(roi_color, (ex, ey),(ex + ew, ey + eh), (0, 255, 0), 2)

                cv.imshow('face_detection', img)        
                k = cv.waitKey(1)
                if k == ord('s'):
                    break
                if frame.time_base < 1.0/60:
                    time_base = 1.0/60
                else:
                    time_base = frame.time_base
                frame_skip = int((time.time() - start_time)/time_base)
                    
    # 모든 에러에 대응
    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
        print('프로그램을 종료합니다.')
        drone.quit()
        cv.destroyAllWindows()
    #finally:
    #    drone.quit()
    #    cv.destroyAllWindows()






def main():
    choice ="""
=============================================================================
                                SELFIE DRONE    
=============================================================================

                        촬영할 영상의 형태를 선택하세요.
                        
------------------------------------------------------------------------------      

                        'f' : 상반신 + 배경의 정면 사진

                        'b' : 상반신+ 배경의 후면 사진

                        't' : 본인의 황금 비율 사진

                        'r' : 본인의 360° 회전 영상

                        'h' : 공중 10m 사진

                        'i' : 공중 10m 영상 

------------------------------------------------------------------------------                                        
YOUR CHOICE >>"""
    print(choice, end='')
    
    choice_msg = None
    mode = None
    while True:  
            if keyboard.is_pressed('f') or keyboard.is_pressed('F'):
                pass
            elif keyboard.is_pressed('b') or keyboard.is_pressed('B'):
                pass
            elif keyboard.is_pressed('t') or keyboard.is_pressed('T'):
                choice_msg = "황금 비율"
                mode = 't'
                print(f"{choice_msg} 모드를 선택했습니다.")
                break
            else:
                continue
    
    msg = """
안전한 공간을 확보하고 드론을 위치시키세요.
준비 후, 's' 를 누르면 이륙 후 촬영을 시작합니다.
YOUR COMMAND >>"""
    print(msg, end='')
    while True:  
            if keyboard.is_pressed('s') or keyboard.is_pressed('S'):
                mode_t()
            else:
                continue

if __name__ == '__main__':
    main()
