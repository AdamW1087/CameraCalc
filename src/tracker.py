import cv2
import mediapipe as mp
import csv

from model import set_up_data
from process import normalise_path, process_path


def main():
    # setting up modes and variables
    write = False
    getData = False
    guessNumber = False
    labelGuess = ""
    path = []
    csv_path = 'src/point_history.csv'
    
    # set up nn model
    nn = set_up_data(csv_path)

    # setting up hand gesture recogniser
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    
    # start video feed
    cap = cv2.VideoCapture(0)
    
    
    while cap.isOpened():
        # get if frame is valid and frame
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # check for any user input
        key = cv2.pollKey()
        match key:
            # quit
            case 113:  # ord('q')
                break
            
            # reset path
            case 114:  # ord('r')
                path = []
                
            # enter guessing mode
            case 103:  # ord('g')
                guessNumber = not guessNumber
                
            # start/stop writing
            case 32:   # ord(' ')
                write = not write
                
            # enter data input mode
            case 105:  # ord('i')
                getData = not getData
                
            # use current path to predict input
            case 112:  # ord('p')
                if guessNumber:
                    # normalises data and updates format
                    processed_path = process_path(path)
                    
                    # use data to predict input
                    labelGuess = nn.predictSingle(processed_path, 1)
                    path = []
                    
            # get input to attach label to new data
            case key if 48 <= key <= 57:  # ord('0') to ord('9')
                if getData and not write:
                    label = chr(key)
                    
                    # normalise path data to be inputted
                    normalised_path = normalise_path(path)

                    # write label and normalised path to file
                    with open(csv_path, 'a', newline="") as file:
                        writer = csv.writer(file)

                        # write data as [label, [[x1,y1], [x2,y2]...] ]
                        writer.writerow([label, *normalised_path])
                        path = []

            
        # undo camera mirror for more natural feed
        frame = cv2.flip(frame, 1)
    
        # show if the user is in writing mode or not
        cv2.putText(frame, "Writing, press space to lift the pen" if write else "Press W to save or Space to continue", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
           0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # show if user is in inputting data mode
        cv2.putText(frame, "Entering data" if getData else "", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
           0.5, (255, 255, 255), 1, cv2.LINE_AA)        
        
        # show if user is in guessing mode
        cv2.putText(frame, "Guessing" if guessNumber else "", (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
           0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # show the predicted label
        cv2.putText(frame, "I think it is a " + labelGuess if guessNumber and labelGuess else "", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 
            1, (255, 255, 255), 1, cv2.LINE_AA)
        
    
        # process the frame to get hand landmarks
        result = hands.process(frame)
        if write:
            if result.multi_hand_landmarks:
                
                for hand_landmarks in result.multi_hand_landmarks:
                    # filter hand points for the index finger
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    
                    # add coords of index to path
                    path.append([index_finger_tip.x, index_finger_tip.y])
                    
        for i in range(1, len(path)):
            # calculate newest points in path
            prev_point = (int(path[i - 1][0] * frame.shape[1]), int(path[i - 1][1] * frame.shape[0]))
            current_point = (int(path[i][0] * frame.shape[1]), int(path[i][1] * frame.shape[0]))
            
            # draw line (thin white with black border) to show user where they have drew
            cv2.line(frame, prev_point, current_point, (0, 0, 0), 3)
            cv2.line(frame, prev_point, current_point, (255, 255, 255), 2)

    
        # show updated frame with info
        cv2.imshow('Hand Gesture Recognition', frame)
    
    
    
    # stop video feed and close the window
    cap.release()
    cv2.destroyAllWindows()

        
if __name__ == '__main__':
    main()