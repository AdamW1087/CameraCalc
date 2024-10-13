import cv2
import mediapipe as mp
import csv

from model import set_up_data
from process import normalise_path, process_path


def main():
    # setting up modes and variables
    write = False
    getData = False
    guessInput = False
    labelGuess = ""
    csv_path = 'src/point_history.csv'
    expression = ""
    answer = False
    
    # current path is used to separate lines shown to users
    current_path = []
    paths = []
    

    
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
                current_path = []
                paths = []
                labelGuess = ""
                
            # enter calculator mode
            case 99:  # ord('c')
                guessInput = not guessInput
                getData = False
                
            # enter data input mode
            case 105:  # ord('i')
                getData = not getData
                guessInput = False
                
            # start/stop writing
            case 32:   # ord(' ')
                write = not write
                # add new path to list of paths
                if not write and current_path:
                    paths.append(current_path)
                    current_path = []
                    
                if not write and guessInput:
                     # normalises data and updates format, flattens paths
                    processed_path = process_path(sum(paths, []))
                    
                    # use data to predict input       
                    labelGuess = nn.predictSingle(processed_path, 1)
                    
            # enter guess
            case 13:  # enter key
                # reset answer
                if answer:
                    answer = False
                    expression = ""
                    
                # evaluate expr
                elif guessInput and not paths:
                    expression = str(eval(expression))
                    answer = True
                    
                # add new input to expression and reset current label and path
                elif guessInput:
                    expression += labelGuess
                    labelGuess = ""
                    paths = []

                    
            # get input to attach label to new data
            # ord('0') to ord('9') and +, -, *, /
            case key if (48 <= key <= 57) or key in {43, 45, 42, 47}:  
                if getData and not write:
                    # get input as label
                    if 48 <= key <= 57:  # 0-9
                        label = chr(key)
                    elif key == 43:  # ord('+')
                        label = '+'
                    elif key == 45:  # ord('-')
                        label = '-'
                    elif key == 42:  # ord('*')
                        label = '*'
                    elif key == 47:  # ord('/')
                        label = '/'

                    # normalise path data to be inputted
                    normalised_path = normalise_path(sum(paths, []))

                    # write label and normalised path to file
                    with open(csv_path, 'a', newline="") as file:
                        writer = csv.writer(file)

                        # write data as [label, [[x1,y1], [x2,y2]...] ]
                        writer.writerow([label, *normalised_path])
                        paths = []

            
        # undo camera mirror for more natural feed
        frame = cv2.flip(frame, 1)
    
        # show the user mode
        mode_type = ""
        if not write and not guessInput and not getData:
            mode_type = "Press C to enter Calculator Mode, or press I to enter Input Mode"
        elif not write and guessInput:
            mode_type = "In Calculator Mode, Space to draw, R to reset, Enter to input/calculate"
        elif write and guessInput:
            mode_type = "In Guessing Mode, press Space to stop drawing"
        elif not write and getData:
            mode_type = "In Input Mode, press the key you have drawn to enter it, Space to draw, R to reset"
        elif write and getData:
            mode_type = "In Input Mode, press Space to stop drawing"
        
        # show mode info
        cv2.putText(frame, mode_type, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
           0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        
        # show the predicted label
        cv2.putText(frame, "Are you inputting a " + labelGuess if guessInput and labelGuess and not write else "", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, (255, 255, 255), 1, cv2.LINE_AA)

        # show the expression
        if guessInput:
            if answer: 
                cv2.putText(frame, "Result is: " + expression + ". Press enter to continue", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Current expression is: " + expression if guessInput else "", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 1, cv2.LINE_AA)

        
    
        # process the frame to get hand landmarks
        result = hands.process(frame)
        if write:
            if result.multi_hand_landmarks:
                
                for hand_landmarks in result.multi_hand_landmarks:
                    # filter hand points for the index finger
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    
                    # add coords of index to path
                    path.append([index_finger_tip.x, index_finger_tip.y])
                    
        all_paths = paths + [current_path]
        for path in all_paths:
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