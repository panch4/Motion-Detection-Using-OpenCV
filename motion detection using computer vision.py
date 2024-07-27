#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Loop through frames
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Apply background subtraction
    fgmask = fgbg.apply(frame)
    
    # Apply thresholding to remove noise
    th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]
    
    # Find contours of objects
    contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Loop through detected objects
    for contour in contours:
        # Calculate area of object
        area = cv2.contourArea(contour)
        
        # Filter out small objects
        if area > 100:
            # Get bounding box coordinates
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw bounding box around object
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:




