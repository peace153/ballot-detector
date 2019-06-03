import os
import cv2
import numpy as np

# function to check if the card is totally in the screen or not
# now return boolean, should return boolean and card picture to send to detect_x
def detect_card():
    found_card = input("Found Card(1=true,other=false) ?:")
    return found_card == "1"


# function to detect result
# step : 
#   - detect x
#   - get the party no of the chosen one
#   - return party no, or 0 if it's a voided ballot
def detect_x():
    res = 0
    input_party = input("which party no.? (1-20, type any text for voided ballot card)")
    try:
        res = int(input_party)
        if res > 20:
            res = 0
    except ValueError:
        res = 0
    print(res)
    return res



# main section

#state 
# 0 = waiting for card
# 1 = waiting card to move out
state = 0

# vote count for each party no., 
# party_count[0] = count of voided ballot card
party_count = [0]*21

print(party_count)

while True:
    # waiting for card
    if state == 0:
        if detect_card():
            print("found card")
            # if found card, switch to state 1, 
            state = 1
            # get vote result
            party = detect_x()
            party_count[party] = party_count[party]+1
            print(party_count)
            
    # waiting card to move out
    elif state == 1:
        if detect_card() != True:
            print("card already move out")
            # if not found card (card already move out), switch to state 0 and wait for new card.
            state = 0


            


