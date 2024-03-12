import sys
import time

for i in range(10):
    LINE_UP = "\033[1F"
    LINE_CLEAR = "\x1b[2K"

    if i != 0:
        for x in range(6):
            sys.stdout.write(LINE_UP + LINE_CLEAR)

    print(
        f"Current State\n-------------\nEpoch: {i+1}/{i} \nBatch: {i+1}/{i} \nCurrent Loss: {i} \nPercentage: {i:.2f}% \nTime Remaining: {i}",
        end="\r",
    )

    time.sleep(0.5)
